import os
import time
from typing import Dict, Any, Optional

import torch
import torch.distributed as dist

# Deprecated helper imports kept for compatibility.
import numpy as np
from PIL import Image
from torchvision import transforms

PARALLEL_GROUP_CACHE = {}


class AdaptiveKLController:
    def __init__(self, target_kl=0.01, init_kl_coef=0.0001, min_kl_coef=0.0001, max_kl_coef=0.01, horizon=100):
        self.target_kl = target_kl
        self.kl_coef = init_kl_coef
        self.min_kl_coef = min_kl_coef
        self.max_kl_coef = max_kl_coef
        self.horizon = horizon
        self.kl_history = []
        self.step_count = 0
    
    def update(self, current_kl):
        self.step_count += 1
        self.kl_history.append(current_kl.item() if torch.is_tensor(current_kl) else current_kl)
        if len(self.kl_history) > self.horizon:
            self.kl_history.pop(0)
        
        # Gradually increase KL weight from small to large
        if self.step_count < 100:
            # First 100 steps: gradually increase from init to base
            progress = self.step_count / 100.0
            self.kl_coef = self.min_kl_coef + (self.target_kl - self.min_kl_coef) * progress
        else:
            # After 100 steps: use PID-style adjustment
            if len(self.kl_history) >= 10:
                avg_kl = np.mean(self.kl_history[-10:])
                kl_error = avg_kl - self.target_kl
                
                # Conservative adjustment
                if kl_error > 0.5 * self.target_kl:  # KL too large
                    self.kl_coef *= 0.9
                elif kl_error < -0.5 * self.target_kl:  # KL too small  
                    self.kl_coef *= 1.1
                
                self.kl_coef = max(self.min_kl_coef, min(self.max_kl_coef, self.kl_coef))
        
        return self.kl_coef


def get_parallel_groups(num_generations, sp_size, world_size):
    """
    Cache NCCL process groups so we do not recreate them every training step.
    """
    if not dist.is_initialized():
        raise RuntimeError("Distributed process group must be initialized before creating subgroups.")
    cache_key = (num_generations, sp_size, world_size)
    cached = PARALLEL_GROUP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    dp_world_size = world_size // sp_size
    if dp_world_size % num_generations != 0:
        raise ValueError(
            f"dp_world_size ({dp_world_size}) must be divisible by num_generations ({num_generations})."
        )
    num_groups = dp_world_size // num_generations

    all_group_process_groups = []
    all_gather_groups = []
    all_group_leader_ranks = []

    # Keep creation order in sync across ranks
    dist.barrier()
    for g_idx in range(num_groups):
        g_dp_ranks = list(range(g_idx * num_generations, (g_idx + 1) * num_generations))
        g_all_global_ranks = []
        for dp_r in g_dp_ranks:
            g_all_global_ranks.extend([dp_r * sp_size + sp_r for sp_r in range(sp_size)])

        g_leader_rank = g_idx * num_generations * sp_size
        g_gather_ranks = [dp_r * sp_size for dp_r in g_dp_ranks]

        all_group_leader_ranks.append(g_leader_rank)
        all_group_process_groups.append(dist.new_group(ranks=g_all_global_ranks))
        all_gather_groups.append(dist.new_group(ranks=g_gather_ranks))

    PARALLEL_GROUP_CACHE[cache_key] = (
        all_group_process_groups,
        all_gather_groups,
        all_group_leader_ranks,
    )
    return PARALLEL_GROUP_CACHE[cache_key]


def sync_cuda_time(sync=True, barrier=True):
    # For accurate time measurement
    if barrier:
        dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])
    if sync:
        torch.cuda.synchronize()
    t = time.time()
    return t


def nanstd(tensor: torch.Tensor) -> float:
    finite = tensor[torch.isfinite(tensor)]
    if finite.numel() == 0:
        return float("nan")
    return finite.std(unbiased=False).item()


def gather_tensor(tensor):
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)


def maybe_update_ref_model_for_kl(args, model, ref_model, scalar_states, logger):
    """
    When using moving KL, refresh the ref_model weights with the current model weights at a fixed step interval.
    - The default ref_model is a copy of the model at step0 (loaded from checkpoint), used for Fixed KL
    - If use_moving_KL=True, refresh the ref_model with the current model every args.update_ref_model_step optimizer update steps
    - If use_moving_KL=False, the ref_model remains the initial model (Fixed KL mode)
    """
    if not getattr(args, "use_moving_KL", False):
        return ref_model

    step_interval = int(getattr(args, "update_ref_model_step", 0) or 0)
    if step_interval <= 0:
        return ref_model

    update_steps = getattr(scalar_states, "update_steps", 0)
    if update_steps == 0 or update_steps % step_interval != 0:
        return ref_model

    if ref_model is None:
        if dist.get_rank() == 0 and logger is not None:
            logger.error("[Moving KL] ref_model is None when trying to update; please ensure it is initialized.")
        return ref_model

    ref_model = model
    ref_model.to("cpu")
    torch.cuda.empty_cache()
    logger.info(f"[Moving KL] Updated ref_model at update_step={update_steps}")

    return ref_model


def _batch_extra_kwargs(extra_kwargs_list):
    """
    Batch a list of per-sample extra_kwargs dicts into a single dict suitable for mini-batch processing.

    For each key:
      - If all values are tensors (and not None), stack them along batch dimension.
      - If all values are None, skip that key.
      - Otherwise, keep the list of values as-is.
    """
    if extra_kwargs_list is None:
        return {}

    # Ensure we are working with a list
    if not isinstance(extra_kwargs_list, (list, tuple)):
        return extra_kwargs_list

    if len(extra_kwargs_list) == 0:
        return {}

    # If elements are not dicts, return as-is (fallback)
    if not isinstance(extra_kwargs_list[0], dict):
        return extra_kwargs_list

    batched = {}
    # Collect all keys across dicts
    all_keys = set().union(*(d.keys() for d in extra_kwargs_list if isinstance(d, dict)))

    for key in all_keys:
        vals = []
        for d in extra_kwargs_list:
            if isinstance(d, dict):
                vals.append(d.get(key, None))
            else:
                vals.append(None)

        # All None -> skip
        if all(v is None for v in vals):
            continue

        # If all are tensors, stack into a batch
        if all(isinstance(v, torch.Tensor) for v in vals):
            batched[key] = torch.stack(vals, dim=0)
        else:
            # Mixed types: keep as list for safety
            batched[key] = vals

    return batched


def _sync_random_tensor(
    generator_fn,
    shape,
    dtype=torch.long,
    device=None,
    nccl_info=None,
):
    """
    Generate a random tensor and sync across SP group if needed.

    Args:
        generator_fn: Function that generates the random tensor (e.g., lambda: torch.randperm(n))
        shape: Shape tuple for the tensor (e.g., (n,) for 1D, (m, n) for 2D)
        dtype: Data type of the tensor
        device: Device for the tensor
        nccl_info: NCCL info object with sp_group and rank_within_spgroup

    Returns:
        Synchronized random tensor
    """
    if nccl_info is not None and nccl_info.sp_group is not None:
        if nccl_info.rank_within_spgroup == 0:
            tensor = generator_fn()
        else:
            tensor = torch.zeros(shape, dtype=dtype, device=device)
        sp_leader_rank = dist.get_process_group_ranks(nccl_info.sp_group)[0]
        dist.broadcast(tensor, src=sp_leader_rank, group=nccl_info.sp_group)
    else:
        tensor = generator_fn()

    return tensor


def _normalize_group(
    rewards: torch.Tensor,
    ranks_per_group: int = 1,
    num_groups: int = 1,
    samples_per_grp: int = None,
) -> torch.Tensor:
    """
    Normalize rewards to compute advantages, with support for both single-rank and multi-rank groups.

    Args:
        rewards: Reward tensor to normalize
        ranks_per_group: Number of ranks per group (1 for single-rank, >1 for multi-rank)
        num_groups: Number of groups per rank (only used when ranks_per_group == 1)
        samples_per_grp: Number of samples per group (only used when ranks_per_group == 1)

    Returns:
        Normalized advantages tensor with same shape as rewards
    """
    advantages = torch.full_like(rewards, float("nan"))

    def _normalize_values(values: torch.Tensor) -> torch.Tensor:
        """Helper function to normalize a tensor with support for 1D and 2D tensors."""
        result = torch.full_like(values, float("nan"))
        finite_mask = torch.isfinite(values)

        if not finite_mask.any():
            return result

        if values.ndim == 1:
            # 1D case: simple normalization across all values
            valid = values[finite_mask]
            mean = valid.mean()
            std = torch.clamp(valid.std(unbiased=False), min=1e-6)
            result[finite_mask] = (valid - mean) / std
        elif values.ndim == 2:
            # 2D case: normalize along axis 0 (across samples), keep timestep dimension
            mean = values.mean(dim=0, keepdim=True)
            std = torch.clamp(values.std(dim=0, keepdim=True, unbiased=False), min=1e-6)
            result = (values - mean) / std
            result[~finite_mask] = float("nan")
        else:
            # Fallback for other dimensions
            valid = values[finite_mask]
            mean = valid.mean()
            std = torch.clamp(valid.std(unbiased=False), min=1e-6)
            result[finite_mask] = (valid - mean) / std

        return result

    if ranks_per_group == 1:
        # Single-rank groups: normalize each group separately
        if samples_per_grp is None:
            samples_per_grp = len(rewards) // num_groups if num_groups > 0 else len(rewards)

        for grp_idx in range(num_groups):
            start_idx = grp_idx * samples_per_grp
            end_idx = start_idx + samples_per_grp
            slice_obj = slice(start_idx, min(end_idx, len(rewards)))
            if slice_obj.start >= slice_obj.stop:
                continue
            advantages[slice_obj] = _normalize_values(rewards[slice_obj])
    else:
        # Multi-rank groups: rewards already gathered for the full group, normalize all together
        advantages = _normalize_values(rewards)

    return advantages

