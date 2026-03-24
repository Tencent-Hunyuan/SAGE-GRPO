import gc
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
from PIL import Image
from tqdm import tqdm

from hyvideo.schedulers.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from hyvideo.commons.grpo_commons import (
    _batch_extra_kwargs,
    _normalize_group,
    _sync_random_tensor,
    gather_tensor,
    get_parallel_groups,
    sync_cuda_time,
    nanstd,
    AdaptiveKLController,
)
from hyvideo.pipelines.hunyuan_video_grpo_pipeline import HunyuanVideoGRPOPipeline
from hyvideo.utils.parallel_states import nccl_info
from hyvideo.utils.file_utils import convert_to_json_serializable, save_videos_grid

_SP_GROUP_CACHE = {}


def _load_reference_image(ref_image_path: Optional[str]) -> Optional[Image.Image]:
    if isinstance(ref_image_path, str) and ref_image_path and os.path.exists(ref_image_path):
        return Image.open(ref_image_path).convert("RGB")
    return None


def _resolve_target_resolution(pipeline: HunyuanVideoGRPOPipeline, bucket_hw_base_size: Optional[int]) -> Optional[str]:
    if bucket_hw_base_size is None or not hasattr(pipeline, "target_size_config"):
        return None
    for res, cfg in pipeline.target_size_config.items():
        if cfg.get("bucket_hw_base_size", None) == bucket_hw_base_size:
            return res
    return None


def _resolve_target_size(
    pipeline: HunyuanVideoGRPOPipeline,
    args,
    reference_image: Optional[Image.Image],
) -> Tuple[int, int, Optional[str]]:
    target_resolution = _resolve_target_resolution(pipeline, getattr(args, "video_bucket_hw_base_size", None))
    if reference_image is not None:
        if target_resolution is None:
            target_resolution = getattr(pipeline, "ideal_resolution", None) or "480p"
        height, width = pipeline.get_closest_resolution_given_reference_image(reference_image, target_resolution)
        return height, width, target_resolution

    target_size = {
        256: (192, 336),
        480: (352, 624),
        640: (480, 864),
        720: (544, 960),
        960: (720, 1280),
        1440: (1080, 1920),
    }
    height, width = target_size.get(getattr(args, "video_bucket_hw_base_size", 480), (352, 624))
    if target_resolution is None:
        target_resolution = getattr(pipeline, "ideal_resolution", None)
    return height, width, target_resolution


def _resolve_sp_size(args, world_size: int) -> int:
    sp_size = None
    if nccl_info is not None:
        parallel_dims = getattr(nccl_info, "parallel_dims", None)
        if parallel_dims is not None:
            sp_size = getattr(parallel_dims, "sp", None)
    if sp_size is None:
        sp_size = getattr(args, "sp_size", None)
    if sp_size is None:
        sp_size = 1
    sp_size = max(int(sp_size), 1)
    if world_size > 0 and world_size % sp_size != 0:
        raise ValueError(
            f"Invalid parallel config: world_size ({world_size}) must be divisible by sp_size ({sp_size})."
        )
    return sp_size


def _get_sp_group_fallback(args, world_size: int):
    if not dist.is_initialized() or world_size <= 1:
        return None, 0
    sp_size = _resolve_sp_size(args, world_size)
    if sp_size == 1:
        return None, 0
    cache_key = (world_size, sp_size)
    cached = _SP_GROUP_CACHE.get(cache_key)
    if cached is None:
        groups = []
        group_ranks_list = []
        dist.barrier()
        for start in range(0, world_size, sp_size):
            group_ranks = list(range(start, start + sp_size))
            groups.append(dist.new_group(ranks=group_ranks))
            group_ranks_list.append(group_ranks)
        _SP_GROUP_CACHE[cache_key] = (groups, group_ranks_list)
        cached = _SP_GROUP_CACHE[cache_key]
    groups, group_ranks_list = cached
    rank = dist.get_rank()
    group_idx = rank // sp_size
    group = groups[group_idx]
    group_ranks = group_ranks_list[group_idx]
    sp_rank = rank % sp_size
    return group, sp_rank


def verify_sp_rank_consistency(
    tensors_dict,
    sp_group,
    sp_rank,
    dp_rank,
    logger,
    tolerance=1e-5,
    enable_check=True
):
    """
    Verify that tensors are consistent across all sp_ranks within the same dp_rank.
    
    Args:
        tensors_dict: Dictionary of {name: tensor} to verify. None values are skipped.
        sp_group: Process group for sequence parallel ranks
        sp_rank: Current sp_rank
        dp_rank: Current dp_rank
        logger: Logger instance
        tolerance: Tolerance for floating point comparison
        enable_check: Whether to enable the check (can be disabled for performance)
    
    Returns:
        bool: True if all tensors are consistent, False otherwise
    """
    if not enable_check or sp_group is None:
        return True
    
    if not dist.is_initialized():
        return True
    
    sp_size = dist.get_world_size(sp_group)
    if sp_size <= 1:
        return True
    
    all_consistent = True
    inconsistent_tensors = []
    
    for name, tensor in tensors_dict.items():
        if tensor is None:
            continue
        
        if not isinstance(tensor, torch.Tensor):
            # For non-tensor values, gather and compare as objects
            gathered_values = [None] * sp_size
            dist.all_gather_object(gathered_values, tensor, group=sp_group)
            
            # Check if all values are the same
            if not all(val == gathered_values[0] for val in gathered_values):
                all_consistent = False
                inconsistent_tensors.append(name)
                logger.warning(
                    f"[SP Consistency Check] dp_rank={dp_rank}, sp_rank={sp_rank}: "
                    f"Non-tensor '{name}' is inconsistent across sp_ranks. "
                    f"Values: {gathered_values}"
                )
            continue
        
        # For tensors, gather and compare
        tensor_flat = tensor.flatten().contiguous()
        gathered_tensors = [torch.zeros_like(tensor_flat) for _ in range(sp_size)]
        dist.all_gather(gathered_tensors, tensor_flat, group=sp_group)
        
        # Compare all gathered tensors with the first one
        reference = gathered_tensors[0]
        for idx, gathered in enumerate(gathered_tensors[1:], start=1):
            if not torch.allclose(gathered, reference, atol=tolerance, rtol=tolerance):
                all_consistent = False
                max_diff = (gathered - reference).abs().max().item()
                inconsistent_tensors.append(name)
                logger.error(
                    f"[SP Consistency Check] dp_rank={dp_rank}, sp_rank={sp_rank}: "
                    f"Tensor '{name}' is inconsistent between sp_rank 0 and sp_rank {idx}. "
                    f"Max diff: {max_diff:.2e}, tolerance: {tolerance:.2e}"
                )
                break
    
    if not all_consistent:
        logger.error(
            f"[SP Consistency Check] dp_rank={dp_rank}, sp_rank={sp_rank}: "
            f"Found {len(inconsistent_tensors)} inconsistent tensor(s): {inconsistent_tensors}"
        )
    else:
        logger.debug(
            f"[SP Consistency Check] dp_rank={dp_rank}, sp_rank={sp_rank}: "
            f"All {len(tensors_dict)} tensor(s) are consistent across sp_ranks"
        )
    
    return all_consistent


def gather_and_process_rewards(
    reward_dicts,
    ranks_per_group,
    num_groups_per_rank,
    group_idx,
    rank_in_group,
    dp_rank,
    sp_rank,
    video_batch_size,
    args,
    device,
    global_step,
    indexs,
    world_size,
    logger,
):
    """
    Simplified reward gathering logic for parallel groups mode.
    """
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    metric_names = set()
    if reward_dicts and len(reward_dicts) > 0:
        for rd in reward_dicts:
            for key in rd.keys():
                if key not in [
                    "prompt",
                    "seed",
                    "video_path",
                    "sample_idx",
                    "ranks_per_group",
                    "group_idx_local",
                    "global_group_idx",
                    "sample_idx_in_group",
                    "group_idx",
                    "rank_in_group",
                ]:
                    metric_names.add(key)

    metric_names = list(metric_names)
    if "avg" not in metric_names:
        metric_names.append("avg")

    if (
        torch.distributed.is_initialized()
        and world_size > 1
        and getattr(args, "enable_global_metric_sync", False)
    ):
        gathered_metric_names = [None] * world_size
        torch.distributed.all_gather_object(gathered_metric_names, metric_names)
        all_metric_names = set()
        for mn_list in gathered_metric_names:
            if isinstance(mn_list, list):
                all_metric_names.update(mn_list)
        all_metric_names.add("avg")
        metric_names = sorted(list(all_metric_names))
        logger.info(f"[GatherRewards] Rank {rank}: Synchronized metric names: {metric_names}")
    else:
        metric_names = sorted(list(set(metric_names)))
        logger.info(
            f"[GatherRewards] Rank {rank}: Using local metric names without global sync: "
            f"{metric_names}"
        )

    group_process_group = None
    gather_group = None
    group_leader_rank = None
    broadcast_group = None
    broadcast_src_rank = None

    fallback_sp_group, fallback_sp_rank = _get_sp_group_fallback(args, world_size)
    runtime_sp_group = getattr(nccl_info, "sp_group", None) if nccl_info is not None else None
    if runtime_sp_group is None:
        runtime_sp_group = fallback_sp_group
    if runtime_sp_group is not None and (nccl_info is None or getattr(nccl_info, "rank_within_spgroup", None) is None):
        sp_rank = fallback_sp_rank

    if ranks_per_group > 1:
        sp_size = _resolve_sp_size(args, world_size)
        all_group_process_groups, all_gather_groups, all_group_leader_ranks = get_parallel_groups(
            ranks_per_group, sp_size, world_size
        )
        group_process_group = all_group_process_groups[group_idx]
        gather_group = all_gather_groups[group_idx]
        group_leader_rank = all_group_leader_ranks[group_idx]
        broadcast_group = group_process_group
        broadcast_src_rank = group_leader_rank
        torch.cuda.empty_cache()
    elif runtime_sp_group is not None:
        broadcast_group = runtime_sp_group
        sp_group_ranks = torch.distributed.get_process_group_ranks(runtime_sp_group)
        broadcast_src_rank = sp_group_ranks[0]

    reward_scores = {}
    if sp_rank == 0:
        local_rewards = {}
        for metric in metric_names:
            scores = []
            for rd in reward_dicts:
                val = rd.get(metric, float("nan"))
                try:
                    scores.append(float(val) if val is not None else float("nan"))
                except (TypeError, ValueError):
                    scores.append(float("nan"))
            local_rewards[metric] = torch.tensor(scores, dtype=torch.float32, device=device)

        if ranks_per_group > 1:
            for metric in metric_names:
                gathered = [torch.zeros_like(local_rewards[metric]) for _ in range(ranks_per_group)]
                torch.distributed.all_gather(gathered, local_rewards[metric], group=gather_group)
                reward_scores[metric] = torch.cat(gathered, dim=0)

            gathered_reward_dicts = [None] * ranks_per_group
            torch.distributed.all_gather_object(gathered_reward_dicts, reward_dicts, group=gather_group)
            complete_reward_dicts = []
            for rd_list in gathered_reward_dicts:
                if isinstance(rd_list, list):
                    complete_reward_dicts.extend(rd_list)

            if rank_in_group == 0:
                save_dir = os.path.join(
                    args.output_dir, "rl_samples", f"{global_step:07d}", f"group_{group_idx}"
                )
                os.makedirs(save_dir, exist_ok=True)
                filename = f"index_{indexs[0] if isinstance(indexs, list) else indexs}_group_{group_idx}_rewards.json"
                with open(os.path.join(save_dir, filename), "w", encoding="utf-8") as f:
                    json.dump(convert_to_json_serializable(complete_reward_dicts), f, ensure_ascii=False, indent=4)
                logger.info(
                    f"[GatherRewards] Rank {rank} saved {len(complete_reward_dicts)} entries for group {group_idx}"
                )
        else:
            reward_scores = local_rewards

            for grp_local_idx in range(num_groups_per_rank):
                global_grp_idx = dp_rank * num_groups_per_rank + grp_local_idx
                start_idx = grp_local_idx * args.num_generations
                end_idx = start_idx + args.num_generations
                group_dicts = reward_dicts[start_idx:end_idx]

                save_dir = os.path.join(
                    args.output_dir, "rl_samples", f"{global_step:07d}", f"group_{global_grp_idx}"
                )
                os.makedirs(save_dir, exist_ok=True)
                first_idx = indexs[start_idx] if start_idx < len(indexs) else 0
                filename = f"index_{first_idx}_group_{global_grp_idx}_rewards.json"
                with open(os.path.join(save_dir, filename), "w", encoding="utf-8") as f:
                    json.dump(convert_to_json_serializable(group_dicts), f, ensure_ascii=False, indent=4)
                logger.info(
                    f"[GatherRewards] Rank {rank} saved {len(group_dicts)} entries for group {global_grp_idx}"
                )
    else:
        expected_size = args.num_generations if ranks_per_group > 1 else video_batch_size
        for metric in metric_names:
            reward_scores[metric] = torch.zeros(expected_size, dtype=torch.float32, device=device)

    if broadcast_group is not None:
        # CRITICAL: Sync metric_names within group before broadcast. sp_rank 0 has full list from
        # reward_dicts; sp_rank != 0 has only ["avg"] from empty reward_dicts. Mismatch causes
        # different broadcast loop counts -> deadlock (leader does 5 broadcasts, receivers do 1).
        metric_names_list = [metric_names]
        torch.distributed.broadcast_object_list(
            metric_names_list, src=broadcast_src_rank, group=broadcast_group
        )
        metric_names = metric_names_list[0]

        expected_size = args.num_generations if ranks_per_group > 1 else video_batch_size
        for metric in metric_names:
            if metric not in reward_scores:
                reward_scores[metric] = torch.zeros(expected_size, dtype=torch.float32, device=device)

        for metric in metric_names:
            torch.distributed.broadcast(reward_scores[metric], src=broadcast_src_rank, group=broadcast_group)

    return reward_scores


@torch.no_grad()
def prepare_samples_online(args, model, ref_model, vae, text_encoder, text_encoder_2, byt5_kwarg, extra_model, reward_inferencer,
                            batch, device, global_step, dp_rank, sp_rank, world_size, logger, mask_type, timesteps_train=None):
    """
    Prepare samples for online GRPO training (Rollout Phase).
    
    This function performs the complete rollout process:
    1. Generate samples using the current policy model
    2. Compute rewards for generated samples
    3. Pre-compute reference model statistics (for KL regularization)
    
    The function supports two parallel group modes based on video_batch_size and num_generations:
    
    Mode 1: Single-rank groups (video_batch_size >= num_generations)
        - Each dp_rank processes complete group(s) locally
        - Example: video_batch_size=8, num_generations=4 -> 2 groups per rank
        - No cross-rank gather needed for rewards
        
    Mode 2: Multi-rank groups (video_batch_size < num_generations)
        - Multiple dp_ranks collaborate to complete one group
        - Example: video_batch_size=2, num_generations=8 -> 4 ranks per group
        - Cross-rank gather needed for reward computation
        
    CRITICAL Constraints:
        - If video_batch_size >= num_generations: video_batch_size % num_generations == 0
        - If video_batch_size < num_generations: num_generations % video_batch_size == 0
        Otherwise, samples within a rank may come from different prompts, breaking group structure!
    
    Args:
        args: Training arguments
        model: Current policy model (for sample generation)
        ref_model: Reference model (for KL regularization, can be None)
        vae: VAE model for encoding/decoding
        text_encoder, text_encoder_2: Text encoders
        byt5_kwarg: ByT5 related arguments
        extra_model: Extra model components
        reward_inferencer: Reward model for computing rewards
        batch: Input batch containing prompts, seeds, etc.
        device: Device to run on
        global_step: Current training step
        dp_rank: Data parallel rank
        sp_rank: Sequence parallel rank (within SP group)
        world_size: Total world size
        logger: Logger instance
        mask_type: Mask type for multitask training
        
    Returns:
        videos: Generated videos tensor [Batch, C, F, H, W]
        reward_scores: Dictionary of reward scores for each sample
        all_latents: All latent states [Batch, Steps+1, C, H, W]
        all_log_probs: Log probabilities for each timestep [Batch, Steps]
        sigma_schedule: Noise schedule used for generation
        generation_info: Dictionary with generation metadata
        all_prompt_embeds: List of prompt embeddings for each sample
        all_ref_means: Pre-computed reference model means [Batch, Steps, ...] or None
    """
    # ========================================================================
    # Phase 1: Parse Input Batch and Initialize Parallel Group Configuration
    # ========================================================================
    indexs, prompts, seeds, ref_image_paths, *text_args = batch
    rank = dist.get_rank() if dist.is_initialized() else 0
    # Derive runtime SP rank from global rank layout to avoid stale/misaligned caller-side sp_rank.
    # This matches the current codebase assumption: ranks are laid out contiguously by SP.
    sp_size = max(int(getattr(args, "sp_size", 1)), 1)
    runtime_sp_rank = (rank % sp_size) if dist.is_initialized() else int(sp_rank)
    
    # Normalize prompts/indexs to list format
    if isinstance(prompts, str):
        prompts = [prompts]
        indexs = [indexs] if not isinstance(indexs, list) else indexs
    
    video_batch_size = len(prompts)
    
    # Get reference image path (use first one if available, all samples in batch share the same ref image)
    ref_image_path = None
    if isinstance(ref_image_paths, list) and len(ref_image_paths) > 0:
        ref_image_path = ref_image_paths[0]
    
    # ------------------------------------------------------------------------
    # Configure Parallel Group Mode
    # ------------------------------------------------------------------------
    # Key concept: ranks_per_group determines how many ranks collaborate on one group
    # Configuration validation is done in get_post_train_video_dataloader() to catch errors early
    generation_mode = "parallel_groups"
    samples_per_rank = video_batch_size
    
    # Calculate group division (validation already done in dataloader initialization)
    if video_batch_size >= args.num_generations:
        # Case 1: Each rank has complete group(s) - no cross-rank communication needed
        ranks_per_group = 1
        num_groups_per_rank = video_batch_size // args.num_generations
    else:
        # Case 2: Multiple ranks form a group - cross-rank gather needed for rewards
        ranks_per_group = args.num_generations // video_batch_size
        num_groups_per_rank = 1
    
    # Calculate group indices for this rank
    group_idx = dp_rank // ranks_per_group  # Which group this rank belongs to
    rank_in_group = dp_rank % ranks_per_group  # Position within the group
    
    logger.info(f"[ParallelGroups] Rank {rank}, dp_rank {dp_rank}: "
               f"video_batch_size={video_batch_size}, num_generations={args.num_generations}, "
               f"ranks_per_group={ranks_per_group}, num_groups_per_rank={num_groups_per_rank}, "
               f"group_idx={group_idx}, rank_in_group={rank_in_group}")
    
    # ------------------------------------------------------------------------
    # Phase 2: Process Seeds for Reproducible Generation
    # ------------------------------------------------------------------------
    # Each sample needs a unique seed to ensure diversity within groups
    processed_seeds = []
    for sidx in range(video_batch_size):
        base_seed = seeds[sidx].item() if isinstance(seeds[sidx], torch.Tensor) else seeds[sidx]
        if args.use_same_noise:
            # Use same seed for all samples (for debugging/testing)
            processed_seeds.append(base_seed)
        else:
            # Each sample gets unique seed based on its position within the group
            if ranks_per_group == 1:
                # Single-rank group: use local sample index within group
                sample_idx_in_group = sidx % args.num_generations
            else:
                # Multi-rank group: offset by rank position to ensure uniqueness across ranks
                sample_idx_in_group = rank_in_group * video_batch_size + sidx
            processed_seeds.append(base_seed + sample_idx_in_group)
    seeds = processed_seeds
    
    # ========================================================================
    # Phase 3: Initialize Pipeline and Scheduler
    # ========================================================================
    infer_flow_shift = args.infer_flow_shift_video 
    scheduler = FlowMatchDiscreteScheduler(
                shift=infer_flow_shift,
                reverse=True,
                solver="euler",
            )
    scheduler.set_timesteps(num_inference_steps=args.grpo_sampling_steps, device=device)
    pipeline = HunyuanVideoGRPOPipeline(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            transformer=model,
            scheduler=scheduler,
            args=args,
            byt5_kwarg=byt5_kwarg,
            extra_model=extra_model
        )
    reorg_token = '888' in args.model_type or '16168' in args.model_type
    sigma_schedule = scheduler.sigmas  # Store schedule for later use in training

    # ========================================================================
    # Phase 4: Initialize Storage Containers
    # ========================================================================
    all_latents = []      # Store all latent states: [Batch, Steps+1, C, H, W]
    all_log_probs = []    # Store log probabilities: [Batch, Steps]
    all_prompt_embeds = []  # Store prompt embeddings for each sample (for training)
    all_videos = []       # Store generated videos: [Batch, C, F, H, W]
    reward_dicts = []     # Store reward dictionaries for each sample

    # Ensure main model is on GPU before starting the loop
    model = model.to(device)
    
    # Get mini-batch size for rollout generation (to manage memory)
    mini_batch_size = getattr(args, 'mini_batch_size_per_update', 1)
    if mini_batch_size <= 0:
        mini_batch_size = 1
    
    # ========================================================================
    # Phase 5: Rollout Loop - Generate Samples with Current Policy
    # ========================================================================
    # Process prompts in mini-batches to manage memory usage
    num_prompts = len(prompts)
    ref_image_pil = _load_reference_image(ref_image_path)
    task_type = mask_type if mask_type in ("t2v", "i2v") else None
    multitask_mask_training_type = getattr(args, "multitask_mask_training_type", None)
    if task_type is None and multitask_mask_training_type in ("concat", "token_replace"):
        task_type = "i2v" if ref_image_pil is not None else "t2v"
    target_height, target_width, target_resolution = _resolve_target_size(
        pipeline, args, ref_image_pil
    )
    target_length = args.sample_n_frames
    data_type = args.data_type

    for batch_start_idx in range(0, num_prompts, mini_batch_size):
        # --------------------------------------------------------------------
        # 5.1: Prepare Mini-Batch
        # --------------------------------------------------------------------
        batch_end_idx = min(batch_start_idx + mini_batch_size, num_prompts)
        batch_prompts = prompts[batch_start_idx:batch_end_idx]
        batch_indices = list(range(batch_start_idx, batch_end_idx))
        batch_size_actual = len(batch_prompts)
        
        # Prepare batch seeds, generators, and save paths
        batch_seeds = []
        batch_generators = []
        batch_save_paths = []
        batch_group_info = []
        
        # For each sample in the mini-batch, prepare generation metadata
        for local_idx, sidx in enumerate(batch_indices):
            seed_value = seeds[sidx] if isinstance(seeds[sidx], (int, float)) else seeds[sidx].item()
            batch_seeds.append(seed_value)
            
            if ranks_per_group == 1:
                group_idx_local = sidx // args.num_generations
                sample_idx_in_group = sidx % args.num_generations
                global_group_idx = dp_rank * num_groups_per_rank + group_idx_local
            else:
                sample_idx_in_group = rank_in_group * video_batch_size + sidx
                global_group_idx = group_idx
            
            save_dir = os.path.join(args.output_dir, "rl_samples", f"{global_step:07d}", f"group_{global_group_idx}")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(
                save_dir,
                f"index_{indexs[sidx]}_seed_{seed_value}_dp_{dp_rank}_rank_{rank}_sample_{sample_idx_in_group}_slot_{sidx}.mp4",
            )
            batch_save_paths.append(save_path)
            batch_generators.append(torch.Generator(device=device).manual_seed(seed_value))
            batch_group_info.append({
                "sidx": sidx,
                "group_idx_local": group_idx_local if ranks_per_group == 1 else None,
                "global_group_idx": global_group_idx,
                "sample_idx_in_group": sample_idx_in_group,
            })
        
        # --------------------------------------------------------------------
        # 5.3: Generate Videos with Current Policy Model
        # --------------------------------------------------------------------
        logger.info(f"Rank {rank} generating batch {batch_start_idx//mini_batch_size + 1}/{(num_prompts + mini_batch_size - 1)//mini_batch_size} "
                   f"(samples {batch_start_idx+1}-{batch_end_idx}/{num_prompts}), "
                   f"eta: {args.eta}, batch_size: {batch_size_actual}, "
                   f"{target_length}x{target_height}x{target_width}, flow_shift: {infer_flow_shift}, infer_steps: {args.grpo_sampling_steps}")
        
        # Configure determistic sampling for progressive training
        # For progressive training: use SDE (deterministic=False) for trainable timesteps, ODE (deterministic=True) for others
        # For "all" strategy (no mixgrpo): use SDE (deterministic=False) for all timesteps
        determistic = None
        training_strategy = getattr(args, 'training_strategy', 'all')
        if training_strategy in ["progressive", "random", "decay", "dynamic"] and timesteps_train is not None:
            # Initialize determistic list: True for all timesteps (ODE by default)
            # Length must match num_inference_steps (args.grpo_sampling_steps) for pipeline
            num_inference_steps = args.grpo_sampling_steps
            determistic = [True] * num_inference_steps
            # Set False for trainable timesteps (use SDE for diversity)
            for timestep_i in timesteps_train:
                if timestep_i < num_inference_steps:
                    determistic[timestep_i] = False
            
            logger.info(f"Rank {rank} using progressive training: timesteps_train={timesteps_train}, "
                       f"num_deterministic={sum(determistic)}/{len(determistic)}")
        elif training_strategy == "all":
            # When not using mixgrpo, use SDE (deterministic=False) for all timesteps
            determistic = False
            logger.info(f"Rank {rank} using 'all' strategy (no mixgrpo): determistic=False for all timesteps")
        
        with torch.no_grad():
            # Configure classifier-free guidance
            guidance_scale = args.cfg_scale
            do_classifier_free_guidance = guidance_scale > 1.0
            
            # Batch generation: pass list of prompts to pipeline for parallel processing
            # Handle generator: use list if batch_size > 1, single generator if batch_size == 1
            generator_arg = batch_generators if batch_size_actual > 1 else batch_generators[0]
            
            # Generate videos and collect latents/log_probs for training
            # Returns:
            #   - videos_batch: [batch_size, C, F, H, W]
            #   - batch_latents_batch: [batch_size, Steps+1, C, H, W] - all latent states
            #   - batch_log_probs_batch: [batch_size, Steps] - log probs for each timestep
            # Prepare pipeline kwargs
            pipeline_kwargs = {
                "prompt": batch_prompts,  # Pass list of prompts for batch generation
                "height": target_height,
                "width": target_width,
                "video_length": target_length,
                "eta": args.eta,
                "num_inference_steps": args.grpo_sampling_steps,
                "guidance_scale": args.cfg_scale,
                "negative_prompt": args.neg_prompt,
                "num_images_per_prompt": 1,
                "generator": generator_arg,
                "vae_ver": args.vae,
                "enable_tiling": args.vae_tiling,
                "data_type": data_type,
                "output_type": 'pt',
                "reorg_token": '888' in args.model_type or '16168' in args.model_type,
                "multitask_mask_training_type": args.multitask_mask_training_type,
                "multitask_type": task_type,
                "reference_image": ref_image_pil,
                "vision_num_semantic_tokens": args.vision_num_semantic_tokens,
                "vision_states_dim": args.vision_states_dim,
                "bucket_hw_base_size": args.video_bucket_hw_base_size,
                "bucket_hw_bucket_stride": args.video_bucket_hw_bucket_stride,
                "return_dict": True,
                "show_progress_bar": False,
                "sde_type": args.sde_type,
            }
            
            # Add determistic parameter (always pass it, whether it's a list, False, or True)
            if determistic is not None:
                pipeline_kwargs["determistic"] = determistic
            
            videos_batch, batch_latents_batch, batch_log_probs_batch, _, _ = pipeline(**pipeline_kwargs)
            
            # --------------------------------------------------------------------
            # 5.4: Pre-compute Prompt Embeddings for Training
            # --------------------------------------------------------------------
            # Pre-compute embeddings to avoid recomputation during training loop
            batch_prompt_embeds_list = []
            for prompt in batch_prompts:
                prompt_bundle = pipeline.prepare_prompt_embeddings(
                    prompt, device, 1, negative_prompt=args.neg_prompt,
                    data_type=data_type, text_encoder=text_encoder, text_encoder_2=text_encoder_2,
                    do_classifier_free_guidance=do_classifier_free_guidance, use_glyph_byT5=args.glyph_byT5_v2,
                )
                # Store all prompt-related embeddings for later use in training
                batch_prompt_embeds_list.append({
                    "prompt_embeds": prompt_bundle.prompt_embeds.detach().clone(),
                    "prompt_mask": prompt_bundle.prompt_mask.detach().clone() if prompt_bundle.prompt_mask is not None else None,
                    "prompt_embeds_2": prompt_bundle.prompt_embeds_2.detach().clone() if prompt_bundle.prompt_embeds_2 is not None else None,
                    "extra_kwargs": prompt_bundle.extra_kwargs,  # Contains ByT5 states if used
                })
        
        # --------------------------------------------------------------------
        # 5.5: Store Generated Results
        # --------------------------------------------------------------------
        # Store results for each sample in the mini-batch
        for local_idx in range(batch_size_actual):
            all_latents.append(batch_latents_batch[local_idx:local_idx+1])      # [1, Steps+1, C, H, W]
            all_log_probs.append(batch_log_probs_batch[local_idx:local_idx+1])  # [1, Steps]
            all_prompt_embeds.append(batch_prompt_embeds_list[local_idx])       # Dict with embeddings
            all_videos.append(videos_batch[local_idx:local_idx+1])              # [1, C, F, H, W]

        # --------------------------------------------------------------------
        # 5.6: Compute Rewards (Only Rank 0 of SP Group)
        # --------------------------------------------------------------------
        # Only SP rank 0 computes rewards to avoid redundant computation
        if runtime_sp_rank == 0:
            batch_video_paths = []
            batch_images_for_reward = []
            
            for local_idx, (sidx, save_path) in enumerate(zip(batch_indices, batch_save_paths)):
                videos_single = videos_batch[local_idx:local_idx+1]  # (1, C, F, H, W)
                
                # Save as PNG when generating single frame (image), otherwise save as video
                if target_length == 1:
                    save_path_png = save_path.replace('.mp4', '.png')
                    image_tensor = videos_single.squeeze(2)
                    torchvision.utils.save_image(image_tensor, save_path_png)
                    save_path = save_path_png
                    logger.info(f"Rank {rank}: Saved image {local_idx+1}/{batch_size_actual} to {save_path_png}")
                    batch_images_for_reward.append(image_tensor.squeeze(0))  # (C, H, W) for reward
                    batch_video_paths.append(None)
                else:
                    save_videos_grid(videos_single, save_path, n_rows=1, fps=args.video_fps)
                    logger.info(f"Rank {rank}: Saved video {local_idx+1}/{batch_size_actual} to {save_path}")
                    batch_video_paths.append(os.path.abspath(save_path))
                    batch_images_for_reward.append(None)
            
            with torch.no_grad():
                if target_length == 1:
                    # For images, stack tensors and compute rewards in batch
                    images_tensor = torch.stack(batch_images_for_reward, dim=0)  # (batch_size, C, H, W)
                    scores_dict, meta_dict = reward_inferencer(images_tensor, batch_prompts, [{}] * batch_size_actual)
                else:
                    # For videos, pass list of file paths
                    scores_dict, meta_dict = reward_inferencer(batch_video_paths, batch_prompts)
            
            for local_idx, (sidx, prompt, seed_value, save_path, group_info) in enumerate(zip(batch_indices, batch_prompts, batch_seeds, batch_save_paths, batch_group_info)):
                reward_entry = {}
                for k, v in scores_dict.items():
                    value = v[local_idx] if isinstance(v, list) and local_idx < len(v) else v
                    reward_entry[k] = value
                
                reward_dict_entry = {
                    "prompt": prompt,
                    "seed": seed_value,
                    "video_path": save_path,
                    "sample_idx": sidx,
                    "ranks_per_group": ranks_per_group,
                }
                
                if ranks_per_group == 1:
                    reward_dict_entry.update({"group_idx_local": group_info["group_idx_local"], "global_group_idx": group_info["global_group_idx"], "sample_idx_in_group": group_info["sample_idx_in_group"]})
                else:
                    reward_dict_entry.update({"group_idx": group_idx, "rank_in_group": rank_in_group, "sample_idx_in_group": group_info["sample_idx_in_group"]})
                
                reward_dict_entry.update(reward_entry)
                reward_dicts.append(reward_dict_entry)
    
    # ========================================================================
    # Phase 6: Gather Rewards Across Ranks and Concatenate All Results
    # ========================================================================
    # Barrier: ensure all ranks have finished reward inference before gather.
    # Prevents NCCL timeout when sp_rank 0 (reward compute) lags behind sp_rank != 0.
    if torch.distributed.is_initialized() and world_size > 1:
        torch.distributed.barrier(device_ids=[int(os.environ.get("LOCAL_RANK", 0))])
    # Gather rewards from all ranks and process them for training
    reward_scores = gather_and_process_rewards(
        reward_dicts=reward_dicts,
        ranks_per_group=ranks_per_group,
        num_groups_per_rank=num_groups_per_rank,
        group_idx=group_idx,
        rank_in_group=rank_in_group,
        dp_rank=dp_rank,
        sp_rank=runtime_sp_rank,
        video_batch_size=video_batch_size,
        args=args,
        device=device,
        global_step=global_step,
        indexs=indexs,
        world_size=world_size,
        logger=logger,
    )
    # Barrier: ensure all ranks have finished gather before proceeding to ref model.
    if torch.distributed.is_initialized() and world_size > 1:
        torch.distributed.barrier(device_ids=[int(os.environ.get("LOCAL_RANK", 0))])

    # Expand reward scores to match timestep dimension for training
    reward_scores["ori_avg"] = reward_scores["avg"]  # Store original for logging
    reward_scores["avg"] = reward_scores["avg"].unsqueeze(1).repeat(1, args.grpo_sampling_steps)  # [Batch, Steps]

    # Concatenate all collected results into tensors
    all_latents = torch.cat(all_latents, dim=0)      # [Total_Batch, Steps+1, C, T, H, W]
    all_log_probs = torch.cat(all_log_probs, dim=0)  # [Total_Batch, Steps]
    videos = torch.cat(all_videos, dim=0)            # [Total_Batch, C, T, H, W]

    # Clean up pipeline to free memory before reference model computation
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    
    # Get mini-batch size for reference model computation (should match generation batch size)
    mini_batch_size = getattr(args, 'mini_batch_size_per_update', 1)
    if mini_batch_size <= 0:
        mini_batch_size = 1

    # ========================================================================
    # Phase 7: Pre-compute Reference Model Statistics (for KL Regularization)
    # ========================================================================
    # Pre-compute reference model statistics to avoid OOM during training
    # This computes mean predictions for all timesteps, stored on CPU to save GPU memory
    # Can be disabled by setting kl_compute_mode="training_phase" to compute on-the-fly during training
    all_ref_means = None
    ref_model_time = 0.0  # Track reference model computation time (CUDA-synced)
    
    # Get KL computation mode: "rollout_phase" (default, pre-compute) or "training_phase" (on-the-fly)
    kl_compute_mode = getattr(args, "kl_compute_mode", "rollout_phase")
    
    # Only pre-compute if:
    # 1. KL regularization is enabled
    # 2. Reference model is available
    # 3. kl_compute_mode is "rollout_phase" (default)
    if getattr(args, "kl_weight", 0.0) > 0 and ref_model is not None and kl_compute_mode == "rollout_phase":
        logger.info(f"Rank {rank}: Pre-computing Reference Model statistics (Batch-wise)...")
        
        # Record start time for reference model computation (CUDA-synced, with barrier)
        ref_model_start_time = sync_cuda_time()
        
        # --------------------------------------------------------------------
        # 7.1: Move Reference Model to GPU (if offloading is enabled)
        # --------------------------------------------------------------------
        if args.reference_mode_offload:
            ref_model = ref_model.to(device)
        
        # Initialize containers for collecting reference statistics
        final_ref_means = []
        
        # Prepare timesteps tensor for reference model computation
        # Note: sigma_schedule[:-1] because we have Steps+1 latents but only Steps timesteps
        timesteps_tensor = (sigma_schedule[:-1] * 1000).to(device=device, dtype=torch.float32)
        num_steps = args.grpo_sampling_steps
        total_batch_size = all_latents.shape[0]

        # --------------------------------------------------------------------
        # 7.2: Process in Mini-Batches to Manage Memory
        # --------------------------------------------------------------------
        # Process reference model computation in mini-batches to avoid OOM
        for start_idx in tqdm(range(0, total_batch_size, mini_batch_size), 
                              desc="Processing reference model mini-batches",
                              disable=rank >= 1):
            end_idx = min(start_idx + mini_batch_size, total_batch_size)
            current_batch_size = end_idx - start_idx
            
            # --------------------------------------------------------------------
            # 7.2.1: Extract Mini-Batch Latents
            # --------------------------------------------------------------------
            # all_latents: [Total_Batch, Steps+1, C, H, W]
            mini_latents = all_latents[start_idx:end_idx]  # [MiniBatch, Steps+1, C, T, H, W]
            
            # --------------------------------------------------------------------
            # 7.2.2: Extract and Stack Mini-Batch Prompt Embeddings
            # --------------------------------------------------------------------
            # all_prompt_embeds is a list, need to slice then stack
            mini_prompts_list = all_prompt_embeds[start_idx:end_idx]
            
            # Stack prompt embeddings for batch processing
            prompt_embeds_mini = torch.cat([pe["prompt_embeds"] for pe in mini_prompts_list], dim=0)
            
            # Handle optional prompt_mask
            if mini_prompts_list[0]["prompt_mask"] is not None:
                prompt_mask_mini = torch.cat([pe["prompt_mask"] for pe in mini_prompts_list], dim=0)
            else:
                prompt_mask_mini = None
                
            # Handle optional prompt_embeds_2 (for dual text encoder)
            if mini_prompts_list[0]["prompt_embeds_2"] is not None:
                prompt_embeds_2_mini = torch.cat([pe["prompt_embeds_2"] for pe in mini_prompts_list], dim=0)
            else:
                prompt_embeds_2_mini = None
            
            # --------------------------------------------------------------------
            # 7.2.3: Process Extra Kwargs (e.g., ByT5 text states)
            # --------------------------------------------------------------------
            extra_kwargs_mini = {}
            first_extra = mini_prompts_list[0]["extra_kwargs"]
            if first_extra is not None and "byt5_text_states" in first_extra and first_extra["byt5_text_states"] is not None:
                 # Stack ByT5 text states if available
                 extra_kwargs_mini["byt5_text_states"] = torch.cat(
                     [pe["extra_kwargs"]["byt5_text_states"] for pe in mini_prompts_list], dim=0
                 )
                 if "byt5_text_mask" in first_extra and first_extra["byt5_text_mask"] is not None:
                     extra_kwargs_mini["byt5_text_mask"] = torch.cat(
                         [pe["extra_kwargs"]["byt5_text_mask"] for pe in mini_prompts_list], dim=0
                     )
                 # Copy other keys from first_extra if any
                 for k, v in first_extra.items():
                     if k not in ["byt5_text_states", "byt5_text_mask"]:
                         extra_kwargs_mini[k] = v
            elif first_extra is not None:
                 extra_kwargs_mini = first_extra.copy() if isinstance(first_extra, dict) else first_extra
            
            # --------------------------------------------------------------------
            # 7.2.4: Compute Reference Model Statistics for All Timesteps
            # --------------------------------------------------------------------
            mini_batch_means = []
            with torch.no_grad():
                # Process each timestep in the sampling schedule
                for t_idx in range(num_steps):
                    # Extract latents for timestep t and t+1
                    latents_t = mini_latents[:, t_idx].to(device)        # [MiniBatch, C, T, H, W] - current state
                    prev_latents_t = mini_latents[:, t_idx+1].to(device) # [MiniBatch, C, T, H, W] - next state (for SDE)
                    
                    # Expand timestep to match batch size
                    timestep_t = timesteps_tensor[t_idx].expand(current_batch_size)
                    
                    # Get determistic value for this timestep (if using progressive training)
                    # determistic can be a list (per-timestep) or a single bool
                    determistic_i = False  # Default: use SDE
                    if determistic is not None:
                        if isinstance(determistic, list):
                            if t_idx < len(determistic):
                                determistic_i = determistic[t_idx]
                        else:
                            determistic_i = determistic
                    
                    # Compute reference model prediction for this timestep
                    # Returns: (log_prob, ref_mean, std_dev)
                    # We only need ref_mean for KL regularization
                    _, ref_mean, _ = grpo_one_step(
                        args, batch, vae, text_encoder, text_encoder_2, byt5_kwarg, extra_model, 
                        ref_model,  # Use reference model instead of policy model
                        latents_t,
                        prev_latents_t,
                        timestep_t,
                        sigma_schedule,
                        logger,
                        prompt_embeds=prompt_embeds_mini,
                        prompt_mask=prompt_mask_mini,
                        prompt_embeds_2=prompt_embeds_2_mini,
                        extra_kwargs=extra_kwargs_mini,
                        determistic=determistic_i,
                    )
                    
                    # Store on CPU to save GPU memory
                    mini_batch_means.append(ref_mean.cpu())
            
            # Stack timestep dimension: [MiniBatch, Steps, ...]
            final_ref_means.append(torch.stack(mini_batch_means, dim=1))

        # --------------------------------------------------------------------
        # 7.3: Move Reference Model back to CPU (if offloading enabled)
        # --------------------------------------------------------------------
        if args.reference_mode_offload:
            ref_model = ref_model.to('cpu')
            torch.cuda.empty_cache()

        # --------------------------------------------------------------------
        # 7.4: Concatenate All Mini-Batch Results
        # --------------------------------------------------------------------
        # Final shape: [Total_Batch, Steps, ...]
        all_ref_means = torch.cat(final_ref_means, dim=0)
        
        # Record end time and log duration (CUDA-synced, with barrier)
        ref_model_end_time = sync_cuda_time()
        ref_model_duration = ref_model_end_time - ref_model_start_time
        ref_model_time = ref_model_duration  # Store for return
        logger.info(
            f"Rank {rank}: Reference stats computed (Batched). "
            f"Shape: {all_ref_means.shape}, Time: {ref_model_duration:.2f}s ({ref_model_duration/60:.2f}min)"
        )

    # ========================================================================
    # Phase 8: Finalize and Return Results
    # ========================================================================
    # Restore main model to trainable mode (was set to eval during generation)
    model.train()

    # Prepare generation metadata for training loop
    generation_info = {
        "generation_mode": generation_mode,
        "video_batch_size": video_batch_size,
        "ranks_per_group": ranks_per_group,
        "num_groups_per_rank": num_groups_per_rank,
        "group_idx": group_idx,
        "rank_in_group": rank_in_group,
        "samples_per_group": args.num_generations,
    }

    # Final memory cleanup
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
    return videos, reward_scores, all_latents, all_log_probs, sigma_schedule, generation_info, all_prompt_embeds, all_ref_means, ref_model_time, determistic


def _build_grpo_scheduler(args, sigma_schedule, device):
    infer_flow_shift = args.infer_flow_shift_video
    scheduler = FlowMatchDiscreteScheduler(shift=infer_flow_shift, reverse=True, solver="euler")
    sigma_schedule_local = sigma_schedule.to(device=device, dtype=torch.float32).clone()
    scheduler.num_inference_steps = sigma_schedule_local.shape[0] - 1
    scheduler.sigmas = sigma_schedule_local
    scheduler.timesteps = (sigma_schedule_local[:-1] * scheduler.config.num_train_timesteps).to(
        device=device, dtype=torch.float32
    )
    scheduler._step_index = None
    scheduler._begin_index = None
    return scheduler


def grpo_one_step(
            args,
            batch,
            vae,
            text_encoder,
            text_encoder_2,
            byt5_kwarg,
            extra_model,
            transformer,
            latents,
            pre_latents,
            timesteps,
            sigma_schedule,
            logger,
            prompt_embeds=None,
            prompt_mask=None,
            prompt_embeds_2=None,
            extra_kwargs=None,
            determistic=False,
):
    transformer.train()
    indexs, prompts, seeds, ref_image_paths, *text_args = batch
    rank = dist.get_rank() if dist.is_initialized() else 0
    # logger.info(f'Rank: {torch.distributed.get_rank()}, grpo one step: prompt: {prompts[0]}, latents: {latents.shape}, pre_latents: {pre_latents.shape}, timesteps: {timesteps}')
    # sample videos
    device = transformer.device
    scheduler = _build_grpo_scheduler(args, sigma_schedule, device)
    pipeline = HunyuanVideoGRPOPipeline(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            transformer=transformer,
            scheduler=scheduler,
            args=args,
            byt5_kwarg=byt5_kwarg,
            extra_model=extra_model
        )
    device = transformer.device

    guidance_scale = getattr(args, "cfg_scale", 1.0)
    
    # Use pre-computed prompt embeddings if provided (aligned with train_sd3.py)
    if prompt_embeds is None:
        # Fallback: compute if not provided (backward compatibility)
        negative_prompt = getattr(args, "neg_prompt", None)
        do_classifier_free_guidance = guidance_scale > 1.0
        prompt = prompts[0]
        prompt_bundle = pipeline.prepare_prompt_embeddings(
            prompt,
            device,
            latents.shape[0],
            negative_prompt=negative_prompt,
            data_type="video",
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            do_classifier_free_guidance=do_classifier_free_guidance,
            use_glyph_byT5=args.glyph_byT5_v2,
        )
        prompt_embeds = prompt_bundle.prompt_embeds
        prompt_mask = prompt_bundle.prompt_mask
        prompt_embeds_2 = prompt_bundle.prompt_embeds_2
        extra_kwargs = prompt_bundle.extra_kwargs
    else:
        # Move pre-computed embeddings to correct device
        prompt_embeds = prompt_embeds.to(device)
        if prompt_mask is not None:
            prompt_mask = prompt_mask.to(device)
        if prompt_embeds_2 is not None:
            prompt_embeds_2 = prompt_embeds_2.to(device)
        if extra_kwargs is None:
            extra_kwargs = {}

    # Get reference image path
    ref_image_path = None
    if isinstance(ref_image_paths, list) and len(ref_image_paths) > 0:
        ref_image_path = ref_image_paths[0]

    vision_states = None
    base_latents = latents
    mask_type = None
    cond_latents = None
    
    if args.multitask_mask_training_type == "concat" or args.multitask_mask_training_type == "token_replace":
        # Use common helper function to prepare reference image and mask
        ref_image_pil = _load_reference_image(ref_image_path)
        mask_type = "i2v" if ref_image_pil is not None else "t2v"
        target_height, target_width, target_resolution = _resolve_target_size(
            pipeline, args, ref_image_pil
        )
        
        multitask_mask = pipeline.get_task_mask(mask_type, latents.shape[2])
        image_cond = pipeline.get_image_condition_latents(
            mask_type, ref_image_pil, target_height, target_width
        )
        cond_latents = pipeline._prepare_cond_latents(
            mask_type, image_cond, latents, multitask_mask
        )
        vision_states = pipeline._prepare_vision_states(
            ref_image_pil, target_resolution, latents, device
        )
        base_latents = torch.concat([latents, cond_latents], dim=1)

    scheduler_timesteps = timesteps.to(device=device, dtype=torch.float32)
    model_pred, _ = pipeline.denoise_step(
        base_latents,
        scheduler_timesteps,
        prompt_embeds,
        prompt_embeds_2,
        prompt_mask,
        vision_states=vision_states,
        mask_type=mask_type,
        extra_kwargs=extra_kwargs,
        guidance_scale=guidance_scale,
        guidance_rescale=getattr(args, "guidance_rescale", 0.0),
    )

    # If scheduler_timeseps is a batch tensor, process each sample separately
    if scheduler_timesteps.dim() > 0 and scheduler_timesteps.shape[0] > 1:
        # Batch processing: loop over each sample
        batch_size = scheduler_timesteps.shape[0]
        latents_list = []
        pred_latents_original_list = []
        log_prob_list = []
        prev_mean_list = []
        std_dev_t_list = []
        
        for i in range(batch_size):
            # Extract single sample
            single_timestep = scheduler_timesteps[i]
            single_model_pred = model_pred[i:i+1]
            single_latents = latents[i:i+1]
            single_pre_latents = pre_latents[i:i+1] if pre_latents is not None else None
            
            # Reset scheduler step_index to ensure correct initialization for each sample
            # (since different samples may have different timesteps)
            scheduler._step_index = None
            
            # Call scheduler for single sample
            single_latents_out, single_pred_original, single_log_prob, single_prev_mean, single_std_dev_t = scheduler.sde_step_with_logprob(
                single_model_pred,
                single_latents,
                single_timestep,
                args.eta,
                prev_sample=single_pre_latents.to(torch.float32) if single_pre_latents is not None else None,
                grpo=True,
                sde_solver=True,
                sde_type=args.sde_type,
                determistic=determistic,
            )
            
            latents_list.append(single_latents_out)
            pred_latents_original_list.append(single_pred_original)
            log_prob_list.append(single_log_prob)
            prev_mean_list.append(single_prev_mean)
            # std_dev_t might be scalar (0-dim tensor), so we need to handle it carefully
            # Convert to 1D tensor for consistent concatenation
            if single_std_dev_t.dim() == 0:
                std_dev_t_list.append(single_std_dev_t.unsqueeze(0))
            elif single_std_dev_t.dim() == 1 and single_std_dev_t.shape[0] == 1:
                std_dev_t_list.append(single_std_dev_t)
            else:
                # If std_dev_t has spatial dimensions, take mean or first element
                # (std_dev_t should be per-sample, not per-spatial-location)
                std_dev_t_list.append(single_std_dev_t.flatten()[0:1] if single_std_dev_t.numel() > 0 else single_std_dev_t)
        
        # Concatenate results
        latents = torch.cat(latents_list, dim=0)
        pred_latents_original = torch.cat(pred_latents_original_list, dim=0)
        log_prob = torch.cat(log_prob_list, dim=0)
        prev_mean = torch.cat(prev_mean_list, dim=0)
        std_dev_t = torch.cat(std_dev_t_list, dim=0)
    else:
        # Single sample or scalar timestep: process directly
        single_timestep = scheduler_timesteps if scheduler_timesteps.dim() == 0 else scheduler_timesteps[0]
        latents, pred_latents_original, log_prob, prev_mean, std_dev_t = scheduler.sde_step_with_logprob(
            model_pred,
            latents,
            single_timestep,
            args.eta,
            prev_sample=pre_latents.to(torch.float32) if pre_latents is not None else None,
            grpo=True,
            sde_solver=True,
            sde_type=args.sde_type,
            determistic=determistic,
        )

    del pipeline
    return log_prob, prev_mean, std_dev_t
    
def train_one_step(args, model, ref_model, vae, text_encoder, text_encoder_2, byt5_kwarg, extra_model, reward_inferencer,
                            batch, device, dp_rank, world_size, logger, scalar_states, optimizer, lr_scheduler, mask_type, timesteps_train=None):
    """
    Execute one training step of GRPO (Group Relative Policy Optimization).
    
    This function performs:
    1. Online sample generation (rollout)
    2. Advantage computation with reward normalization
    3. Sample shuffling and timestep permutation
    4. Nested training loops (samples x timesteps)
    5. Policy loss computation with optional KL regularization
    """
    sp_rank = nccl_info.rank_within_spgroup
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    # ==================== Phase 1: Online Sample Generation ====================
    rollout_start_time = sync_cuda_time()
    videos, reward_scores, all_latents, all_log_probs, sigma_schedule, generation_info, all_prompt_embeds, all_ref_means, ref_model_time, determistic = prepare_samples_online(
        args, model, ref_model, vae, text_encoder, text_encoder_2, byt5_kwarg, extra_model, reward_inferencer,
        batch, device, scalar_states.update_steps, dp_rank, sp_rank, world_size, logger, mask_type, timesteps_train=timesteps_train
    )
    rollout_end_time = sync_cuda_time()
    rollout_time = rollout_end_time - rollout_start_time
    
    # ==================== Phase 2: Prepare Training Data ====================
    batch_size = all_latents.shape[0]
    
    # Prepare timesteps for all samples
    timestep_value = [sigma * 1000 for sigma in sigma_schedule][:args.grpo_sampling_steps]
    timestep_values = [timestep_value[:] for _ in range(batch_size)]
    timesteps = torch.tensor(timestep_values, device=device, dtype=torch.float32)
    
    # Build samples dict with aligned latents and log_probs
    # Note: log_probs has length num_steps, latents has num_steps+1
    # We take latents[:, :-1] (pre-step) and latents[:, 1:] (post-step) to match log_probs/timesteps
    samples = {
        "timesteps": timesteps.detach().clone(),    # [batch, num_steps]
        "latents": all_latents[:, :-1],             # latent before timestep t
        "next_latents": all_latents[:, 1:],         # latent after timestep t
        "log_probs": all_log_probs,                 # [batch, num_steps]
    }
    
    # Add pre-computed reference model statistics if available
    if all_ref_means is not None:
        # Move to device for training (they were stored on CPU to save memory)
        samples["ref_means"] = all_ref_means.to(device)  # [batch, num_steps, ...]

    # Stack prompt embeddings from all samples (avoid recomputation during training)
    samples["prompt_embeds"] = torch.cat([pe["prompt_embeds"].unsqueeze(0) for pe in all_prompt_embeds], dim=0)
    if all_prompt_embeds[0]["prompt_mask"] is not None:
        samples["prompt_mask"] = torch.cat([pe["prompt_mask"].unsqueeze(0) for pe in all_prompt_embeds], dim=0)
    if all_prompt_embeds[0]["prompt_embeds_2"] is not None:
        samples["prompt_embeds_2"] = torch.cat([pe["prompt_embeds_2"].unsqueeze(0) for pe in all_prompt_embeds], dim=0)
    samples["extra_kwargs"] = [pe["extra_kwargs"] for pe in all_prompt_embeds]
    
    # ==================== Phase 3: Reward Processing and Advantage Computation ====================
    # Add reward scores to samples and gather statistics
    gathered_reward_stats = {}
    for metric_name, reward_tensor in reward_scores.items():
        reward_key = f"{metric_name}_rewards"
        samples[reward_key] = reward_tensor.to(torch.float32)
        gathered_reward_stats[metric_name] = gather_tensor(samples[reward_key])
        # Clean up NaN/inf values
        samples[reward_key] = torch.nan_to_num(samples[reward_key], nan=0.0, posinf=0.0, neginf=0.0)
    
    # Extract grouping info for advantage computation
    ranks_per_group = generation_info.get("ranks_per_group", 1)
    num_groups = generation_info.get("num_groups_per_rank", 1)
    samples_per_grp = generation_info.get("samples_per_group", args.num_generations)
    
    # Normalize rewards to compute advantages (handles both single-rank and multi-rank groups)
    rewards = samples["avg_rewards"]
    advantages = _normalize_group(rewards, ranks_per_group, num_groups, samples_per_grp)
    advantages = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)
    samples["avg_advantages"] = advantages
    
    # For multi-rank groups, keep full-group rewards for normalization above,
    # then slice this rank's local chunk so shapes match local latents.
    if ranks_per_group > 1:
        chunk_start = generation_info.get("rank_in_group", 0) * batch_size
        chunk_end = chunk_start + batch_size
        reward_keys = [f"{metric_name}_rewards" for metric_name in reward_scores.keys()]
        for reward_key in reward_keys:
            samples[reward_key] = samples[reward_key][chunk_start:chunk_end]
        samples["avg_advantages"] = samples["avg_advantages"][chunk_start:chunk_end]

    # ==================== Phase 4: Sample Shuffling ====================
    # Shuffle samples along batch dimension for better training stability
    # Note: This is generally beneficial even when using all samples, as it breaks potential
    # ordering biases and improves training stability. Cost is negligible.
    perm = _sync_random_tensor(
        generator_fn=lambda: torch.randperm(batch_size, device=device),
        shape=(batch_size,),
        dtype=torch.long,
        device=device,
        nccl_info=nccl_info
    )
    
    # Apply permutation to all samples
    reordered_samples = {}
    for k, v in samples.items():
        if isinstance(v, torch.Tensor) and v.dim() > 0:
            # Ensure perm is on the same device as v
            if v.device != perm.device:
                perm_v = perm.to(v.device)
            else:
                perm_v = perm
            reordered_samples[k] = v[perm_v]
        elif isinstance(v, list):
            perm_list = perm.cpu().tolist()
            reordered_samples[k] = [v[idx] for idx in perm_list]
        else:
            reordered_samples[k] = v
    samples = reordered_samples
    
    # ==================== Phase 5: Timestep Permutation ====================
    # Create random permutations for timesteps (synced across SP ranks)
    # Note: When using all timesteps, permutation has limited benefit since all timesteps
    # are trained anyway and gradients are accumulated. However, it adds randomness which
    # may help training. For progressive training (mix_grpo), permutation is less meaningful
    # since only a subset of timesteps are trained.
    # 
    # Skip timestep permutation when using all timesteps
    # to reduce overhead, unless explicitly enabled via config
    enable_timestep_permutation = getattr(args, "enable_timestep_permutation", True)
    num_timesteps = len(samples["timesteps"][0])

    if enable_timestep_permutation:
        timestep_perms = _sync_random_tensor(
            generator_fn=lambda: torch.stack([
                torch.randperm(num_timesteps, device=device)
                for _ in range(batch_size)
            ]),
            shape=(batch_size, num_timesteps),
            dtype=torch.long,
            device=device,
            nccl_info=nccl_info
        )
        
        # Apply timestep permutations
        batch_indices = torch.arange(batch_size, device=device)[:, None]
        keys_to_permute = ["timesteps", "latents", "next_latents", "log_probs"]
        # Also permute pre-computed reference statistics if available
        if "ref_means" in samples:
            keys_to_permute.append("ref_means")
        for key in keys_to_permute:
            samples[key] = samples[key][batch_indices, timestep_perms]
    else:
        timestep_perms = torch.arange(num_timesteps, device=device)
    samples["timestep_perms"] = timestep_perms

    # ==================== Phase 6: Build Per-Sample / Per-MiniBatch Batches ====================
    # Support mini-batch processing along the sample dimension to improve throughput.
    # When mini_batch_size_per_update == 1, this reduces to the original per-sample behavior.
    mini_batch_size = getattr(args, "mini_batch_size_per_update", 1)
    if mini_batch_size <= 0:
        mini_batch_size = 1

    samples_batched_list = []
    for start_idx in range(0, batch_size, mini_batch_size):
        end_idx = min(start_idx + mini_batch_size, batch_size)
        sample_dict = {}
        for k, v in samples.items():
            if isinstance(v, torch.Tensor) and v.dim() > 0:
                # Slice mini-batch along the first (batch) dimension
                sample_dict[k] = v[start_idx:end_idx]
                # print(f'Rank: {torch.distributed.get_rank()}, sample_dict[{k}]: {sample_dict[k].shape}')
            elif isinstance(v, list):
                # Special handling for extra_kwargs: list of dicts -> single batched dict
                if k == "extra_kwargs":
                    sample_dict[k] = _batch_extra_kwargs(v[start_idx:end_idx])
                else:
                    # Slice list entries for this mini-batch
                    sample_dict[k] = v[start_idx:end_idx]
            else:
                # Scalars / non-batched values are shared across the mini-batch
                sample_dict[k] = v
        samples_batched_list.append(sample_dict)

    # ==================== Phase 6.5: Pre-compute per-step grad balancing factors (optional) ====================
    per_step_balancing = None
    if getattr(args, "use_grad_balancing", True) and sigma_schedule is not None:
        eta = getattr(args, "eta", None)
        eta = 1.0 if eta is None else eta
        scheduler_eq = _build_grpo_scheduler(args, sigma_schedule.to(device), device)
        per_step_balancing = scheduler_eq.compute_grad_balancing_factors(
            eta=eta,
            device=device,
            sde_type=getattr(args, "sde_type", "dance_grpo"),
            sde_solver=True,
            sigmas=scheduler_eq.sigmas,
        )
        per_step_balancing = per_step_balancing.to(device=device)
    
    # ==================== Phase 7: GRPO Training Loop ====================
    kl_beta = getattr(args, "kl_weight", 0.0)
    clip_range = getattr(args, "clip_range", 1e-4)
    adv_clip_max = getattr(args, "adv_clip_max", 5.0)
    reference_mode_offload = getattr(args, "reference_mode_offload", True)
    
    # Determine which timesteps to train on
    # For progressive/random/decay/dynamic strategies, only train on timesteps in the current group
    if timesteps_train is not None:
        # Use the provided timesteps for progressive training
        train_timesteps_list = timesteps_train
        logger.info(f"[Progressive Training] Training on timesteps: {train_timesteps_list}")
    else:
        # Default behavior: train on all timesteps (up to timestep_fraction)
        train_timesteps = max(int(len(samples["timesteps"][0]) * args.timestep_fraction), 1)
        train_timesteps_list = list(range(train_timesteps))
        logger.info(f"[All Timesteps] Training on timesteps: {train_timesteps_list}")


    # Initialize metrics collection
    info = defaultdict(list)
    inner_step = 0
    grad_norm = 0.0
    param_delta_mean = 0.0
    param_delta_max = 0.0
    debug_grad_flow = bool(getattr(args, "debug_grad_flow", True))
    debug_diag_interval = max(int(getattr(args, "debug_train_diagnostics_interval", 1) or 1), 1)
    tracked_name = None
    tracked_param = None
    for n, p in model.named_parameters():
        if p.requires_grad:
            tracked_name = n
            tracked_param = p
            break
    if debug_grad_flow:
        if tracked_param is None:
            logger.warning(f"Rank {rank}, sp_rank {sp_rank}: debug_grad_flow enabled but no trainable parameter found.")
        else:
            logger.info(f"Rank {rank}, sp_rank {sp_rank}: debug tracking parameter -> {tracked_name}, dtype={tracked_param.dtype}")
    
    # ========== Verify SP rank consistency before training loop ==========
    # Ensure all sp_ranks within the same dp_rank have identical input data
    enable_sp_check = getattr(args, "enable_sp_consistency_check", False)
    if enable_sp_check and nccl_info.sp_group is not None:
        # Build tensors dict from samples for verification
        tensors_to_verify = {}
        for key in ["timesteps", "latents", "next_latents", "log_probs", "prompt_embeds"]:
            if key in samples and isinstance(samples[key], torch.Tensor):
                tensors_to_verify[key] = samples[key]
        if "prompt_mask" in samples and samples["prompt_mask"] is not None:
            tensors_to_verify["prompt_mask"] = samples["prompt_mask"]
        if "prompt_embeds_2" in samples and samples["prompt_embeds_2"] is not None:
            tensors_to_verify["prompt_embeds_2"] = samples["prompt_embeds_2"]
        
        verify_sp_rank_consistency(
            tensors_dict=tensors_to_verify,
            sp_group=nccl_info.sp_group,
            sp_rank=sp_rank,
            dp_rank=dp_rank,
            logger=logger,
            tolerance=getattr(args, "sp_consistency_tolerance", 1e-5),
            enable_check=enable_sp_check,
        )
    
    grpo_loop_start_time = sync_cuda_time()
    
    for i, sample in tqdm(
        list(enumerate(samples_batched_list)),
        desc=f"Global Step {scalar_states.update_steps}: training",
        position=0,
        disable=rank >= 1,
    ):
        # Extract prompt embeddings once per (mini-)batch.
        prompt_embeds = sample["prompt_embeds"]
        prompt_mask = sample.get("prompt_mask")
        prompt_embeds_2 = sample.get("prompt_embeds_2")
        extra_kwargs = sample.get("extra_kwargs")
        if mini_batch_size > 1:
            prompt_embeds = prompt_embeds.reshape(-1, prompt_embeds.shape[-2], prompt_embeds.shape[-1])
            prompt_mask = prompt_mask.reshape(-1, prompt_mask.shape[-1]) if prompt_mask is not None else None
            prompt_embeds_2 = prompt_embeds_2.reshape(-1, prompt_embeds_2.shape[-2], prompt_embeds_2.shape[-1]) if prompt_embeds_2 is not None else None
            extra_kwargs["byt5_text_states"] = extra_kwargs["byt5_text_states"].reshape(-1, extra_kwargs["byt5_text_states"].shape[-2], extra_kwargs["byt5_text_states"].shape[-1]) if extra_kwargs["byt5_text_states"] is not None else None
            extra_kwargs["byt5_text_mask"] = extra_kwargs["byt5_text_mask"].reshape(-1, extra_kwargs["byt5_text_mask"].shape[-1]) if extra_kwargs["byt5_text_mask"] is not None else None
        else:
            prompt_embeds = prompt_embeds.squeeze(0)
            prompt_mask = prompt_mask.squeeze(0) if prompt_mask is not None else None
            prompt_embeds_2 = prompt_embeds_2.squeeze(0) if prompt_embeds_2 is not None else None
            extra_kwargs["byt5_text_states"] = extra_kwargs["byt5_text_states"].squeeze(0) if extra_kwargs["byt5_text_states"] is not None else None
            extra_kwargs["byt5_text_mask"] = extra_kwargs["byt5_text_mask"].squeeze(0) if extra_kwargs["byt5_text_mask"] is not None else None
        
        for j in tqdm(
            train_timesteps_list,
            desc=f"GRPO TimeStep Training, inner_step {inner_step+1}",
            position=1,
            leave=False,
            disable=rank >= 1,
        ):
            # ========== Forward pass: compute new log probs ==========
            new_log_probs, prev_means, std_devs = grpo_one_step(
                args, batch, vae, text_encoder, text_encoder_2, byt5_kwarg, extra_model, model,
                sample["latents"][:, j],
                sample["next_latents"][:, j],
                sample["timesteps"][:, j],
                sigma_schedule,
                logger,
                prompt_embeds=prompt_embeds,
                prompt_mask=prompt_mask,
                prompt_embeds_2=prompt_embeds_2,
                extra_kwargs=extra_kwargs,
            )
            # ========== KL divergence computation (optional) ==========
            # KL regularization (for consistency with multimodal): requires the reference model mean μ_ref; otherwise skip
            kl_w_base = float(getattr(args, "kl_weight", 0.0) or 0.0)
            kl_coef = float(getattr(args, "kl_coef", 1e-7) or 0.0)
            use_kl = (kl_w_base > 0.0) and (ref_model is not None)

            # Adaptive KL weight controller (starts small and gradually increases)
            # Only build/use it when KL path is explicitly enabled.
            if use_kl and (not hasattr(args, "_kl_controller")):
                min_kl_coef = float(getattr(args, "kl_min_coef", kl_coef) or kl_coef)
                min_kl_coef = max(0.0, min(min_kl_coef, kl_w_base))
                init_kl_coef = max(min_kl_coef, min(kl_coef, kl_w_base))
                args._kl_controller = AdaptiveKLController(
                    target_kl=kl_w_base,
                    init_kl_coef=init_kl_coef,
                    min_kl_coef=min_kl_coef,
                    max_kl_coef=kl_w_base
                )
                logger.info(
                    f"Rank {rank}, sp_rank {sp_rank}: init adaptive KL controller "
                    f"(init={init_kl_coef:.3e}, min={min_kl_coef:.3e}, max={kl_w_base:.3e})"
                )
            
            kl_w = args._kl_controller.kl_coef if use_kl else 0.0
            # kl_w = kl_w_base
            
            # Dual KL Settings：
            # - kl_loss: KL based on ref_model mean, can be Fixed (use_moving_KL=False) or Moving (use_moving_KL=True)
            # - kl_stepwise: Step-wise KL based on old/new log_probs (old model vs new model sampling)
            # Dual KL Supports Two Modes:
            #   1. Fixed + step-wise: use_moving_KL=False, use_dual_kl=True
            #   2. Moving N steps + step-wise: use_moving_KL=True, update_ref_model_step=N, use_dual_kl=True
            use_dual_kl = getattr(args, "use_dual_kl", False)
            dual_moving_weight = float(getattr(args, "dual_kl_moving_weight", 1.0))
            dual_step_weight = float(getattr(args, "dual_kl_step_weight", 0.1))
            
            kl_loss = torch.tensor(0.0, device=device, requires_grad=False)  # Fixed/Moving KL: KL(π_current || π_ref)
            kl_stepwise = torch.tensor(0.0, device=device, requires_grad=False)  # Step-wise KL: KL(π_old_sampling || π_current)
            effective_kl = torch.tensor(0.0, device=device, requires_grad=False)  # Combined KL (Fixed/Moving + Step-wise if dual_kl enabled)
            
            if use_kl and kl_w > 0 and isinstance(prev_means, torch.Tensor) and isinstance(std_devs, torch.Tensor):
                # Get KL computation mode
                kl_compute_mode = getattr(args, "kl_compute_mode", "rollout_phase")
                
                # Choose computation method based on mode
                if kl_compute_mode == "rollout_phase" and "ref_means" in sample:
                    # Use pre-computed reference statistics from rollout phase
                    # Pre-computed reference statistics: [mini_batch, num_steps, ...]
                    # Extract the j-th timestep for all samples in this mini-batch
                    prev_means_ref = sample["ref_means"][:, j]  # [mini_batch, ...]
                else:
                    # Compute on-the-fly during training
                    # This happens when:
                    # 1. kl_compute_mode == "training_phase" (explicitly requested)
                    # 2. kl_compute_mode == "rollout_phase" but pre-computation was skipped/disabled
                    if kl_compute_mode == "training_phase":
                        # Explicitly requested on-the-fly computation
                        pass  # Continue to compute below
                    else:
                        # Fallback: should not happen if rollout mode worked correctly
                        logger.warning(f'Rank {rank}, sp_rank {sp_rank}: Using on-the-fly KL computation (fallback mode). '
                                     f'Expected pre-computed ref_means not found.')
                    with torch.no_grad():
                        # Offload models if needed to manage memory
                        if reference_mode_offload:
                            logger.info(f'Rank {rank}, sp_rank {sp_rank}: offloading model to CPU for KL computation')
                            model = model.to('cpu')
                            ref_model = ref_model.to(device)
                            torch.cuda.empty_cache()
                        
                        _, prev_means_ref, _ = grpo_one_step(
                            args, batch, vae, text_encoder, text_encoder_2, byt5_kwarg, extra_model, ref_model,
                            sample["latents"][:, j],
                            sample["next_latents"][:, j],
                            sample["timesteps"][:, j],
                            sigma_schedule,
                            logger,
                            prompt_embeds=prompt_embeds,
                            prompt_mask=prompt_mask,
                            prompt_embeds_2=prompt_embeds_2,
                            extra_kwargs=extra_kwargs,
                        )
                        
                        # Restore models after reference computation
                        if reference_mode_offload:
                            logger.info(f'Rank {rank}, sp_rank {sp_rank}: offloading ref_model to CPU after KL computation')
                            model = model.to(device)
                            ref_model = ref_model.to('cpu')
                            torch.cuda.empty_cache()
                
                # Compute Fixed/Moving KL loss if we have valid means and stds
                if isinstance(prev_means_ref, torch.Tensor):
                    # Align with Gemini implementation:
                    # 1. Compute squared difference
                    # 2. Mean over spatial dimensions (keepdim=True)
                    # 3. Divide by (2 * std_dev_t^2)
                    # 4. Final mean over all dimensions
                    # Ensure shapes are exactly aligned to avoid silent broadcasting bugs
                    assert prev_means.shape == prev_means_ref.shape, (
                        f"Shape mismatch in KL: {prev_means.shape} vs {prev_means_ref.shape}"
                    )
                    mu_diff_sq = (prev_means.to(torch.float32) - prev_means_ref.to(device=device, dtype=torch.float32)) ** 2
                    
                    # For video data: spatial dims are all non-batch dims except channel
                    spatial_dims = tuple(range(1, mu_diff_sq.ndim))
                    mu_diff_sq_mean = mu_diff_sq.mean(dim=spatial_dims, keepdim=True)
                    
                    # Divide by (2 * std_dev_t^2) - ensure std_devs is broadcastable
                    std_dev_t_sq = std_devs.to(device=device, dtype=torch.float32) ** 2 + 1e-6
                    while std_dev_t_sq.ndim < mu_diff_sq_mean.ndim:
                        std_dev_t_sq = std_dev_t_sq.unsqueeze(-1)
                    
                    kl_step = mu_diff_sq_mean / (2 * std_dev_t_sq)
                    kl_loss = torch.mean(kl_step)  # Fixed/Moving KL: KL(π_current || π_ref)
                    
                    # Step-wise KL: KL(π_old_sampling || π_current)
                    if use_dual_kl:
                        # KL(old || new) ≈ E_old[log π_old - log π_new] (old model vs new model sampling)
                        old_logp = sample["log_probs"][:, j]
                        kl_stepwise = torch.mean(old_logp - new_log_probs)
                        effective_kl = dual_moving_weight * kl_loss + dual_step_weight * kl_stepwise
                    else:
                        effective_kl = kl_loss
                    
                    # Update KL controller with current total KL (without weight kl_w)
                    args._kl_controller.update(effective_kl)
                    kl_beta = kl_w  # Use adaptive KL weight
                else:
                    effective_kl = torch.tensor(0.0, device=device, requires_grad=False)
                    kl_beta = 0.0
            else:
                # No KL computation needed
                kl_beta = 0.0
            
            # ========== Policy loss computation ==========
            # Clamp advantages to prevent extreme values
            advantages = torch.clamp(
                sample["avg_advantages"][:, j],
                -adv_clip_max,
                adv_clip_max,
            )
            
            # Compute probability ratio and PPO-style clipped loss
            ratio = torch.exp(new_log_probs - sample["log_probs"][:, j])
            unclipped_loss = -advantages * ratio
            clipped_loss = -advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
            policy_loss_per = torch.maximum(unclipped_loss, clipped_loss)

            # Optional per-step balancing scaling (per-sample)
            if per_step_balancing is not None:
                diffusion_indices = sample["timestep_perms"][:, j].long()
                loss_scale = per_step_balancing[diffusion_indices].to(policy_loss_per.dtype)
                policy_loss_per = policy_loss_per * loss_scale

            policy_loss = torch.mean(policy_loss_per)


            # Total loss: policy loss + KL regularization
            loss = policy_loss + kl_beta * effective_kl if (kl_beta > 0 and effective_kl is not None) else policy_loss
            
            # ========== Collect training metrics ==========
            log_prob_diff = new_log_probs - sample["log_probs"][:, j]
            info["approx_kl"].append(0.5 * torch.mean(log_prob_diff ** 2))
            info["clipfrac"].append(torch.mean((torch.abs(ratio - 1.0) > clip_range).float()))
            info["clipfrac_gt_one"].append(torch.mean((ratio - 1.0 > clip_range).float()))
            info["clipfrac_lt_one"].append(torch.mean((1.0 - ratio > clip_range).float()))
            info["policy_loss"].append(policy_loss)
            info["total_loss"].append(loss)
            if kl_beta > 0 and effective_kl is not None:
                # Keep as Tensor for torch.stack(), convert to scalar later
                if isinstance(effective_kl, torch.Tensor):
                    info["kl_loss"].append(effective_kl.detach())
                else:
                    info["kl_loss"].append(torch.tensor(effective_kl, device=device, dtype=torch.float32))
                if use_dual_kl:
                    if isinstance(kl_loss, torch.Tensor):
                        info["kl_moving"].append(kl_loss.detach())
                    else:
                        info["kl_moving"].append(torch.tensor(kl_loss, device=device, dtype=torch.float32))
                    if isinstance(kl_stepwise, torch.Tensor):
                        info["kl_stepwise"].append(kl_stepwise.detach())
                    else:
                        info["kl_stepwise"].append(torch.tensor(kl_stepwise, device=device, dtype=torch.float32))

            # ========== Backward pass and optimization ==========
            # Scale loss by gradient accumulation steps
            final_loss = loss / args.gradient_accumulation_steps
            final_loss.backward()
            if debug_grad_flow and tracked_param is not None and tracked_param.grad is not None:
                current_step = int(getattr(scalar_states, "update_steps", 0))
                if current_step % debug_diag_interval == 0:
                    grad_abs = tracked_param.grad.detach().abs()
                    logger.info(
                        f"Rank {rank}, sp_rank {sp_rank}: grad[{tracked_name}] "
                        f"mean={grad_abs.mean().item():.3e}, max={grad_abs.max().item():.3e}"
                    )

            # When mini_batch_size_per_update > 1, each inner loop processes multiple samples,
            # so train_steps should increase by mini_batch_size instead of 1
            scalar_states.add(train_steps=mini_batch_size)
            inner_step += mini_batch_size

            # Perform optimizer step if we've accumulated enough gradients
            is_update_step = scalar_states.train_steps % args.gradient_accumulation_steps == 0
            if is_update_step:
                tracked_before = None
                if tracked_param is not None:
                    try:
                        # FSDP shards params: some ranks may have empty/invalid storage; skip safely
                        n = min(4096, tracked_param.numel())
                        if n > 0:
                            tracked_before = tracked_param.detach().reshape(-1)[:n].float().clone()
                    except (RuntimeError, ValueError):
                        tracked_before = None
                grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.max_grad_norm,
                    foreach=True
                ).item()
                
                optimizer.step()
                
                lr_scheduler.step()
                optimizer.zero_grad()
                
                scalar_states.add(update_steps=1, current_run_update_steps=1)
                scalar_states.lr = optimizer.param_groups[0]["lr"]
                if tracked_before is not None and tracked_param is not None:
                    try:
                        n = min(4096, tracked_param.numel(), tracked_before.numel())
                        tracked_after = tracked_param.detach().reshape(-1)[:n].float()
                        delta = (tracked_after - tracked_before[:n]).abs()
                        param_delta_mean = delta.mean().item()
                        param_delta_max = delta.max().item()
                        nnz = int((delta > 0).sum().item())
                        if debug_grad_flow:
                            logger.info(
                                f"Rank {rank}, sp_rank {sp_rank}: param_update[{tracked_name}] "
                                f"delta_mean={param_delta_mean:.3e}, delta_max={param_delta_max:.3e}, changed_elems={nnz}/{delta.numel()}"
                            )
                            if nnz == 0:
                                logger.warning(
                                    f"Rank {rank}, sp_rank {sp_rank}: tracked parameter did not change after optimizer.step(). "
                                    "Check grad flow, optimizer states, and accumulation schedule."
                                )
                    except (RuntimeError, ValueError):
                        pass  # FSDP/invalid storage; skip param_delta tracking
                logger.info(f'Rank {rank}, sp_rank {sp_rank}: update step: {scalar_states.update_steps}, train_steps: {scalar_states.train_steps}')
    
    grpo_loop_end_time = sync_cuda_time()
    grpo_loop_time = grpo_loop_end_time - grpo_loop_start_time

    # ==================== Phase 8: Aggregate Metrics and Return ====================
    # Aggregate training metrics across all timesteps
    info_aggregated = {}
    for k, v in info.items():
        if len(v) == 0:
            # Skip empty lists
            continue
        # Ensure all elements are tensors before stacking
        tensor_list = []
        for item in v:
            if isinstance(item, torch.Tensor):
                tensor_list.append(item)
            elif isinstance(item, (int, float)):
                # Convert scalar to tensor
                tensor_list.append(torch.tensor(item, device=device, dtype=torch.float32))
            else:
                raise TypeError(f"Unexpected type in info['{k}']: {type(item)}")
        if len(tensor_list) > 0:
            info_aggregated[k] = torch.mean(torch.stack(tensor_list))
    
    # Reduce metrics across all distributed ranks
    for key, value in info_aggregated.items():
        if isinstance(value, torch.Tensor):
            dist.all_reduce(value, op=dist.ReduceOp.AVG)
            info_aggregated[key] = value.item()
        else:
            info_aggregated[key] = value
    
    # Reduce timing metrics across ranks
    def reduce_time_metric(time_value):
        """Helper to reduce time metrics across ranks."""
        time_tensor = torch.tensor(time_value, device=device, dtype=torch.float32)
        dist.all_reduce(time_tensor, op=dist.ReduceOp.AVG)
        return time_tensor.item()
    
    rollout_time_avg = reduce_time_metric(rollout_time)
    grpo_loop_time_avg = reduce_time_metric(grpo_loop_time)
    ref_model_time_avg = reduce_time_metric(ref_model_time)
    
    dist.barrier()
    
    # Build return dictionary with all metrics
    return_dict = {
        "grad_norm": grad_norm,
        "KL_weight": kl_beta,
        "param_delta_mean": param_delta_mean,
        "param_delta_max": param_delta_max,
        "rollout_time": rollout_time_avg,
        "grpo_loop_time": grpo_loop_time_avg,
        "ref_model_time": ref_model_time_avg,
    }
    
    # Add aggregated training metrics
    return_dict.update(info_aggregated)
    
    # Add reward statistics
    for metric_name in reward_scores.keys():
        gathered = gathered_reward_stats.get(metric_name)
        if gathered is not None:
            return_dict[f"gathered_{metric_name}_reward_mean"] = torch.nanmean(gathered).item()
            return_dict[f"gathered_{metric_name}_reward_std"] = nanstd(gathered)
    
    return return_dict
