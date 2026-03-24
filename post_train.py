# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

"""
HunyuanVideo-1.5 Training Script

This script provides a complete training pipeline for HunyuanVideo-1.5 model.

Quick Start:
1. Implement your own dataloader:
   - Replace the `create_dummy_dataloader()` function with your own implementation
   - Your dataset's __getitem__ method should return a single sample:
     * "pixel_values": torch.Tensor - Video: [C, F, H, W] or Image: [C, H, W]
       Pixel values must be in range [-1, 1] 
       Note: For video data, temporal dimension F must be 4n+1 (e.g., 1, 5, 9, 13, 17, ...)
     * "text": str - Text prompt for this sample
     * "data_type": str - "video" or "image"
     * Optional: "latents" - Pre-encoded VAE latents for faster training
     * Optional: "byt5_text_ids" and "byt5_text_mask" - Pre-tokenized byT5 inputs
   - See `create_dummy_dataloader()` function for detailed format documentation

2. Configure training parameters:
   - Set `--pretrained_model_root` to your pretrained model path
   - Adjust training hyperparameters (learning_rate, batch_size, etc.)
   - Configure distributed training settings (sp_size, enable_fsdp, etc.)

3. Run training:
   - Single GPU: python train.py --pretrained_model_root <path> [other args]
   - Multi-GPU: torchrun --nproc_per_node=N train.py --pretrained_model_root <path> [other args]

4. Monitor training:
   - Checkpoints are saved to `output_dir` at intervals specified by `--save_interval`
   - Validation videos are generated at intervals specified by `--validation_interval`
   - Training logs are printed to console at intervals specified by `--log_interval`

5. Resume training:
   - Use `--resume_from_checkpoint <checkpoint_dir>` to resume from a saved checkpoint

For detailed format requirements, see the docstring of `create_dummy_dataloader()` function.
"""

import os
import random
import math
import argparse
import datetime
import csv
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union
from enum import Enum

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
)
from diffusers.optimization import get_scheduler
from loguru import logger
import einops
import imageio

from hyvideo.pipelines.hunyuan_video_pipeline import HunyuanVideo_1_5_Pipeline
from hyvideo.pipelines.hunyuan_video_grpo_pipeline import get_post_train_video_dataloader
from hyvideo.commons.parallel_states import get_parallel_state, initialize_parallel_state
from hyvideo.optim.muon import get_muon_optimizer

from hyvideo.models.reward_models.rewards import get_reward_fn
from hyvideo.utils.grpo_utils import (
    prepare_samples_online as grpo_prepare_samples_online,
    train_one_step as grpo_train_one_step,
)

from torch.distributed._composable.fsdp import (
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)

from transformers.utils import import_utils
from transformers.utils.import_utils import _is_package_available
from packaging import version
import importlib.metadata
from functools import lru_cache

@lru_cache
def is_torch_greater_or_equal(library_version: str, accept_dev: bool = True) -> bool:
    if not _is_package_available("torch"):
        return False

    if accept_dev:
        return version.parse(version.parse(importlib.metadata.version("torch")).base_version) >= version.parse(
            library_version
        )
    else:
        return version.parse(importlib.metadata.version("torch")) >= version.parse(library_version)


def _get_transformer_dtype(dtype_str: str):
    """Map dtype string to torch dtype for transformer."""
    _DTYPE_MAP = {"bf16": torch.bfloat16, "fp32": torch.float32}
    if dtype_str not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return _DTYPE_MAP[dtype_str]


def patch_transformers_is_torch_greater_or_equal():
    if hasattr(import_utils, 'is_torch_greater_or_equal'):
        import_utils.is_torch_greater_or_equal = is_torch_greater_or_equal

patch_transformers_is_torch_greater_or_equal()


class SNRType(str, Enum):
    UNIFORM = "uniform"
    LOGNORM = "lognorm"
    MIX = "mix"
    MODE = "mode"


def str_to_bool(value):
    """Convert string to boolean, supporting true/false, 1/0, yes/no.
    If value is None (when flag is provided without value), returns True."""
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.lower().strip()
        if value in ('true', '1', 'yes', 'on'):
            return True
        elif value in ('false', '0', 'no', 'off'):
            return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value}")


def save_video(video: torch.Tensor, path: str):
    if video.ndim == 5:
        assert video.shape[0] == 1, f"Expected batch size 1, got {video.shape[0]}"
        video = video[0]
    vid = (video * 255).clamp(0, 255).to(torch.uint8)
    vid = einops.rearrange(vid, 'c f h w -> f h w c')
    imageio.mimwrite(path, vid.cpu().numpy(), fps=24)


@dataclass
class TrainingConfig:
    # Model paths
    pretrained_model_root: str
    pretrained_transformer_version: str = "720p_t2v"
    post_train_type: str = "grpo"  # "grpo" or "standard"
    
    # Training parameters
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_steps: int = 10000
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    use_muon: bool = True
    
    # Diffusion parameters
    num_train_timesteps: int = 1000
    train_timestep_shift: float = 5.0
    validation_timestep_shift: float = 3.0
    snr_type: SNRType = SNRType.LOGNORM  # Timestep sampling strategy: uniform, lognorm, mix, or mode
    
    # Task configuration
    task_type: str = "t2v"  # "t2v" or "i2v"
    i2v_prob: float = 0.3  # Probability of using i2v task when data_type is video (default: 0.3 for video training)
    
    # FSDP configuration
    enable_fsdp: bool = True  # Enable FSDP for distributed training
    enable_gradient_checkpointing: bool = True  # Enable gradient checkpointing
    sp_size: int = 2  # Sequence parallelism size (must divide world_size evenly)
    dp_replicate: int = 1  # Data parallelism replicate size (must divide world_size evenly)
    
    # Data configuration
    batch_size: int = 2
    num_workers: int = 4
    prefetch_factor: int = 2
    num_generations: int = 4
    use_same_noise: bool = False
    
    # Output configuration
    output_dir: str = "./outputs"
    save_interval: int = 100
    log_interval: int = 1
    
    # Device configuration
    dtype: str = "bf16"  # "bf16" or "fp32"
    master_weight_type: str = "fp32"  # "fp32" or "bf16", controls FSDP parameter dtype
    
    # Seed
    seed: int = 42
    validation_global_seed: int = 930
    
    # Validation configuration
    validation_interval: int = 50  # Run validation every N steps
    validation_prompts: Optional[List[str]] = None  # Prompts for validation (default: single prompt)
    validate_video_length: int = 81  # Video length (number of frames) for validation
    validation_aspect_ratio: str = "16:9"  # Aspect ratio for validation inference
    validation_fixed_size: Optional[str] = "480x864"  # Fixed output size for validation, e.g. "480x864"
    validation_num_inference_steps: Optional[int] = 40  # Number of inference steps for validation
    validation_negative_prompt: str = ""  # Negative prompt for validation inference
    validation_enable_sr: bool = False  # Whether to enable SR during validation
    validation_prompt_rewrite: bool = False  # Whether to enable prompt rewrite during validation
    validate_at_step0: bool = True  # Whether to run validation at step 0
    
    # Resume training configuration
    resume_from_checkpoint: Optional[str] = None  # Path to checkpoint directory to resume from
    
    # LoRA configuration
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: Optional[List[str]] = None  # Target modules for LoRA (default: all Linear layers)
    pretrained_lora_path: Optional[str] = None
    
    # Data loading configuration
    train_video_csv: Union[str, List[str]] = "assets/demo_train.csv"
    valid_video_csv: Union[str, List[str]] = "assets/demo_sample.csv"
    
    # Reward configuration
    reward_model: str = "videoalign_local"
    reward_config: Optional[Dict[str, Any]] = None
    reward_checkpoint_mode: str = "v3"
    reward_weights: Optional[Union[Dict[str, float], str]] = '{"VQ":0.5,"MQ":0.5,"TA":1.0}'
    remote_reward_url: Optional[str] = None

    # GRPO configuration
    enable_grpo_policy_update: bool = True
    grpo_sampling_steps: int = 20
    sample_n_frames: int = 81
    eta: float = 0.5
    cfg_scale: float = 6.0
    neg_prompt: Optional[str] = None
    data_type: str = "video"
    model_type: str = "16164"
    video_fps: int = 24
    vae_name: str = "16164-32c-hy20250605"
    vae_tiling: bool = False
    infer_flow_shift_video: float = 5.0
    video_bucket_hw_base_size: int = 480
    video_bucket_hw_bucket_stride: int = 32
    multitask_mask_training_type: Optional[str] = "concat"
    vision_num_semantic_tokens: Optional[int] = 729
    vision_states_dim: Optional[int] = 1152
    sde_type: str = "sage_grpo"
    mini_batch_size_per_update: int = 1
    clip_range: float = 1e-4
    adv_clip_max: float = 5.0
    timestep_fraction: float = 1.0
    use_grad_balancing: bool = True
    enable_timestep_permutation: bool = True
    reference_mode_offload: bool = False
    kl_weight: float = 1e-5
    kl_coef: float = 1e-7
    kl_min_coef: float = 1e-7
    use_moving_KL: bool = True
    update_ref_model_step: int = 10
    use_dual_kl: bool = True
    dual_kl_moving_weight: float = 1.0
    dual_kl_step_weight: float = 0.1
    kl_compute_mode: str = "rollout_phase"
    glyph_byT5_v2: bool = True
    byt5_max_length: int = 256
    video_uncond_p_byt5: float = 0.0

    precision: str = "bf16"
    vae_precision: str = "fp16"
    val_disable_autocast: bool = False
    debug_grad_flow: bool = False
    debug_train_diagnostics_interval: int = 1


@dataclass
class ScalarStatesLite:
    train_steps: int = 0
    update_steps: int = 0
    current_run_update_steps: int = 0
    lr: float = 0.0

    def add(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)
            else:
                setattr(self, k, getattr(self, k) + v)

class LinearInterpolationSchedule:
    """Simple linear interpolation schedule for flow matching"""
    def __init__(self, T: int = 1000):
        self.T = T
    
    def forward(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Linear interpolation: x_t = (1 - t/T) * x0 + (t/T) * x1
        Args:
            x0: starting point (clean latents)
            x1: ending point (noise)
            t: timesteps
        """
        t_normalized = t / self.T
        t_normalized = t_normalized.view(-1, *([1] * (x0.ndim - 1)))
        return (1 - t_normalized) * x0 + t_normalized * x1


class TimestepSampler:

    TRAIN_EPS = 1e-5
    SAMPLE_EPS = 1e-3
    
    def __init__(
        self, 
        T: int = 1000, 
        device: torch.device = None,
        snr_type: SNRType = SNRType.LOGNORM,
    ):
        self.T = T
        self.device = device
        self.snr_type = SNRType(snr_type) if isinstance(snr_type, str) else snr_type
    
    def _check_interval(self, eval: bool = False):
        # For ICPlan-like path with velocity model, use [eps, 1-eps]
        eps = self.SAMPLE_EPS if eval else self.TRAIN_EPS
        t0 = eps
        t1 = 1.0 - eps
        return t0, t1
    
    def sample(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        if device is None:
            device = self.device if self.device is not None else torch.device("cuda")
        
        t0, t1 = self._check_interval(eval=False)
        
        if self.snr_type == SNRType.UNIFORM:
            # Uniform sampling: t = rand() * (t1 - t0) + t0
            t = torch.rand((batch_size,), device=device) * (t1 - t0) + t0
            
        elif self.snr_type == SNRType.LOGNORM:
            # Log-normal sampling: t = 1 / (1 + exp(-u)) * (t1 - t0) + t0
            u = torch.normal(mean=0.0, std=1.0, size=(batch_size,), device=device)
            t = 1.0 / (1.0 + torch.exp(-u)) * (t1 - t0) + t0
            
        elif self.snr_type == SNRType.MIX:
            # Mix sampling: 30% lognorm + 70% clipped uniform
            u = torch.normal(mean=0.0, std=1.0, size=(batch_size,), device=device)
            t_lognorm = 1.0 / (1.0 + torch.exp(-u)) * (t1 - t0) + t0
            
            # Clipped uniform: delta = 0.0 (0.0~0.01 clip)
            delta = 0.0
            t0_clip = t0 + delta
            t1_clip = t1 - delta
            t_clip_uniform = torch.rand((batch_size,), device=device) * (t1_clip - t0_clip) + t0_clip
            
            # Mix with 30% lognorm, 70% uniform
            mask = (torch.rand((batch_size,), device=device) > 0.3).float()
            t = mask * t_lognorm + (1 - mask) * t_clip_uniform
            
        elif self.snr_type == SNRType.MODE:
            # Mode sampling: t = 1 - u - mode_scale * (cos(pi * u / 2)^2 - 1 + u)
            mode_scale = 1.29
            u = torch.rand(size=(batch_size,), device=device)
            t = 1.0 - u - mode_scale * (torch.cos(math.pi * u / 2.0) ** 2 - 1.0 + u)
            # Scale to [t0, t1] range
            t = t * (t1 - t0) + t0
        else:
            raise ValueError(f"Unknown SNR type: {self.snr_type}")
        
        # Scale to [0, T] range
        timesteps = t * self.T
        return timesteps


def timestep_transform(timesteps: torch.Tensor, T: int, shift: float = 1.0) -> torch.Tensor:
    """Transform timesteps with shift"""
    if shift == 1.0:
        return timesteps
    timesteps_normalized = timesteps / T
    timesteps_transformed = shift * timesteps_normalized / (1 + (shift - 1) * timesteps_normalized)
    return timesteps_transformed * T


def is_src(src, group_src, group):
    assert src is not None or group_src is not None
    assert src is None or group_src is None
    if src is not None:
        return dist.get_rank() == src
    if group_src is not None:
        return dist.get_rank() == dist.get_global_rank(group, group_src)
    raise RuntimeError("src and group_src cannot be both None")

def broadcast_object(
        obj,
        src = None,
        group = None,
        device = None,
        group_src = None,
):
    kwargs = dict(
        src=src,
        group_src=group_src,
        group=group,
        device=device,
    )
    buffer = [obj] if is_src(src, group_src, group) else [None]

    dist.broadcast_object_list(buffer, **kwargs)
    return buffer[0]

def broadcast_tensor(
        tensor,
        src  = None,
        group = None,
        async_op: bool = False,
        group_src = None,
):
    """shape and dtype safe broadcast of tensor"""
    kwargs = dict(
        src=src,
        group_src=group_src,
        group=group,
        async_op=async_op,
    )
    if is_src(src, group_src, group):
        tensor = tensor.cuda().contiguous()
    if is_src(src, group_src, group):
        shape, dtype = tensor.shape, tensor.dtype
    else:
        shape, dtype = None, None
    shape = broadcast_object(shape, src=src, group_src=group_src, group=group)
    dtype = broadcast_object(dtype, src=src, group_src=group_src, group=group)

    buffer = tensor if is_src(src, group_src, group) else torch.empty(shape, device='cuda', dtype=dtype)
    dist.broadcast(buffer, **kwargs)
    return buffer


def sync_tensor_for_sp(tensor: torch.Tensor, sp_group) -> torch.Tensor:
    """
    Sync tensor within sequence parallel group.
    Ensures all ranks in the SP group have the same tensor values.
    """
    if sp_group is None:
        return tensor
    if not isinstance(tensor, torch.Tensor):
        obj_list = [tensor]
        dist.broadcast_object_list(obj_list, group_src=0, group=sp_group)
        return obj_list[0]
    return broadcast_tensor(tensor, group_src=0, group=sp_group)


def _materialize_state_dict_to_cpu(state_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Convert state_dict values to plain CPU tensors.
    FSDP2 policy model yields DTensors from get_model_state_dict(); ref_model may have
    plain Tensors (e.g. when offloaded) or DTensors. load_state_dict fails with
    "got mixed torch.Tensor and DTensor" when types differ across GPU arch / world_size.
    Materializing to CPU tensors avoids this.
    """
    result = {}
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            result[k] = v
            continue
        # DTensor: gather full tensor then move to CPU
        if hasattr(v, "full_tensor"):
            try:
                result[k] = v.full_tensor().cpu()
            except Exception:
                result[k] = v.cpu()
        else:
            result[k] = v.cpu()
    return result


def maybe_update_ref_model_for_kl(args, model, ref_model, scalar_states, logger_obj):
    """
    Refresh ref_model weights using current policy weights at a fixed interval.
    This matches the moving-KL behavior from post_train_puretorch:
    - use_moving_KL=False: fixed ref model.
    - use_moving_KL=True: update every `update_ref_model_step` optimizer update steps.
    """
    if not getattr(args, "use_moving_KL", False):
        return ref_model

    step_interval = int(getattr(args, "update_ref_model_step", 0) or 0)
    if step_interval <= 0:
        return ref_model

    update_steps = int(getattr(scalar_states, "update_steps", 0))
    if update_steps <= 0 or (update_steps % step_interval) != 0:
        return ref_model

    if ref_model is None:
        logger_obj.error("[Moving KL] ref_model is None when trying to update.")
        return ref_model

    try:
        state_dict = get_model_state_dict(model)
        state_dict = _materialize_state_dict_to_cpu(state_dict)
        missing, unexpected = ref_model.load_state_dict(state_dict, strict=False)
        logger_obj.info(
            f"[Moving KL] Updated ref_model at update_step={update_steps} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
        if getattr(args, "reference_mode_offload", False):
            ref_model.to("cpu")
    except Exception as exc:
        logger_obj.error(f"[Moving KL] Failed to update ref_model: {exc}", exc_info=True)
    return ref_model


class HunyuanVideoTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if "RANK" in os.environ:
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
            self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            self.device = torch.device(f"cuda:{self.local_rank}")
            self.is_main_process = self.rank == 0
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
            self.is_main_process = True
        
        if config.sp_size > self.world_size:
            raise ValueError(
                f"sp_size ({config.sp_size}) cannot be greater than world_size ({self.world_size})"
            )
        if self.world_size % config.sp_size != 0:
            raise ValueError(
                f"sp_size ({config.sp_size}) must evenly divide world_size ({self.world_size}). "
                f"world_size % sp_size = {self.world_size % config.sp_size}"
            )
        
        initialize_parallel_state(sp=config.sp_size, dp_replicate=config.dp_replicate)
        torch.cuda.set_device(self.local_rank)
        self.parallel_state = get_parallel_state()
        self.dp_rank = self.parallel_state.world_mesh['dp'].get_local_rank()
        self.dp_size = self.parallel_state.world_mesh['dp'].size()
        self.sp_enabled = self.parallel_state.sp_enabled
        self.sp_rank = self.parallel_state.sp_rank
        self.sp_group = self.parallel_state.sp_group if self.sp_enabled else None
        # Align rank mapping with grpo_utils.prepare_samples_online assumptions:
        # dp_rank = global_rank // sp_size, sp_rank = global_rank % sp_size.
        if self.world_size > 1:
            expected_sp_size = max(int(self.config.sp_size), 1)
            expected_dp_rank = int(self.rank // expected_sp_size)
            expected_dp_size = int(self.world_size // expected_sp_size)
            expected_sp_rank = int(self.rank % expected_sp_size)
            if self.dp_rank != expected_dp_rank or self.sp_rank != expected_sp_rank:
                if self.is_main_process:
                    logger.warning(
                        f"Override dp/sp rank mapping for GRPO compatibility: "
                        f"parallel_state(dp_rank={self.dp_rank}, sp_rank={self.sp_rank}) -> "
                        f"expected(dp_rank={expected_dp_rank}, sp_rank={expected_sp_rank})"
                    )
                self.dp_rank = expected_dp_rank
                self.sp_rank = expected_sp_rank
                self.dp_size = expected_dp_size
        if dist.is_initialized() and self.world_size > 1:
            output_dir_list = [config.output_dir if self.is_main_process else None]
            dist.broadcast_object_list(output_dir_list, src=0)
            config.output_dir = output_dir_list[0]

        self._set_seed(config.seed + self.dp_rank)
        self._build_models()
        self._build_optimizer()
        self.reward_inferencer = self._build_reward_inferencer() if self.config.post_train_type == "grpo" else None
        self.scalar_states = ScalarStatesLite(lr=self.config.learning_rate)
        self.grpo_args = self._build_grpo_args()
        
        self.noise_schedule = LinearInterpolationSchedule(T=config.num_train_timesteps)
        self.timestep_sampler = TimestepSampler(
            T=config.num_train_timesteps, 
            device=self.device,
            snr_type=config.snr_type,
        )
        
        self.global_step = 0
        self.current_epoch = 0
        self.tb_writer = None
        
        if self.is_main_process:
            os.makedirs(config.output_dir, exist_ok=True)
            tb_dir = os.path.join(config.output_dir, "tb")
            os.makedirs(tb_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=tb_dir)
        logs_dir = os.path.join(config.output_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        logger.add(
            os.path.join(logs_dir, f"training_rank_{self.rank}.log"),
            level="INFO",
            enqueue=True,
            backtrace=False,
            diagnose=False,
        )
        if self.is_main_process:
            logger.add(
                os.path.join(config.output_dir, "train.log"),
                level="INFO",
                enqueue=True,
                backtrace=False,
                diagnose=False,
            )
        if self.config.post_train_type == "grpo":
            logger.info(f"[RewardDebug] reward_weights(raw): {self.config.reward_weights}")
            logger.info(f"[RewardDebug] reward_config(effective): {self.config.reward_config}")
        
        self.validation_output_dir = os.path.join(config.output_dir, "samples")
        if self.is_main_process:
            os.makedirs(self.validation_output_dir, exist_ok=True)
        
        if config.validation_prompts is None:
            config.validation_prompts = ["A beautiful sunset over the ocean with waves gently crashing on the shore"]
        if self.is_main_process:
            self.print_training_configuration()
            logger.info(
                f"[Effective Config] world_size={self.world_size}, rank={self.rank}, local_rank={self.local_rank}, "
                f"sp_size={self.config.sp_size}, dp_size={self.dp_size}, dp_rank={self.dp_rank}, sp_rank={self.sp_rank}, "
                f"batch_size={self.config.batch_size}, num_generations={self.config.num_generations}, "
                f"grad_accum={self.config.gradient_accumulation_steps}"
            )

    def print_training_configuration(self):
        if not self.is_main_process:
            return

        line = "-" * 80
        logger.info("SYSTEM & HARDWARE CONFIGURATION")
        logger.info(line)
        logger.info(f"Number of GPUs                                   : {self.world_size}")
        logger.info(f"Local rank                                       : {self.local_rank}")
        logger.info(f"Global rank                                      : {self.rank}")
        logger.info(f"Device                                           : {self.device}")
        logger.info(f"Master weight dtype                              : {next(self.transformer.parameters()).dtype}")

        logger.info(line)
        logger.info("MODEL PARAMETERS & ARCHITECTURE")
        logger.info(line)
        total_params = sum(p.numel() for p in self.transformer.parameters())
        trainable_params = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        logger.info(f"Number of total                                  : {total_params:,}")
        logger.info(f"Total trainable parameters                        : {trainable_params:,}")
        logger.info(f"DIT model root                                   : {self.config.pretrained_model_root}")
        logger.info(f"Transformer version                              : {self.config.pretrained_transformer_version}")
        logger.info(f"Attention mode                                   : {getattr(self.transformer, 'attn_mode', 'unknown')}")
        logger.info(f"Gradient checkpointing                           : {self.config.enable_gradient_checkpointing}")
        logger.info(f"FSDP enabled                                     : {self.config.enable_fsdp and self.world_size > 1}")
        logger.info(f"LoRA enabled                                     : {self.config.use_lora}")
        logger.info(f"Glyph byT5 v2                                    : {self.config.glyph_byT5_v2}")
        logger.info(f"ByT5 max length                                  : {self.config.byt5_max_length}")

        logger.info(line)
        logger.info("DISTRIBUTED TRAINING CONFIGURATION")
        logger.info(line)
        logger.info(f"World size                                       : {self.world_size}")
        logger.info(f"DP degree                                        : {self.dp_size}")
        logger.info(f"DP rank                                          : {self.dp_rank}")
        logger.info(f"SP size                                          : {self.config.sp_size}")
        logger.info(f"SP rank                                          : {self.sp_rank}")

        logger.info(line)
        logger.info("BATCH SIZE & DATA FLOW CONFIGURATION")
        logger.info(line)
        logger.info(f"Per GPU batch size                               : {self.config.batch_size}")
        logger.info(f"DP degree batch size                             : {self.config.batch_size * self.dp_size}")
        logger.info(f"Gradient accumulation steps                      : {self.config.gradient_accumulation_steps}")
        logger.info(f"Num generations                                  : {self.config.num_generations}")
        logger.info(f"Use same noise                                   : {self.config.use_same_noise}")
        logger.info(f"Num workers                                      : {self.config.num_workers}")
        logger.info(f"Prefetch factor                                  : {self.config.prefetch_factor}")

        logger.info(line)
        logger.info("OPTIMIZATION CONFIGURATION")
        logger.info(line)
        logger.info(f"Optimizer                                        : {'muon' if self.config.use_muon else 'adamw'}")
        logger.info(f"Learning rate                                    : {self.config.learning_rate}")
        logger.info(f"Weight decay                                     : {self.config.weight_decay}")
        logger.info(f"Max gradient norm                                : {self.config.max_grad_norm}")
        logger.info(f"LR scheduler                                     : constant")
        logger.info(f"Warmup steps                                     : {self.config.warmup_steps}")
        logger.info(f"Max training steps                               : {self.config.max_steps}")

        logger.info(line)
        logger.info("INFERENCE & VALIDATION CONFIGURATION")
        logger.info(line)
        logger.info(f"Validation interval                              : {self.config.validation_interval}")
        logger.info(f"Validate at step 0                               : {self.config.validate_at_step0}")
        logger.info(f"Validation video length                          : {self.config.validate_video_length}")
        logger.info(f"Validation global seed                           : {self.config.validation_global_seed}")
        logger.info(f"Validation inference steps                        : {self.config.validation_num_inference_steps}")
        logger.info(f"Validation aspect ratio                          : {self.config.validation_aspect_ratio}")
        logger.info(f"Validation fixed size                            : {self.config.validation_fixed_size}")
        logger.info(f"Validation prompt rewrite                         : {self.config.validation_prompt_rewrite}")
        logger.info(f"Validation flow shift                             : {self.config.validation_timestep_shift}")
        logger.info(f"Sampling CFG scale                                : {self.config.cfg_scale}")
        logger.info(f"Sampling eta                                      : {self.config.eta}")
        logger.info(f"Sampling steps (GRPO rollout)                     : {self.config.grpo_sampling_steps}")
        logger.info(f"SDE type                                          : {self.config.sde_type}")

        logger.info(line)
        logger.info("GRPO & KL CONFIGURATION")
        logger.info(line)
        logger.info(f"Reward model                                     : {self.config.reward_model}")
        logger.info(f"Reward checkpoint mode                           : {self.config.reward_checkpoint_mode}")
        logger.info(f"KL weight                                        : {self.config.kl_weight}")
        logger.info(f"KL coef                                          : {self.config.kl_coef}")
        logger.info(f"Use moving KL                                    : {self.config.use_moving_KL}")
        logger.info(f"Update ref model step                            : {self.config.update_ref_model_step}")
        logger.info(f"Use dual KL                                      : {self.config.use_dual_kl}")
        logger.info(f"Dual KL moving weight                            : {self.config.dual_kl_moving_weight}")
        logger.info(f"Dual KL step weight                              : {self.config.dual_kl_step_weight}")
        logger.info(f"KL compute mode                                  : {self.config.kl_compute_mode}")
        logger.info(f"Use grad balancing                               : {self.config.use_grad_balancing}")
        logger.info(f"Enable timestep permutation                      : {self.config.enable_timestep_permutation}")
        logger.info(f"Reference mode offload                           : {self.config.reference_mode_offload}")

        logger.info(line)
        logger.info("LOGGING & CHECKPOINTING CONFIGURATION")
        logger.info(line)
        logger.info(f"Output directory                                 : {self.config.output_dir}")
        logger.info(f"Checkpoint interval                              : {self.config.save_interval}")
        logger.info(f"Log interval                                     : {self.config.log_interval}")
        logger.info(f"Resume checkpoint                                : {self.config.resume_from_checkpoint}")
        logger.info(f"Train CSV                                         : {self.config.train_video_csv}")
        logger.info(f"Valid CSV                                         : {self.config.valid_video_csv}")
        logger.info(f"Seed                                             : {self.config.seed}")

    def _log_metrics_to_tb(self, metrics: Dict[str, Any], step: int):
        if self.tb_writer is None or not self.is_main_process:
            return
        lr_val = float(
            metrics.get(
                "lr",
                self.lr_scheduler.get_last_lr()[0]
                if hasattr(self.lr_scheduler, "get_last_lr")
                else self.config.learning_rate,
            )
        )
        self.tb_writer.add_scalar("Train/learning_rate", lr_val, step)
        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            low_key = str(key).lower()
            if "reward" in low_key:
                tag = f"Reward/{key}"
            elif "kl" in low_key:
                tag = f"KL/{key}"
            elif "time" in low_key:
                tag = f"Time/{key}"
            else:
                tag = f"Train/{key}"
            self.tb_writer.add_scalar(tag, float(value), step)

    def _build_reward_inferencer(self):
        reward_model = getattr(self.config, "reward_model", "videoalign_local")
        self.config.reward_model = reward_model

        reward_weights_raw = getattr(self.config, "reward_weights", None)
        reward_weights = None
        if isinstance(reward_weights_raw, dict):
            reward_weights = reward_weights_raw
        elif isinstance(reward_weights_raw, str):
            text = reward_weights_raw.strip()
            if text:
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        reward_weights = parsed
                    else:
                        logger.warning(
                            f"reward_weights JSON is not a dict, ignored: {reward_weights_raw}"
                        )
                except Exception as exc:
                    logger.warning(
                        f"Failed to parse reward_weights JSON '{reward_weights_raw}', ignored. err={exc}"
                    )

        logger.info(f"Reward weights(raw): {reward_weights_raw}")
        logger.info(f"Reward weights(parsed): {reward_weights}")
        if isinstance(reward_weights, dict) and reward_weights:
            reward_config = {
                "models": {
                    reward_model: {
                        "weight": 1.0,
                        "sub_reward": reward_weights,
                    }
                }
            }
            if getattr(self.config, "reward_checkpoint_mode", None) is not None:
                reward_config["reward_checkpoint_mode"] = self.config.reward_checkpoint_mode
            if getattr(self.config, "remote_reward_url", None) is not None:
                reward_config["server_url"] = self.config.remote_reward_url
            self.config.reward_config = reward_config
            self.config.reward_weights = reward_weights
            logger.info(f"Reward config(overridden by reward_weights): {reward_config}")
        else:
            logger.warning(
                "reward_weights is empty/unavailable; reward model will use fallback/default sub_reward weights."
            )

        return get_reward_fn(self.config, self.device, logger)
    
    def _set_seed(self, seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_grpo_args(self):
        args_dict = dict(self.config.__dict__)
        neg_prompt = args_dict.get("neg_prompt", None)
        if isinstance(neg_prompt, str) and neg_prompt.strip() == "":
            neg_prompt = None
        args_dict["neg_prompt"] = neg_prompt
        args_dict.setdefault("global_seed", self.config.seed)
        args_dict["glyph_byT5_v2"] = bool(getattr(self.config, "glyph_byT5_v2", True))
        args_dict.setdefault("byt5_max_length", 256)
        args_dict.setdefault("video_uncond_p_byt5", 0.0)
        args_dict.setdefault("vae", self.config.vae_name)
        args_dict.setdefault("enable_global_metric_sync", False)
        args_dict.setdefault("enable_sp_consistency_check", False)
        args_dict.setdefault("sp_consistency_tolerance", 1e-5)
        args_dict.setdefault("use_moving_KL", True)
        args_dict.setdefault("update_ref_model_step", 10)
        args_dict.setdefault("use_dual_kl", True)
        args_dict.setdefault("dual_kl_moving_weight", 1.0)
        args_dict.setdefault("dual_kl_step_weight", 0.1)
        args_dict.setdefault("kl_min_coef", max(0.0, float(getattr(self.config, "kl_min_coef", getattr(self.config, "kl_coef", 1e-7)))))
        args_dict.setdefault("precision", self.config.precision)
        args_dict.setdefault("vae_precision", self.config.vae_precision)
        args_dict.setdefault("val_disable_autocast", self.config.val_disable_autocast)
        args_dict.setdefault("debug_grad_flow", bool(getattr(self.config, "debug_grad_flow", False)))
        args_dict.setdefault("debug_train_diagnostics_interval", int(getattr(self.config, "debug_train_diagnostics_interval", 1)))
        args_dict.setdefault("use_same_noise", bool(getattr(self.config, "use_same_noise", False)))

        img_in_proj = getattr(getattr(self.transformer, "img_in", None), "proj", None)
        expected_in_channels = getattr(img_in_proj, "in_channels", None)
        if expected_in_channels is not None:
            current_multitask_mode = args_dict.get("multitask_mask_training_type", None)
            if (current_multitask_mode is None or current_multitask_mode == "") and expected_in_channels > 32:
                args_dict["multitask_mask_training_type"] = "concat"
                if self.is_main_process:
                    logger.info(
                        f"Auto-set multitask_mask_training_type='concat' for GRPO "
                        f"(transformer expects in_channels={expected_in_channels})."
                    )

        return argparse.Namespace(**args_dict)
    
    def _build_models(self):
        transformer_dtype = _get_transformer_dtype(self.config.dtype)

        # Don't create SR pipeline for training (validation uses enable_sr=False)
        self.pipeline = HunyuanVideo_1_5_Pipeline.create_pipeline(
            pretrained_model_name_or_path=self.config.pretrained_model_root,
            transformer_version=self.config.pretrained_transformer_version,
            transformer_dtype=transformer_dtype,
            enable_offloading=False,
            enable_group_offloading=False,
            overlap_group_offloading=False,
            create_sr_pipeline=False,
            create_ref_model=float(getattr(self.config, "kl_weight", 0.0)) > 0.0,
            flow_shift=self.config.infer_flow_shift_video,
            device=self.device,
        )
        
        self.transformer = self.pipeline.transformer
        self.vae = self.pipeline.vae
        self.text_encoder = self.pipeline.text_encoder
        self.text_encoder_2 = self.pipeline.text_encoder_2
        self.vision_encoder = self.pipeline.vision_encoder
        self.byt5_kwargs = {
            "byt5_model": self.pipeline.byt5_model,
            "byt5_tokenizer": self.pipeline.byt5_tokenizer,
        }
        self.ref_model = getattr(self.pipeline, "ref_model", None)
        self.pipeline.noise_init_device = torch.device("cuda") if self.device.type == "cuda" else torch.device("cpu")
        if self.is_main_process:
            logger.info(f"Validation/infer noise_init_device: {self.pipeline.noise_init_device}")
        
        self.transformer.train()

        if self.config.use_lora:
            self._apply_lora()
        
        if self.config.enable_gradient_checkpointing:
            self._apply_gradient_checkpointing()
        
        # Offload ref_model to CPU before FSDP to avoid OOM (transformer + ref_model both on GPU)
        if self.ref_model is not None and bool(getattr(self.config, "reference_mode_offload", False)):
            self.ref_model = self.ref_model.to("cpu")
            if self.is_main_process:
                logger.info("ref_model offloaded to CPU (reference_mode_offload) to save GPU memory")
        
        if self.config.enable_fsdp and self.world_size > 1:
            self._apply_fsdp()

        if self.ref_model is not None:
            if self.is_main_process:
                logger.info(
                    f"KL regularization enabled: init ref_model done "
                    f"(use_moving_KL={self.config.use_moving_KL}, "
                    f"update_ref_model_step={self.config.update_ref_model_step}, "
                    f"use_dual_kl={self.config.use_dual_kl})."
                )
        if self.ref_model is None and self.is_main_process:
            logger.info("KL regularization is disabled (kl_weight=0).")
        
        if self.is_main_process:
            logger.info(f"Models loaded. Transformer dtype: {transformer_dtype}")
            total_params = sum(p.numel() for p in self.transformer.parameters())
            trainable_params = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
            logger.info(f"Transformer parameters: {total_params:,} (trainable: {trainable_params:,})")
            logger.info(f"LoRA enabled: {self.config.use_lora}")
            logger.info(f"FSDP enabled: {self.config.enable_fsdp and self.world_size > 1}")
            logger.info(f"Gradient checkpointing enabled: {self.config.enable_gradient_checkpointing}")
            logger.info(f"Timestep sampling strategy: {self.config.snr_type.value}")
    
    def _apply_lora(self):
        if self.is_main_process:
            logger.info("Applying LoRA to transformer using PeftAdapterMixin...")
        
        if self.config.pretrained_lora_path is not None:
            if self.is_main_process:
                logger.info(f"Loading pretrained LoRA from {self.config.pretrained_lora_path}")
            self.load_pretrained_lora(self.config.pretrained_lora_path)
        else:
            from peft import LoraConfig
            
            if self.config.lora_target_modules is None:
                target_modules = "all-linear"
            else:
                target_modules = self.config.lora_target_modules
            
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            
            self.transformer.add_adapter(lora_config, adapter_name="default")

        
        if self.is_main_process:
            trainable_params = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.transformer.parameters())
            logger.info(f"LoRA applied successfully. Trainable parameters: {trainable_params:,} / {total_params:,} "
                       f"({100 * trainable_params / total_params:.2f}%)")
    
    def _apply_fsdp(self):
        if self.is_main_process:
            logger.info("Applying FSDP2 to transformer/ref_model...")
        
        master_weight_type = str(getattr(self.config, "master_weight_type", "fp32")).lower()
        if master_weight_type not in ("fp32", "bf16"):
            raise ValueError(f"Unsupported master_weight_type: {master_weight_type}. Must be 'fp32' or 'bf16'.")
        param_dtype = torch.float32 if master_weight_type == "fp32" else torch.bfloat16
        reduce_dtype = torch.float32  # Reduce in float32 for stability

        self.transformer = self.transformer.to(dtype=param_dtype)
        ref_on_gpu = self.ref_model is not None and next(self.ref_model.parameters(), None) is not None and next(self.ref_model.parameters()).device.type == "cuda"
        if self.ref_model is not None and ref_on_gpu:
            self.ref_model = self.ref_model.to(dtype=param_dtype)
        
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
        )
        
        fsdp_config = {"mp_policy": mp_policy}
        if self.world_size > 1:
            try:
                fsdp_config["mesh"] = get_parallel_state().fsdp_mesh
            except Exception as e:
                if self.is_main_process:
                    logger.warning(f"Could not create DeviceMesh: {e}. FSDP will use process group instead.")
        
        for block in list(self.transformer.double_blocks) + list(self.transformer.single_blocks):
            if block is not None:
                fully_shard(block, **fsdp_config)
        
        fully_shard(self.transformer, **fsdp_config)
        
        if self.ref_model is not None and ref_on_gpu:
            for block in list(self.ref_model.double_blocks) + list(self.ref_model.single_blocks):
                if block is not None:
                    fully_shard(block, **fsdp_config)
            fully_shard(self.ref_model, **fsdp_config)
        
        if self.is_main_process:
            logger.info("FSDP2 applied successfully")
    
    def _apply_gradient_checkpointing(self):
        if self.is_main_process:
            logger.info("Applying gradient checkpointing to transformer blocks...")
        
        no_split_module_type = None
        for block in self.transformer.double_blocks:
            if block is not None:
                no_split_module_type = type(block)
                break
        
        if no_split_module_type is None:
            for block in self.transformer.single_blocks:
                if block is not None:
                    no_split_module_type = type(block)
                    break
        
        if no_split_module_type is None:
            logger.warning("Could not find block type for gradient checkpointing. Using fallback.")
            if hasattr(self.transformer, "gradient_checkpointing_enable"):
                self.transformer.gradient_checkpointing_enable()
            return
        
        def non_reentrant_wrapper(module):
            return checkpoint_wrapper(
                module,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )
        
        def selective_checkpointing(submodule):
            return isinstance(submodule, no_split_module_type)
        
        apply_activation_checkpointing(
            self.transformer,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=selective_checkpointing,
        )
        
        if self.is_main_process:
            logger.info("Gradient checkpointing applied successfully")
    
    def _build_optimizer(self):
        if self.config.use_muon:
            self.optimizer = get_muon_optimizer(
                model=self.transformer,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            trainable_params = list(self.transformer.parameters())
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=self.config.weight_decay,
            )
        
        self.lr_scheduler = get_scheduler(
            "constant",
            optimizer=self.optimizer,
            num_warmup_steps=self.config.warmup_steps * self.world_size,
            num_training_steps=self.config.max_steps * self.world_size,
        )
        
        if self.is_main_process:
            logger.info(f"Optimizer and scheduler initialized")

    def _log_precision_diagnostics_once(self):
        if getattr(self, "_logged_precision_diagnostics", False):
            return
        if getattr(self.scalar_states, "update_steps", 0) < 1:
            return

        # Parameter storage dtype after FSDP/mixed-precision wrapping.
        param_dtypes = sorted({str(p.dtype) for p in self.transformer.parameters() if p.requires_grad})
        logger.info(f"[PrecisionDiag] trainable param dtypes: {param_dtypes}")

        if self.config.enable_fsdp and self.world_size > 1:
            logger.info(
                "[PrecisionDiag] Using composable FSDP2 mixed precision (param_dtype=bf16, reduce_dtype=float32). "
                "This path does not maintain an explicit FP32 master-weight copy for parameters."
            )

        # Optimizer-state dtype check after first optimizer step.
        state_dtypes = set()
        tensor_state_count = 0
        for state in self.optimizer.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    state_dtypes.add(str(value.dtype))
                    tensor_state_count += 1
        logger.info(f"[PrecisionDiag] optimizer tensor-state dtypes: {sorted(state_dtypes)} (count={tensor_state_count})")
        if tensor_state_count > 0 and "torch.float32" not in state_dtypes:
            logger.warning(
                "[PrecisionDiag] Optimizer has no FP32 tensor states. If you expect FP32 master/state behavior, "
                "consider switching optimizer strategy or explicitly promoting optimizer states to FP32."
            )

        self._logged_precision_diagnostics = True
    
    def encode_text(self, prompts, data_type: str = "image"):
        text_inputs = self.text_encoder.text2tokens(prompts, data_type=data_type)
        text_outputs = self.text_encoder.encode(text_inputs, data_type=data_type, device=self.device)
        text_emb = text_outputs.hidden_state
        text_mask = text_outputs.attention_mask
        
        text_emb_2 = None
        text_mask_2 = None
        if self.text_encoder_2 is not None:
            text_inputs_2 = self.text_encoder_2.text2tokens(prompts)
            text_outputs_2 = self.text_encoder_2.encode(text_inputs_2, device=self.device)
            text_emb_2 = text_outputs_2.hidden_state
            text_mask_2 = text_outputs_2.attention_mask
        
        return text_emb, text_mask, text_emb_2, text_mask_2
    
    def encode_byt5(self, text_ids: torch.Tensor, attention_mask: torch.Tensor):
        if self.byt5_kwargs["byt5_model"] is None:
            return None, None
        byt5_outputs = self.byt5_kwargs["byt5_model"](text_ids, attention_mask=attention_mask.float())
        byt5_emb = byt5_outputs[0]
        return byt5_emb, attention_mask
    
    def encode_images(self, images):
        """Encode images to vision states (for i2v)"""
        if self.vision_encoder is None:
            return None
        assert images.max() <= 1.0 and images.min() >= -1.0, f"Images must be in the range [-1, 1], but got {images.min()} {images.max()}"
        images = (images + 1) / 2 # [-1, 1] -> [0, 1]
        images_np = (images.cpu().permute(0, 2, 3, 1).numpy() * 255).clip(0, 255).astype("uint8")
        vision_states = self.vision_encoder.encode_images(images_np)
        return vision_states.last_hidden_state.to(device=self.device, dtype=self.transformer.dtype)
    
    def encode_vae(self, images: torch.Tensor) -> torch.Tensor:
        if images.max() > 1.0 or images.min() < -1.0:
            raise ValueError(f"Images must be in the range [-1, 1], but got {images.min()} {images.max()}")
        
        if images.ndim == 4:
            images = images.unsqueeze(2)
        
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16), self.vae.memory_efficient_context():
            latents = self.vae.encode(images).latent_dist.sample()
            if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor:
                latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
            else:
                latents = latents * self.vae.config.scaling_factor
        
        return latents
    
    def get_condition(self, latents: torch.Tensor, task_type: str) -> torch.Tensor:
        b, c, f, h, w = latents.shape
        cond = torch.zeros([b, c + 1, f, h, w], device=latents.device, dtype=latents.dtype)
        
        if task_type == "t2v":
            return cond
        elif task_type == "i2v":
            cond[:, :-1, :1] = latents[:, :, :1]
            cond[:, -1, 0] = 1
            return cond
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def sample_task(self, data_type: str) -> str:
        """
        Sample task type based on data type and configuration.
        
        For video data: samples between t2v and i2v based on i2v_prob
        For image data: always returns t2v (image-to-video generation)
        """
        if data_type == "image":
            return "t2v"
        elif data_type == "video":
            if random.random() < self.config.i2v_prob:
                return "i2v"
            else:
                return "t2v"
        else:
            return "t2v"
    
    def save_checkpoint(self, step: int):
        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{step}")
        transformer_dir = os.path.join(checkpoint_dir, "transformer")
        
        if self.is_main_process:
            os.makedirs(checkpoint_dir, exist_ok=True)
        if self.world_size > 1:
            dist.barrier()
        
        if self.config.use_lora and hasattr(self.transformer, "save_lora_adapter"):
            lora_dir = os.path.join(checkpoint_dir, "lora")
            os.makedirs(lora_dir, exist_ok=True)
            
            if hasattr(self.transformer, "peft_config") and self.transformer.peft_config:
                adapter_names = list(self.transformer.peft_config.keys())
                if self.is_main_process:
                    logger.info(f"Saving {len(adapter_names)} LoRA adapter(s): {adapter_names}")
                
                for adapter_name in adapter_names:
                    adapter_dir = os.path.join(lora_dir, adapter_name)
                    os.makedirs(adapter_dir, exist_ok=True)
                    self.transformer.save_lora_adapter(
                        save_directory=adapter_dir,
                        adapter_name=adapter_name,
                        safe_serialization=True,
                    )
                    if self.is_main_process:
                        logger.info(f"LoRA adapter '{adapter_name}' saved to {adapter_dir}")
            else:
                raise RuntimeError("No LoRA adapter found in the model")
            
            if self.world_size > 1:
                dist.barrier()
        
        # Save full model state dict
        model_state_dict = get_model_state_dict(self.transformer)
        dcp.save(
            state_dict={"model": model_state_dict},
            checkpoint_id=transformer_dir,
        )

        optimizer_state_dict = get_optimizer_state_dict(
            self.transformer,
            self.optimizer,
        )
        optimizer_dir = os.path.join(checkpoint_dir, "optimizer")
        dcp.save(
            state_dict={"optimizer": optimizer_state_dict},
            checkpoint_id=optimizer_dir,
        )
        
        if self.is_main_process:
            training_state_path = os.path.join(checkpoint_dir, "training_state.pt")
            torch.save({
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "global_step": step,
            }, training_state_path)
        
        if self.world_size > 1:
            dist.barrier()
        
        if self.is_main_process:
            logger.info(f"Checkpoint saved at step {step} to {checkpoint_dir}")

    def load_pretrained_lora(self, lora_dir: str):
        self.transformer.load_lora_adapter(
            pretrained_model_name_or_path_or_dict=lora_dir,
            prefix=None,
            adapter_name="default",
            use_safetensors=True,
            hotswap=False,
        )
    
    def load_checkpoint(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
        
        if self.is_main_process:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        if self.world_size > 1:
            dist.barrier()
        
        
        transformer_dir = os.path.join(checkpoint_path, "transformer")
        if os.path.exists(transformer_dir):
            model_state_dict = get_model_state_dict(self.transformer)
            dcp.load(
                state_dict={"model": model_state_dict},
                checkpoint_id=transformer_dir,
            )
            if self.is_main_process:
                logger.info("Transformer model state loaded")
        else:
            logger.warning(f"Transformer dcp checkpoint not found from {checkpoint_path}")

        optimizer_dir = os.path.join(checkpoint_path, "optimizer")
        if os.path.exists(optimizer_dir):
            optimizer_state_dict = get_optimizer_state_dict(
                self.transformer,
                self.optimizer,
            )
            dcp.load(
                state_dict={"optimizer": optimizer_state_dict},
                checkpoint_id=optimizer_dir,
            )
            if self.is_main_process:
                logger.info("Optimizer state loaded")
        
        training_state_path = os.path.join(checkpoint_path, "training_state.pt")
        if os.path.exists(training_state_path):
            if self.is_main_process:
                training_state = torch.load(training_state_path, map_location=self.device)
                self.lr_scheduler.load_state_dict(training_state["lr_scheduler"])
                self.global_step = training_state.get("global_step", 0)
                logger.info(f"Training state loaded: global_step={self.global_step}")
            else:
                # Non-main processes will get global_step via broadcast
                self.global_step = 0
        
        if self.world_size > 1:
            global_step_tensor = torch.tensor(self.global_step, device=self.device)
            dist.broadcast(global_step_tensor, src=0)
            self.global_step = global_step_tensor.item()
        
        if self.world_size > 1:
            dist.barrier()
        
        if self.is_main_process:
            logger.info(f"Checkpoint loaded successfully. Resuming from step {self.global_step}")
    
    def _build_video_dataset(self, train_video_csv: Optional[Union[str, List[str]]] = None):
        csv_paths = train_video_csv if train_video_csv is not None else self.config.train_video_csv
        if not csv_paths:
            csv_paths = TrainingConfig.__dataclass_fields__["train_video_csv"].default
        if isinstance(csv_paths, str):
            csv_paths = [csv_paths]
        elif isinstance(csv_paths, tuple):
            csv_paths = list(csv_paths)

        if not isinstance(csv_paths, list) or len(csv_paths) == 0:
            raise ValueError(
                f"Invalid train_video_csv: {csv_paths}. "
                "Please provide a csv path or list of csv paths."
            )

        # Keep sampler constraints aligned with the original complex codebase:
        # RepeatRandomDistributedSampler requires (video_batch_size * dp_size) % num_generations == 0.
        # Here video_batch_size is per-dp-rank batch size.
        video_batch_size = int(self.config.batch_size)
        num_generations = int(self.config.num_generations)
        dp_replicas = int(self.dp_size)
        if (video_batch_size * dp_replicas) % max(num_generations, 1) != 0:
            raise ValueError(
                "Invalid GRPO sampler config: (batch_size * dp_replicas) must be divisible by num_generations. "
                f"Got batch_size={video_batch_size}, dp_replicas={dp_replicas}, "
                f"num_generations={num_generations}, product={video_batch_size * dp_replicas}. "
                "Typical fixes: increase batch_size, reduce num_generations, or reduce sp_size to increase dp_replicas."
            )
        if self.is_main_process:
            if video_batch_size >= num_generations:
                ranks_per_group = 1
                num_groups_per_rank = video_batch_size // max(num_generations, 1)
            else:
                ranks_per_group = num_generations // max(video_batch_size, 1)
                num_groups_per_rank = 1
            logger.info(
                f"[Effective GRPO Grouping] video_batch_size={video_batch_size}, dp_replicas={dp_replicas}, "
                f"num_generations={num_generations}, ranks_per_group={ranks_per_group}, "
                f"num_groups_per_rank={num_groups_per_rank}, product={video_batch_size * dp_replicas}"
            )

        # Minimal args namespace compatible with get_post_train_video_dataloader/VideoPromptDataset.
        dataset_args = {
            "post_train_type": "grpo",
            "train_video_csv": csv_paths,
            "video_micro_batch_size": [self.config.batch_size],
            "num_workers": self.config.num_workers,
            "prefetch_factor": self.config.prefetch_factor,
            "global_seed": self.config.seed,
            "num_generations": self.config.num_generations,
            "text_len": 256,
            "byt5_max_length": 256,
            "glyph_byT5_v2": False,
            "video_uncond_p_byt5": 0.0,
        }
        args_ns = argparse.Namespace(**dataset_args)
        return get_post_train_video_dataloader(
            args_ns,
            logger,
            self.text_encoder,
            self.text_encoder_2,
            self.dp_size,
            self.dp_rank,
            self.byt5_kwargs,
            local_seed=self.config.seed + self.dp_rank,
        )

    def _fmt_metric(self, v: float) -> str:
        """Format metric value for logging."""
        av = abs(v)
        if av == 0.0:
            return "0.00000000"
        if av < 1e-6:
            return f"{v:.3e}"
        return f"{v:.8f}"

    def _log_progress_step(self, metrics: Dict[str, Any]) -> None:
        """Log training progress for current step (main process only)."""
        if self.global_step % self.config.log_interval != 0 or not self.is_main_process:
            return
        loss_val = float(metrics.get("loss", metrics.get("total_loss", 0.0)))
        grad_norm_val = float(metrics.get("grad_norm", 0.0))
        lr_val = float(metrics.get("lr", self.lr_scheduler.get_last_lr()[0] if hasattr(self.lr_scheduler, "get_last_lr") else self.config.learning_rate))
        progress_info = {
            "step": f"{self.global_step + 1}/{self.config.max_steps}",
            "learning_rate": self._fmt_metric(lr_val),
            "loss": self._fmt_metric(loss_val),
            "grad_norm": self._fmt_metric(grad_norm_val),
        }
        for key, value in metrics.items():
            if key in ("loss", "total_loss", "grad_norm", "lr"):
                continue
            if isinstance(value, (int, float)):
                progress_info[key] = self._fmt_metric(float(value))
        logger.info(
            f"Training Progress: update_steps {self.global_step + 1}\t"
            f"train_steps {getattr(self.scalar_states, 'train_steps', self.global_step + 1)}\t"
            f"{progress_info}"
        )

    def _run_post_step_checks(self, next_step: int) -> None:
        """Run validation and checkpoint saving at step boundaries."""
        if self.config.validation_interval > 0 and next_step % self.config.validation_interval == 0:
            self.validate(next_step)
        if next_step % self.config.save_interval == 0:
            self.save_checkpoint(next_step)
            if self.world_size > 1:
                dist.barrier()

    def _train_grpo(self, dataloader=None):
        if self.is_main_process:
            logger.info("Starting GRPO training...")
            logger.info(f"Max steps: {self.config.max_steps}")
            logger.info(f"Batch size: {self.config.batch_size}")
            logger.info(f"Learning rate: {self.config.learning_rate}")

        if self.config.resume_from_checkpoint is not None:
            self.load_checkpoint(self.config.resume_from_checkpoint)

        if dataloader is None:
            _, _, dataloader = self._build_video_dataset(self.config.train_video_csv)

        self.transformer.train()
        if self.config.validate_at_step0 and self.global_step == 0:
            self.validate(0)
            if self.world_size > 1:
                dist.barrier()
        # if self.is_main_process and int(getattr(self.config, "num_generations", 1)) <= 1:
        #     logger.warning(
        #         "[GRPO Config Warning] num_generations=1 means each group has only one sample; "
        #         "reward json will contain one entry per step and normalized advantages collapse. "
        #         "Set --num_generations > 1 (and align batch/parallel config) for standard GRPO grouping."
        #     )

        while self.global_step < self.config.max_steps:
            for batch in dataloader:
                if self.global_step >= self.config.max_steps:
                    break

                if self.is_main_process and not getattr(self, "_logged_new_train_flag", False):
                    logger.info("[FLAG] post_train train path -> train_one_step")
                    self._logged_new_train_flag = True

                self.grpo_args.kl_weight = float(self.config.kl_weight)
                self.grpo_args.gradient_accumulation_steps = (
                    int(self.config.gradient_accumulation_steps)
                    * int(self.config.grpo_sampling_steps)
                    * int(max(self.config.batch_size, 1))
                )
                self.grpo_args.max_grad_norm = float(self.config.max_grad_norm)
                self.grpo_args.output_dir = self.config.output_dir

                if self.ref_model is not None:
                    self.ref_model = maybe_update_ref_model_for_kl(
                        self.grpo_args,
                        self.transformer,
                        self.ref_model,
                        self.scalar_states,
                        logger,
                    )
                metrics = grpo_train_one_step(
                    self.grpo_args,
                    self.transformer,
                    self.ref_model,
                    self.vae,
                    self.text_encoder,
                    self.text_encoder_2,
                    self.byt5_kwargs,
                    {"vision_encoder": self.vision_encoder} if self.vision_encoder is not None else {},
                    self.reward_inferencer,
                    batch,
                    self.device,
                    self.dp_rank,
                    self.world_size,
                    logger,
                    self.scalar_states,
                    self.optimizer,
                    self.lr_scheduler,
                    mask_type=None,
                    timesteps_train=None,
                )
                if self.is_main_process:
                    self._log_precision_diagnostics_once()

                self._log_progress_step(metrics)
                if self.is_main_process:
                    self._log_metrics_to_tb(metrics, self.global_step + 1)

                self._run_post_step_checks(self.global_step + 1)
                self.global_step += 1

        if self.is_main_process:
            self.save_checkpoint(self.global_step)
            logger.info("Training completed!")
            if self.tb_writer is not None:
                self.tb_writer.flush()
                self.tb_writer.close()

        if self.world_size > 1:
            dist.barrier()
            dist.destroy_process_group()

    def _load_validation_items(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        csv_paths_raw = getattr(self.config, "valid_video_csv", None)
        csv_paths: List[str] = []
        if isinstance(csv_paths_raw, str):
            csv_paths = [csv_paths_raw]
        elif isinstance(csv_paths_raw, list):
            csv_paths = [str(p) for p in csv_paths_raw if p is not None and str(p).strip() != ""]

        for csv_path in csv_paths:
            if not os.path.exists(csv_path):
                logger.warning(f"valid_video_csv not found, skip: {csv_path}")
                continue
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    prompt = str(row.get("prompt", "")).strip()
                    if not prompt:
                        continue
                    seed_raw = row.get("seed", None)
                    seed_val = None
                    if seed_raw is not None and str(seed_raw).strip() != "":
                        try:
                            seed_val = int(seed_raw)
                        except Exception:
                            seed_val = None
                    items.append(
                        {
                            "prompt": prompt,
                            "seed": seed_val,
                            "sample_idx": i,
                        }
                    )

        if items:
            return items

        prompts = self.config.validation_prompts or []
        for i, p in enumerate(prompts):
            items.append({"prompt": str(p), "seed": None, "sample_idx": i})
        return items

    @staticmethod
    def _sanitize_prompt_for_filename(prompt: str, max_len: int = 50) -> str:
        p = prompt.replace("/", " ").replace("\\", " ").replace("\n", " ").replace("\t", " ").strip()
        return p[:max_len] if p else "empty_prompt"

    @staticmethod
    def _parse_fixed_size(fixed_size: Optional[str]):
        if fixed_size is None:
            return None, None
        text = str(fixed_size).strip().lower()
        if text in ("", "none"):
            return None, None
        sep = "x" if "x" in text else ("*" if "*" in text else None)
        if sep is None:
            raise ValueError(f"Invalid validation_fixed_size format: {fixed_size}. Expected like 480x864")
        h_str, w_str = [s.strip() for s in text.split(sep, 1)]
        if not h_str.isdigit() or not w_str.isdigit():
            raise ValueError(f"Invalid validation_fixed_size format: {fixed_size}. Expected like 480x864")
        return int(h_str), int(w_str)

    def validate(self, step: int):
        validation_items = self._load_validation_items()
        if not validation_items:
            return
        validation_video_length = int(getattr(self.config, "validate_video_length", 81))
        validation_global_seed = int(getattr(self.config, "validation_global_seed", 930))
        validation_fixed_size = getattr(self.config, "validation_fixed_size", "480x864")
        fixed_height, fixed_width = self._parse_fixed_size(validation_fixed_size)
        
        self.transformer.eval()
        if self.is_main_process:
            logger.info(f"Running validation at step {step}...")
            logger.info(f"Validation samples: {len(validation_items)}")
            logger.info(f"Validation video_length={validation_video_length}")
            logger.info(f"Validation fixed global_seed={validation_global_seed}")
            logger.info(f"Validation fixed_size={validation_fixed_size}")

        for idx, item in enumerate(validation_items):
            prompt = item["prompt"]
            sample_seed = int(item["seed"]) if item.get("seed", None) is not None else validation_global_seed
            generator = torch.Generator(device=self.pipeline.noise_init_device).manual_seed(sample_seed)
            if self.is_main_process:
                logger.info(
                    f"Generating validation video {idx + 1}/{len(validation_items)}: {prompt[:50]}..."
                )
            with torch.no_grad():
                output = self.pipeline(
                    enable_sr=self.config.validation_enable_sr,
                    prompt=prompt,
                    aspect_ratio=self.config.validation_aspect_ratio,
                    eta=self.config.eta,
                    num_inference_steps=self.config.validation_num_inference_steps,
                    video_length=validation_video_length,
                    negative_prompt=self.config.validation_negative_prompt,
                    seed=sample_seed,
                    generator=generator,
                    height=fixed_height,
                    width=fixed_width,
                    output_type="pt",
                    prompt_rewrite=self.config.validation_prompt_rewrite,
                )

            step_dir = os.path.join(self.validation_output_dir, f"step_{step:06d}")
            if self.is_main_process:
                os.makedirs(step_dir, exist_ok=True)
            prompt_name = self._sanitize_prompt_for_filename(prompt, max_len=50)
            video_path = os.path.join(
                step_dir,
                f"sample_{idx:03d}_seed_{sample_seed}_{prompt_name}.mp4",
            )
            video_to_save = output.sr_videos if self.config.validation_enable_sr and hasattr(output, "sr_videos") else output.videos
            if self.is_main_process:
                save_video(video_to_save, video_path)
                logger.info(f"Validation video saved to {video_path}")

        self.transformer.train()


def create_dummy_dataloader(config: TrainingConfig):
    """
    Create a dummy dataloader for testing.
    
    Note: This is a placeholder - users should implement their own dataset and dataloader
    that loads actual video/image data.
    
    Required fields for Dataset __getitem__:
    - "pixel_values": torch.Tensor
        * For video: shape [C, F, H, W] where F is the number of frames
        * For image: shape [C, H, W]
        * Pixel values must be in range [-1, 1]
        * Data type: torch.float32
        * Note: For video data, temporal dimension F must be 4n+1 (e.g., 1, 5, 9, 13, 17, 21, ...)
          to satisfy VAE requirements. The dataset should ensure this before returning data.
    
    - "text": str
        * Text prompt for this sample
    
    - "data_type": str
        * "video" for video data (supports both t2v and i2v tasks based on i2v_prob)
        * "image" for image data (always uses t2v task)
    
    Optional fields (for performance optimization):
    - "latents": torch.Tensor, shape [C_latent, F, H_latent, W_latent]
        * Pre-encoded VAE latents. If provided, pixel_values will be ignored and VAE encoding
          will be skipped, significantly speeding up training.
        * Should be in the same format as VAE encoder output (after scaling_factor applied)
        * Temporal dimension F must still be 4n+1 for video data
    
    Optional fields (for byT5 text encoding):
    - "byt5_text_ids": Optional[torch.Tensor], shape [seq_len]
        * Pre-tokenized byT5 token IDs. If provided, will be used directly.
        * If not provided, text will be tokenized on-the-fly.
    
    - "byt5_text_mask": Optional[torch.Tensor], shape [seq_len]
        * Attention mask for byT5 tokens (1 for valid tokens, 0 for padding)
        * Required if byt5_text_ids is provided
    
    Task type selection (automatic based on data_type and config.i2v_prob):
    - For "video" data: randomly samples between t2v (text-to-video) and i2v (image-to-video)
      based on config.i2v_prob probability
    - For "image" data: always uses t2v task
    
    Example sample format (what dataset __getitem__ should return):
    {
        "pixel_values": torch.Tensor([3, 81, 480, 848]),  # Video example
        "text": "A cat playing",
        "data_type": "video",
        "byt5_text_ids": torch.Tensor([256]),  # Optional
        "byt5_text_mask": torch.Tensor([256]),  # Optional
    }
    
    Or with pre-encoded latents (faster):
    {
        "latents": torch.Tensor([32, 31, 30, 53]),  # Pre-encoded VAE latents
        "text": "A cat playing",
        "data_type": "video",
    }
    """
    # This is a placeholder - users should implement their own dataloader
    class DummyDataset:
        def __init__(self, size=100):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            # Video: temporal dimension must be 4n+1, using 17 frames
            # Generate data in range [-1, 1]

            resolution = (121, 480, 848)
            latent_resolution = [(resolution[0] - 1) // 4 + 1, resolution[1] // 16, resolution[2] // 16]

            data = torch.rand(3, *resolution) * 2.0 - 1.0  # [0, 1] -> [-1, 1]
            data_type = "video"

            return {
                "pixel_values": data,
                "text": "A sample prompt",
                "data_type": data_type,
                "latents": torch.randn(32, *latent_resolution),
                # "byt5_text_ids": torch.zeros((256), dtype=torch.int64),
                # "byt5_text_mask": torch.zeros((256), dtype=torch.int64),
            }
    
    dataset = DummyDataset()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    return dataloader


def main():
    parser = argparse.ArgumentParser(description="Train HunyuanVideo-1.5 on video data")
    
    # Model paths
    parser.add_argument("--pretrained_model_root", type=str, default='ckpts', help="Path to pretrained model")
    parser.add_argument("--pretrained_transformer_version", type=str, default="480p_t2v", help="Transformer version")
    parser.add_argument("--post_train_type", type=str, default="grpo", choices=["grpo", "standard"], help="Post-train mode")
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_steps", type=int, default=10000, help="Maximum training steps")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--train_timestep_shift", type=float, default=3.0, help="Train Timestep shift")
    parser.add_argument("--flow_snr_type", type=str, default="lognorm", 
                        choices=["uniform", "lognorm", "mix", "mode"],
                        help="SNR type for flow matching: uniform, lognorm, mix, or mode (default: lognorm)")
    
    # Data parameters
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="Dataloader prefetch factor")
    parser.add_argument("--num_generations", type=int, default=4, help="GRPO repeat generations per prompt")
    parser.add_argument("--train_video_csv", type=str, nargs="+", default=None, help="CSV path(s) with prompt/index/seed for post-train")
    parser.add_argument("--reward_model", type=str, default="videoalign_local", help="Reward model name")
    parser.add_argument(
        "--reward_weights",
        type=str,
        default=TrainingConfig.__dataclass_fields__["reward_weights"].default,
        help='JSON string for sub_reward weights, e.g. \'{"VQ":0.5,"MQ":0.5,"TA":1.0}\'',
    )
    parser.add_argument("--reward_checkpoint_mode", type=str, default="v3", help="Reward checkpoint mode")
    parser.add_argument("--remote_reward_url", type=str, default=None, help="Remote reward server url")
    parser.add_argument("--kl_weight", type=float, default=1e-5, help="KL regularization weight")
    parser.add_argument("--kl_coef", type=float, default=1e-7, help="Initial KL coefficient")
    parser.add_argument("--kl_min_coef", type=float, default=1e-7, help="Lower bound for adaptive KL coefficient")
    parser.add_argument("--use_moving_KL", type=str_to_bool, nargs='?', const=True, default=True,
                        help="Enable moving KL ref-model update")
    parser.add_argument("--update_ref_model_step", type=int, default=10,
                        help="Moving KL ref-model update interval (optimizer update steps)")
    parser.add_argument("--use_dual_kl", type=str_to_bool, nargs='?', const=True, default=True,
                        help="Enable dual KL (moving/fixed + step-wise KL)")
    parser.add_argument("--dual_kl_moving_weight", type=float, default=1.0,
                        help="Weight for moving/fixed KL term in dual KL")
    parser.add_argument("--dual_kl_step_weight", type=float, default=0.1,
                        help="Weight for step-wise KL term in dual KL")
    parser.add_argument(
        "--sde_type",
        type=str,
        default="sage_grpo",
        choices=["dance_grpo", "sage_grpo", "flow_grpo", "cps"],
        help="SDE type used by GRPO scheduler/log-prob step",
    )
    parser.add_argument("--use_grad_balancing", type=str_to_bool, nargs='?', const=True, default=True,
                        help="Enable gradient balancing for GRPO training")
    parser.add_argument("--enable_timestep_permutation", type=str_to_bool, nargs='?', const=True, default=True,
                        help="Enable timestep permutation for GRPO training")
    parser.add_argument("--debug_grad_flow", type=str_to_bool, nargs='?', const=True, default=False,
                        help="Enable gradient/parameter update debug diagnostics in GRPO training")
    parser.add_argument("--debug_train_diagnostics_interval", type=int, default=1,
                        help="Log debug diagnostics every N optimizer updates when debug_grad_flow is enabled")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--save_interval", type=int, default=100, help="Checkpoint save interval")
    parser.add_argument("--log_interval", type=int, default=1, help="Logging interval")
    
    # Other parameters
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"], help="Data type")
    parser.add_argument("--master_weight_type", type=str, default="fp32", choices=["fp32", "bf16"],
                        help="FSDP parameter(master) dtype: fp32 or bf16 (default: fp32)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--validation_global_seed",
        type=int,
        default=930,
        help="Fixed global seed used only for validate() generation",
    )
    parser.add_argument("--i2v_prob", type=float, default=0.3, help="Probability of i2v task for video data (default: 0.3)")
    parser.add_argument("--use_muon", type=str_to_bool, nargs='?', const=True, default=True,
        help="Use Muon optimizer for training (default: true). "
             "Use --use_muon or --use_muon true/1 to enable, --use_muon false/0 to disable"
    )
    # FSDP and gradient checkpointing
    parser.add_argument(
        "--enable_fsdp", type=str_to_bool, nargs='?', const=True, default=True,
        help="Enable FSDP for distributed training (default: true). "
             "Use --enable_fsdp or --enable_fsdp true/1 to enable, --enable_fsdp false/0 to disable"
    )
    parser.add_argument(
        "--enable_gradient_checkpointing", type=str_to_bool, nargs='?', const=True, default=True,
        help="Enable gradient checkpointing (default: true). "
             "Use --enable_gradient_checkpointing or --enable_gradient_checkpointing true/1 to enable, "
             "--enable_gradient_checkpointing false/0 to disable"
    )
    parser.add_argument(
        "--sp_size", type=int, default=8,
        help="Sequence parallelism size (default: 1). Must evenly divide world_size. "
             "For example, if world_size=8, valid sp_size values are 1, 2, 4, 8."
    )
    parser.add_argument(
        "--dp_replicate", type=int, default=1,
        help="Data parallelism replicate size (default: 1). "
    )
    
    # Validation parameters
    parser.add_argument("--validation_interval", type=int, default=100, help="Run validation every N steps (default: 100)")
    parser.add_argument("--validate_at_step0", type=str_to_bool, nargs='?', const=True, default=False, help="Run validation at step 0 before training loop")
    parser.add_argument("--validation_prompts", type=str, nargs="+", default=None, 
                        help="Prompts for validation (default: single default prompt). Can specify multiple prompts.")
    parser.add_argument("--valid_video_csv", type=str, nargs="+", default=None, help="Validation CSV path(s), takes priority over validation_prompts")
    parser.add_argument("--validation_timestep_shift", type=float, default=5.0, help="Validation Timestep shift")
    parser.add_argument(
        "--validation_fixed_size",
        type=str,
        default=TrainingConfig.__dataclass_fields__["validation_fixed_size"].default,
        help="Fixed validation resolution as HxW (e.g., 480x864). Use 'none' to disable.",
    )
    parser.add_argument(
        "--validate_video_length",
        "--validation_video_length",
        "--video_length",
        dest="validate_video_length",
        type=int,
        default=TrainingConfig.__dataclass_fields__["validate_video_length"].default,
        help="Video length (number of frames) for validation",
    )

    parser.add_argument("--reference_mode_offload", type=str_to_bool, nargs='?', const=True, default=False,
                        help="Enable reference mode offload (default: false)")
    # Resume training parameters
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint directory to resume training from (e.g., ./outputs/checkpoint-1000)")
    
    # LoRA parameters
    parser.add_argument("--use_lora", type=str_to_bool, nargs='?', const=True, default=False,
                        help="Enable LoRA training (default: false). "
                             "Use --use_lora or --use_lora true/1 to enable, --use_lora false/0 to disable")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank (default: 8)")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha scaling parameter (default: 16)")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                        help="LoRA dropout rate (default: 0.0)")
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=None,
                        help="Target modules for LoRA (default: all Linear layers). "
                             "Example: --lora_target_modules img_attn_q img_attn_v img_mlp.fc1")
    parser.add_argument("--pretrained_lora_path", type=str, default=None,
                        help="Path to pretrained LoRA adapter to load. If provided, will load this adapter instead of creating a new one.")
    
    args = parser.parse_args()
    project_root = os.path.dirname(os.path.abspath(__file__))
    run_root = os.path.join(project_root, "SAGE-GRPO-logs", "SAGE_GRPO_results")
    run_time = datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
    run_output_dir = os.path.join(run_root, run_time)

    config = TrainingConfig(
        pretrained_model_root=args.pretrained_model_root,
        pretrained_transformer_version=args.pretrained_transformer_version,
        post_train_type=args.post_train_type,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        num_generations=args.num_generations,
        output_dir=run_output_dir,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        dtype=args.dtype,
        master_weight_type=args.master_weight_type,
        seed=args.seed,
        validation_global_seed=args.validation_global_seed,
        i2v_prob=args.i2v_prob,
        enable_fsdp=args.enable_fsdp,
        enable_gradient_checkpointing=args.enable_gradient_checkpointing,
        sp_size=args.sp_size,
        use_muon=args.use_muon,
        dp_replicate=args.dp_replicate,
        validation_interval=args.validation_interval,
        validate_at_step0=args.validate_at_step0,
        validation_prompts=args.validation_prompts,
        valid_video_csv=(
            args.valid_video_csv
            if args.valid_video_csv is not None
            else TrainingConfig.__dataclass_fields__["valid_video_csv"].default
        ),
        train_timestep_shift=args.train_timestep_shift,
        validation_timestep_shift=args.validation_timestep_shift,
        validation_fixed_size=args.validation_fixed_size,
        snr_type=SNRType(args.flow_snr_type),
        validate_video_length=args.validate_video_length,
        resume_from_checkpoint=args.resume_from_checkpoint,
        reference_mode_offload=args.reference_mode_offload,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        pretrained_lora_path=args.pretrained_lora_path,
        train_video_csv=(
            args.train_video_csv
            if args.train_video_csv is not None
            else TrainingConfig.__dataclass_fields__["train_video_csv"].default
        ),
        reward_model=args.reward_model,
        reward_weights=args.reward_weights,
        reward_checkpoint_mode=args.reward_checkpoint_mode,
        remote_reward_url=args.remote_reward_url,
        kl_weight=args.kl_weight,
        kl_coef=args.kl_coef,
        kl_min_coef=args.kl_min_coef,
        use_moving_KL=args.use_moving_KL,
        update_ref_model_step=args.update_ref_model_step,
        use_dual_kl=args.use_dual_kl,
        dual_kl_moving_weight=args.dual_kl_moving_weight,
        dual_kl_step_weight=args.dual_kl_step_weight,
        sde_type=args.sde_type,
        use_grad_balancing=args.use_grad_balancing,
        enable_timestep_permutation=args.enable_timestep_permutation,
        debug_grad_flow=args.debug_grad_flow,
        debug_train_diagnostics_interval=args.debug_train_diagnostics_interval,
    )
    
    trainer = HunyuanVideoTrainer(config)
    if config.post_train_type == "grpo":
        _, _, dataloader = trainer._build_video_dataset(config.train_video_csv)
    else:
        dataloader = create_dummy_dataloader(config)
    trainer._train_grpo(dataloader)


if __name__ == "__main__":
    main()

