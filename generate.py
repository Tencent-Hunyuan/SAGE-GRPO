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

import os

if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import copy
import csv
import datetime
import json
import re
from typing import List, Dict, Any

import loguru
import torch
import argparse
import einops
import imageio
from torch import distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_model_state_dict

from hyvideo.pipelines.hunyuan_video_pipeline import HunyuanVideo_1_5_Pipeline
from hyvideo.commons.parallel_states import initialize_parallel_state
from hyvideo.commons.infer_state import initialize_infer_state

parallel_dims = initialize_parallel_state(sp=int(os.environ.get('WORLD_SIZE', '1')))
torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))

def save_video(video, path):
    if video.ndim == 5:
        assert video.shape[0] == 1
        video = video[0]
    vid = (video * 255).clamp(0, 255).to(torch.uint8)
    vid = einops.rearrange(vid, 'c f h w -> f h w c')
    imageio.mimwrite(path, vid, fps=24)

def rank0_log(message, level):
    if int(os.environ.get('RANK', '0')) == 0:
        loguru.logger.log(level, message)

def save_config(args, output_path, task, transformer_version, sample_overrides=None):
    """Save generation config. sample_overrides: dict of per-sample values to override in arguments (e.g. prompt, seed)."""
    arguments = {}
    for key, value in vars(args).items():
        if not key.startswith('_') and not callable(value):
            try:
                json.dumps(value)
                arguments[key] = value
            except (TypeError, ValueError):
                arguments[key] = str(value)
    if sample_overrides:
        for k, v in sample_overrides.items():
            arguments[k] = v
    arguments['output_path'] = output_path

    config = {
        'timestamp': datetime.datetime.now().isoformat(),
        'task': task,
        'transformer_version': transformer_version,
        'output_path': output_path,
        'arguments': arguments
    }
    
    base_path, _ = os.path.splitext(output_path)
    config_path = f"{base_path}_config.json"
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"Saved generation config to: {config_path}")
    return config_path

def str_to_bool(value):
    """Convert string to boolean, supporting true/false, 1/0, yes/no.
    If value is None (when flag is provided without value), returns True."""
    if value is None:
        return True  # When --flag is provided without value, enable it
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.lower().strip()
        if value in ('true', '1', 'yes', 'on'):
            return True
        elif value in ('false', '0', 'no', 'off'):
            return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value}")


def _sanitize_prompt_for_filename(prompt: str, max_len: int = 50) -> str:
    p = str(prompt).replace("/", " ").replace("\\", " ").replace("\n", " ").replace("\t", " ").strip()
    p = re.sub(r"\s+", " ", p)
    p = re.sub(r"[^a-zA-Z0-9 _.-]", "", p)
    return (p[:max_len].strip() or "empty_prompt")


def _load_valid_items_from_csv(csv_paths: List[str]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for csv_path in csv_paths:
        if not csv_path:
            continue
        if not os.path.exists(csv_path):
            rank0_log(f"valid_video_csv not found, skip: {csv_path}", "WARNING")
            continue
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                prompt = str(row.get("prompt", "")).strip()
                if not prompt:
                    continue
                seed_val = None
                seed_raw = row.get("seed", None)
                if seed_raw is not None and str(seed_raw).strip() != "":
                    try:
                        seed_val = int(seed_raw)
                    except Exception:
                        seed_val = None
                items.append({"prompt": prompt, "seed": seed_val, "sample_idx": idx})
    return items


def _parse_fixed_size(fixed_size: str):
    if fixed_size is None:
        return None, None
    text = str(fixed_size).strip().lower()
    if text in ("", "none"):
        return None, None
    sep = "x" if "x" in text else ("*" if "*" in text else None)
    if sep is None:
        raise ValueError(f"Invalid --fixed_size format: {fixed_size}. Expected like 480x864")
    h_str, w_str = [s.strip() for s in text.split(sep, 1)]
    if not h_str.isdigit() or not w_str.isdigit():
        raise ValueError(f"Invalid --fixed_size format: {fixed_size}. Expected like 480x864")
    return int(h_str), int(w_str)


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip())


def _build_original_latents_index(original_noise_dir: str):
    if not original_noise_dir:
        return []
    if not os.path.isdir(original_noise_dir):
        raise ValueError(f"--original_noise_dir not found: {original_noise_dir}")

    entries = []
    prefix = "original_latents_"
    for fname in os.listdir(original_noise_dir):
        if not fname.endswith(".pt") or not fname.startswith(prefix):
            continue
        prompt_key = fname[len(prefix):-3]
        entries.append(
            {
                "path": os.path.join(original_noise_dir, fname),
                "prompt_key_raw": prompt_key,
                "prompt_key_norm": _normalize_text(prompt_key),
                "prompt_key_safe": _sanitize_prompt_for_filename(prompt_key),
            }
        )
    return entries


def _find_original_latents_path(prompt_text: str, entries: List[Dict[str, Any]]):
    if not entries:
        return None

    p_norm = _normalize_text(prompt_text)
    p_safe = _sanitize_prompt_for_filename(prompt_text)

    # 1) exact normalized prompt match
    for e in entries:
        if e["prompt_key_norm"] == p_norm:
            return e["path"]

    # 2) truncated filename key prefix match (common for long prompts)
    prefix_matches = [e for e in entries if p_norm.startswith(e["prompt_key_norm"])]
    if prefix_matches:
        prefix_matches.sort(key=lambda x: len(x["prompt_key_norm"]), reverse=True)
        return prefix_matches[0]["path"]

    # 3) safe-filename key match
    for e in entries:
        if e["prompt_key_safe"] == p_safe:
            return e["path"]

    return None

def load_checkpoint_to_transformer(pipe, checkpoint_path):
    
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
    
    rank0_log(f"Loading checkpoint from {checkpoint_path}", "INFO")
    
    try:
        model_state_dict = get_model_state_dict(pipe.transformer)
        dcp.load(
            state_dict={"model": model_state_dict},
            checkpoint_id=checkpoint_path,
        )
        rank0_log("Transformer model state loaded successfully", "INFO")
    except Exception as e:
        rank0_log(f"Error loading checkpoint: {e}", "ERROR")
        raise

def load_lora_adapter(pipe, lora_path):
    rank0_log(f"Loading LoRA adapter from {lora_path}", "INFO")
    try:
        pipe.transformer.load_lora_adapter(
            pretrained_model_name_or_path_or_dict=lora_path,
            prefix=None,
            adapter_name="default",
            use_safetensors=True,
            hotswap=False,
        )
        rank0_log("LoRA adapter loaded successfully", "INFO")
    except Exception as e:
        rank0_log(f"Error loading LoRA adapter: {e}", "ERROR")
        raise


def _validate_generate_args(args):
    """Validate mutually exclusive and dependency args before generation."""
    if args.sparse_attn and args.use_sageattn:
        raise ValueError("sparse_attn and use_sageattn cannot be enabled simultaneously. Please enable only one of them.")
    if args.use_fp8_gemm and 'sgl' in args.quant_type:
        try:
            import sgl_kernel
        except Exception:
            raise ValueError("sgl_kernel is not installed. Please install it using `pip install sgl-kernel==0.3.18`")
    if args.enable_step_distill and args.enable_cache:
        raise ValueError("Enabling both step distilled model and cache will lead to performance degradation.")


def _get_transformer_dtype(dtype_str: str):
    """Map dtype string to torch dtype."""
    _DTYPE_MAP = {"bf16": torch.bfloat16, "fp32": torch.float32}
    if dtype_str not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Must be 'bf16' or 'fp32'")
    return _DTYPE_MAP[dtype_str]


def _setup_device_and_offloading(args):
    """Resolve device, offloading, and group offloading settings."""
    enable_offloading = args.offloading
    if args.group_offloading is None:
        offloading_config = HunyuanVideo_1_5_Pipeline.get_offloading_config()
        enable_group_offloading = offloading_config['enable_group_offloading']
    else:
        enable_group_offloading = args.group_offloading
    device = torch.device('cpu') if enable_offloading else torch.device('cuda')
    transformer_init_device = torch.device('cpu') if enable_group_offloading else device
    return device, transformer_init_device, enable_offloading, enable_group_offloading, args.overlap_group_offloading


def _resolve_generation_items(args):
    """Load or fallback to prompt-based generation items."""
    csv_paths = args.valid_video_csv
    csv_paths = [csv_paths] if isinstance(csv_paths, str) else (csv_paths or [])
    items = _load_valid_items_from_csv(csv_paths)
    if items:
        rank0_log(f"Using valid_video_csv with {len(items)} prompts.", "INFO")
        return items
    if args.prompt:
        rank0_log("valid_video_csv is empty/unavailable, fallback to --prompt.", "WARNING")
        return [{"prompt": args.prompt, "seed": args.seed, "sample_idx": 0}]
    raise ValueError("No valid prompts found. Provide --valid_video_csv or --prompt.")


def _resolve_output_paths(output_path, generation_items, transformer_version, args):
    """Resolve output_path and build target_paths for each sample."""
    if output_path is None:
        now = f'{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}'
        output_path = f'./outputs/output_{transformer_version}_{now}.mp4'
    if len(generation_items) == 1:
        return output_path, [output_path]
    base_path, ext = os.path.splitext(output_path)
    ext = ext if ext else ".mp4"
    os.makedirs(base_path, exist_ok=True)
    target_paths = []
    for i, item in enumerate(generation_items):
        safe_prompt = _sanitize_prompt_for_filename(item["prompt"])
        sample_seed = int(item["seed"]) if item.get("seed", None) is not None else int(args.seed)
        target_paths.append(os.path.join(base_path, f"sample_{i:03d}_seed_{sample_seed}_{safe_prompt}{ext}"))
    return output_path, target_paths


def _prepare_call_kwargs(args, item, extra_kwargs, original_latents_entries):
    """Build call_kwargs for pipe(), including optional original latents."""
    call_kwargs = dict(extra_kwargs)
    if not args.original_noise_dir:
        return call_kwargs
    prompt_text = item["prompt"]
    latents_path = _find_original_latents_path(prompt_text, original_latents_entries)
    if latents_path is None:
        raise ValueError(
            f"Original latent not found for prompt: {prompt_text[:120]}... "
            f"in dir: {args.original_noise_dir}"
        )
    latents = torch.load(latents_path, map_location="cpu")
    if latents.ndim == 4:
        latents = latents.unsqueeze(0)
    call_kwargs["latents"] = latents
    rank0_log(f"[OriginalNoise] Loaded latent: {latents_path}", "INFO")
    return call_kwargs


def _save_generation_output(args, out, sample_output_path, enable_sr):
    """Save generated video(s) to disk (rank0 only)."""
    if int(os.environ.get('RANK', '0')) != 0:
        return
    sample_output_dir = os.path.dirname(sample_output_path)
    if sample_output_dir:
        os.makedirs(sample_output_dir, exist_ok=True)
    if enable_sr and hasattr(out, 'sr_videos'):
        save_video(out.sr_videos, sample_output_path)
        print(f"Saved SR video to: {sample_output_path}")
        if args.save_pre_sr_video:
            base_p, ext = os.path.splitext(sample_output_path)
            original_path = f"{base_p}_before_sr{ext}"
            save_video(out.videos, original_path)
            print(f"Saved original video (before SR) to: {original_path}")
    else:
        save_video(out.videos, sample_output_path)
        print(f"Saved video to: {sample_output_path}")


def generate_video(args):
    infer_state = initialize_infer_state(args)
    _validate_generate_args(args)

    task = 'i2v' if args.image_path else 't2v'
    enable_sr = args.sr
    transformer_version = HunyuanVideo_1_5_Pipeline.get_transformer_version(
        args.resolution, task, args.cfg_distilled, args.enable_step_distill, args.sparse_attn
    )
    transformer_dtype = _get_transformer_dtype(args.dtype)

    device, transformer_init_device, enable_offloading, enable_group_offloading, overlap_group_offloading = _setup_device_and_offloading(args)

    pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=args.model_path,
        transformer_version=transformer_version,
        create_sr_pipeline=enable_sr,
        transformer_dtype=transformer_dtype,
        device=device,
        transformer_init_device=transformer_init_device,
    )

    noise_init_device = torch.device('cuda') if device.type == 'cuda' else torch.device('cpu')
    pipe.noise_init_device = noise_init_device
    rank0_log(f"Noise init device: {noise_init_device}", "INFO")
    
    loguru.logger.info(f"{enable_offloading=} {enable_group_offloading=} {overlap_group_offloading=}")

    pipe.apply_infer_optimization(
        infer_state=infer_state,
        enable_offloading=enable_offloading,
        enable_group_offloading=enable_group_offloading,
        overlap_group_offloading=overlap_group_offloading,
    )
    if args.checkpoint_path:
        load_checkpoint_to_transformer(pipe, args.checkpoint_path)
    if args.lora_path:
        load_lora_adapter(pipe, args.lora_path)

    if enable_sr and hasattr(pipe, 'sr_pipeline'):
        sr_infer_state = copy.deepcopy(infer_state)
        sr_infer_state.enable_cache = False
        pipe.sr_pipeline.apply_infer_optimization(
            infer_state=sr_infer_state,
            enable_offloading=enable_offloading,
            enable_group_offloading=enable_group_offloading,
            overlap_group_offloading=overlap_group_offloading,
        )

    extra_kwargs = {'reference_image': args.image_path} if task == 'i2v' else {}
    if args.video_length != 121:
        rank0_log(f"Warning: 121 frames is the optimal value for best quality. "
                  f"Attempting to generate {args.video_length} frames...", "WARNING")
    if not args.rewrite:
        rank0_log("Warning: Prompt rewriting is disabled. This may affect the quality of generated videos.", "WARNING")

    fixed_height, fixed_width = _parse_fixed_size(args.fixed_size)
    if fixed_height is not None and fixed_width is not None:
        rank0_log(f"Using fixed resolution override: {fixed_height}x{fixed_width}", "INFO")

    generation_items = _resolve_generation_items(args)

    original_latents_entries = _build_original_latents_index(args.original_noise_dir) if args.original_noise_dir else []
    if args.original_noise_dir:
        rank0_log(f"Original noise mode enabled: {args.original_noise_dir}", "INFO")
        rank0_log(f"Indexed original noise files: {len(original_latents_entries)}", "INFO")
        if args.output_path is None:
            noise_base = os.path.basename(os.path.normpath(args.original_noise_dir))
            args.output_path = f"./outputs/{noise_base}_replay.mp4"
            rank0_log(f"Auto-set output_path for replay mode: {args.output_path}", "INFO")

    output_path, target_paths = _resolve_output_paths(
        args.output_path, generation_items, transformer_version, args
    )
    enable_rewrite = args.rewrite

    for i, item in enumerate(generation_items):
        prompt_text = item["prompt"]
        sample_seed = int(item["seed"]) if item.get("seed", None) is not None else int(args.seed)
        generator = torch.Generator(device=pipe.noise_init_device).manual_seed(sample_seed)
        call_kwargs = _prepare_call_kwargs(args, item, extra_kwargs, original_latents_entries)

        rank0_log(
            f"[Generate] sample={i+1}/{len(generation_items)}, seed={sample_seed}, prompt={prompt_text[:80]}",
            "INFO",
        )
        out = pipe(
            enable_sr=enable_sr,
            prompt=prompt_text,
            aspect_ratio=args.aspect_ratio,
            num_inference_steps=args.num_inference_steps,
            sr_num_inference_steps=None,
            video_length=args.video_length,
            negative_prompt=args.negative_prompt,
            generator=generator,
            seed=sample_seed,
            height=fixed_height,
            width=fixed_width,
            output_type="pt",
            prompt_rewrite=enable_rewrite,
            return_pre_sr_video=args.save_pre_sr_video,
            **call_kwargs,
        )

        _save_generation_output(args, out, target_paths[i], enable_sr)
        if args.save_generation_config and int(os.environ.get('RANK', '0')) == 0:
            try:
                save_config(args, target_paths[i], task, transformer_version, sample_overrides={'prompt': prompt_text, 'seed': sample_seed})
            except Exception:
                pass

def main():
    parser = argparse.ArgumentParser(description='Generate video using HunyuanVideo-1.5')

    parser.add_argument(
        '--prompt', type=str, default=None,
        help='Text prompt for video generation (used when valid_video_csv is empty)'
    )
    parser.add_argument(
        '--valid_video_csv', type=str, nargs="+", default=["assets/ssae_0728_en.csv"],
        help='Validation CSV path(s), priority over --prompt'
    )
    parser.add_argument(
        '--negative_prompt', type=str, default='',
        help='Negative prompt for video generation (default: empty string)'
    )
    parser.add_argument(
        '--resolution', type=str, required=True, choices=['480p', '720p'],
        help='Video resolution (480p or 720p)'
    )
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='Path to pretrained model'
    )
    parser.add_argument(
        '--aspect_ratio', type=str, default='16:9',
        help='Aspect ratio (default: 16:9)'
    )
    parser.add_argument(
        '--fixed_size', type=str, default=None,
        help='Fixed output size as HxW (e.g., 480x864). Overrides aspect_ratio mapping.'
    )
    parser.add_argument(
        '--num_inference_steps', type=int, default=None,
        help='Number of inference steps (default: 50)'
    )
    parser.add_argument(
        '--video_length', type=int, default=121,
        help='Number of frames to generate (default: 121)'
    )
    parser.add_argument(
        '--sr', type=str_to_bool, nargs='?', const=True, default=True,
        help='Enable super resolution (default: true). '
             'Use --sr or --sr true/1 to enable, --sr false/0 to disable'
    )
    parser.add_argument(
        '--save_pre_sr_video', type=str_to_bool, nargs='?', const=True, default=False,
        help='Save original video before super resolution (default: false). '
             'Use --save_pre_sr_video or --save_pre_sr_video true/1 to enable, '
             '--save_pre_sr_video false/0 to disable'
    )
    parser.add_argument(
        '--rewrite', type=str_to_bool, nargs='?', const=True, default=False,
        help='Enable prompt rewriting (default: true). '
             'Use --rewrite or --rewrite true/1 to enable, --rewrite false/0 to disable'
    )
    parser.add_argument(
        '--cfg_distilled', type=str_to_bool, nargs='?', const=True, default=False,
        help='Enable CFG distilled model (default: false). '
             'Use --cfg_distilled or --cfg_distilled true/1 to enable, '
             '--cfg_distilled false/0 to disable'
    )
    parser.add_argument(
        '--enable_step_distill', type=str_to_bool, nargs='?', const=True, default=False,
        help='Enable step distilled model (default: false). '
             'Use --enable_step_distill or --enable_step_distill true/1 to enable, '
             '--enable_step_distill false/0 to disable'
    )
    parser.add_argument(
        '--sparse_attn', type=str_to_bool, nargs='?', const=True, default=False,
        help='Enable sparse attention (default: false). '
             'Use --sparse_attn or --sparse_attn true/1 to enable, '
             '--sparse_attn false/0 to disable'
    )
    parser.add_argument(
        '--offloading', type=str_to_bool, nargs='?', const=True, default=True,
        help='Enable offloading (default: true). '
             'Use --offloading or --offloading true/1 to enable, '
             '--offloading false/0 to disable'
    )
    parser.add_argument(
        '--group_offloading', type=str_to_bool, nargs='?', const=True, default=None,
        help='Enable group offloading (default: None, automatically enabled if offloading is enabled). '
             'Use --group_offloading or --group_offloading true/1 to enable, '
             '--group_offloading false/0 to disable'
    )
    parser.add_argument(
        '--overlap_group_offloading', type=str_to_bool, nargs='?', const=True, default=True,
        help='Enable overlap group offloading (default: true). '
             'Significantly increases CPU memory usage but speeds up inference. '
             'Use --overlap_group_offloading or --overlap_group_offloading true/1 to enable, '
             '--overlap_group_offloading false/0 to disable'
    )
    parser.add_argument(
        '--dtype', type=str, default='bf16', choices=['bf16', 'fp32'],
        help='Data type for transformer (default: bf16). '
             'bf16: faster, lower memory; fp32: better quality, slower, higher memory'
    )
    parser.add_argument(
        '--seed', type=int, default=930,
        help='Global random seed for generator (default: 930, aligned with post_train)'
    )
    parser.add_argument(
        '--image_path', type=str, default=None,
        help='Path to reference image for i2v (if provided, uses i2v mode)'
    )
    parser.add_argument(
        '--output_path', type=str, default=None,
        help='Output file path for generated video (if not provided, saves to ./outputs/output.mp4)'
    )
    parser.add_argument(
        '--original_noise_dir', type=str, default=None,
        help='Directory containing pre-saved original latents (.pt). '
             'Files should follow original_latents_{prompt}.pt naming.'
    )
    parser.add_argument(
        '--use_sageattn', type=str_to_bool, nargs='?', const=True, default=False,
        help='Enable sageattn (default: false). '
             'Use --use_sageattn or --use_sageattn true/1 to enable, '
             '--use_sageattn false/0 to disable'
    )
    parser.add_argument(
        '--sage_blocks_range', type=str, default="0-53",
        help='Sageattn blocks range (e.g., 0-5 or 0,1,2,3,4,5)'
    )
    parser.add_argument(
        '--enable_torch_compile', type=str_to_bool, nargs='?', const=True, default=False,
        help='Enable torch compile for transformer (default: false). '
             'Use --enable_torch_compile or --enable_torch_compile true/1 to enable, '
             '--enable_torch_compile false/0 to disable'
    )
    parser.add_argument(
        '--enable_cache', type=str_to_bool, nargs='?', const=True, default=False,
        help='Enable cache for transformer (default: false). '
             'Use --enable_cache or --enable_cache true/1 to enable, '
             '--enable_cache false/0 to disable'
    )
    parser.add_argument(
        '--cache_type', type=str, default="deepcache",
        help='Cache type for transformer (e.g., deepcache, teacache, taylorcache)'
    )
    parser.add_argument(
        '--no_cache_block_id', type=str, default="53",
        help='Blocks to exclude from deepcache (e.g., 0-5 or 0,1,2,3,4,5)'
    )
    parser.add_argument(
        '--cache_start_step', type=int, default=11,
        help='Start step to skip when using cache (default: 11)'
    )
    parser.add_argument(
        '--cache_end_step', type=int, default=45,
        help='End step to skip when using cache (default: 45)'
    )
    parser.add_argument(
        '--total_steps', type=int, default=50,
        help='Total inference steps (default: 50)'
    )
    parser.add_argument(
        '--cache_step_interval', type=int, default=4,
        help='Step interval to skip when using cache (default: 4)'
    )
    parser.add_argument(
        '--save_generation_config', type=str_to_bool, nargs='?', const=True, default=True,
        help='Save generation config file (default: true). '
             'Use --save_generation_config or --save_generation_config true/1 to enable, '
             '--save_generation_config false/0 to disable'
    )
    parser.add_argument(
        '--checkpoint_path', type=str, default=None,
        help='Path to checkpoint directory containing transformer weights (e.g., ./outputs/checkpoint-1000/transformer). '
             'The checkpoint directory should contain a "transformer" subdirectory. '
             'If provided, the transformer model weights will be loaded from this checkpoint.'
    )
    parser.add_argument(
        '--lora_path', type=str, default=None,
        help='Path to LoRA adapter directory or checkpoint directory containing LoRA adapter. '
             'If provided, the LoRA adapter will be loaded to the transformer model.'
    )

    # fp8 gemm related
    parser.add_argument(
        '--use_fp8_gemm', type=str_to_bool, nargs='?', const=True, default=False,
        help='Enable fp8 gemm for transformer (default: false). '
             'Use --use_fp8_gemm or --use_fp8_gemm true/1 to enable, '
             '--use_fp8_gemm false/0 to disable'
    )
    parser.add_argument(
        '--quant_type', type=str, default="fp8-per-token-sgl",
        help='Quantization type for fp8 gemm (e.g., fp8-per-tensor-weight-only, fp8-per-tensor, fp8-per-token-sgl)'
    )
    parser.add_argument(
        '--include_patterns', type=str, default="double_blocks",
        help='Include patterns for fp8 gemm (default: double_blocks)'
    )

    args = parser.parse_args()
    
    # Convert string "none" to None for image_path
    if args.image_path is not None and args.image_path.lower().strip() == 'none':
        args.image_path = None
    
    
    generate_video(args)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
