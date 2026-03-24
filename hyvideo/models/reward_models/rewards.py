"""
Unified reward interface for video generation models.
Provides a standard interface for all reward models including video and image rewards.
All reward functions are self-contained without external flow_grpo dependencies.
"""
import json
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Union, Callable

# VideoAlign model path: default to local path
_DEFAULT_VIDEOALIGN_PATH = "./ckpts/VideoReward"
REWARD_MODEL_PATH = {
    "videoalign": _DEFAULT_VIDEOALIGN_PATH,
}


# ============================================================================
# Video Reward Models
# ============================================================================

def videoalign_local_score(device, reward_checkpoint_mode="none"):
    """Local VideoAlign reward model."""
    from hyvideo.models.reward_models.videoalign.inference import VideoVLMRewardInference
    
    dtype = torch.bfloat16
    scorer = VideoVLMRewardInference(
        REWARD_MODEL_PATH["videoalign"],
        device=device,
        dtype=dtype,
        reward_checkpoint_mode=reward_checkpoint_mode,
    )
    
    def _fn(video_paths: List[str], prompts: List[str], metadata: List[Dict] = None):
        """
        Args:
            video_paths: List of video file paths
            prompts: List of text prompts
            metadata: Optional metadata for each sample
            
        Returns:
            scores_dict: Dict of {metric_name: List[float]}
            meta_dict: Additional metadata (empty for now)
        """
        if metadata is None:
            metadata = [{}] * len(video_paths)
        
        scores_dict = {}
        all_entries = []
        
        # Batch inference
        for video_path, prompt in zip(video_paths, prompts):
            try:
                reward_entry = scorer.reward([video_path], [prompt])[0]
                all_entries.append(reward_entry)
            except Exception as e:
                # Return NaN on error
                all_entries.append({})
        
        # Collect all metrics
        if all_entries:
            # Get all possible keys from entries
            all_keys = set()
            for entry in all_entries:
                all_keys.update(entry.keys())
            
            # Extract each metric
            for key in all_keys:
                scores = []
                for entry in all_entries:
                    val = entry.get(key)
                    try:
                        scores.append(float(val) if val is not None else float('nan'))
                    except (TypeError, ValueError):
                        scores.append(float('nan'))
                scores_dict[key] = scores
        
        return scores_dict, {}
    
    return _fn


def video_align_remote_score(server_url: str, model: str = "video_align"):
    """
    Implementing your own remote video reward service here if required.
    Default using local video reward service.
    """
    
    pass




# ============================================================================
# Utility Functions
# ============================================================================

def _normalize_key(key: str) -> str:
    """Normalize key to lowercase for case-insensitive lookup."""
    return str(key).strip().lower()


_VIDEO_MODELS = frozenset(["videoalign_local", "video_align", "video_score2"])


def _init_reward_metadata(video_paths_or_images, metadata, is_image_input: bool) -> List[Dict]:
    """Initialize metadata list for reward computation."""
    if metadata is not None:
        return metadata
    batch_size = video_paths_or_images.shape[0] if is_image_input else len(video_paths_or_images)
    return [{}] * batch_size


def _should_skip_model(model_name: str, is_image_input: bool) -> bool:
    """Return True if model should be skipped for given input type."""
    is_video_model = model_name in _VIDEO_MODELS
    if is_video_model and is_image_input:
        print(f"Warning: Skipping video model '{model_name}' for image input")
        return True
    if not is_video_model and not is_image_input:
        print(f"Warning: Skipping image model '{model_name}' for video path input")
        return True
    return False


def _store_model_scores(model_name: str, scores, all_scores: Dict, model_raw_scores: Dict) -> None:
    """Store scores from a model into all_scores and model_raw_scores."""
    if isinstance(scores, dict):
        model_raw_scores[model_name] = scores
        for key, values in scores.items():
            prefixed_key = f"{model_name}_{_normalize_key(key)}"
            all_scores[prefixed_key] = values
    elif isinstance(scores, (list, np.ndarray)):
        scores_list = list(scores) if isinstance(scores, np.ndarray) else scores
        model_raw_scores[model_name] = scores_list
        all_scores[model_name] = scores_list
    else:
        raise ValueError(f"Unexpected score format from {model_name}: {type(scores)}")


def _compute_video_model_weighted_scores(raw_scores: Dict, per_model_metric_weights, model_weight: float) -> List[float]:
    """Compute weighted scores for a video model with multiple metrics."""
    if per_model_metric_weights:
        model_metric_scores = []
        for metric_name, weight in per_model_metric_weights.items():
            metric_key = _normalize_key(metric_name)
            metric_values = raw_scores.get(metric_key, raw_scores.get(metric_name))
            if metric_values is not None:
                model_metric_scores.append([weight * s for s in metric_values])
        if model_metric_scores:
            model_sum = [sum(col) for col in zip(*model_metric_scores)]
            return [model_weight * s for s in model_sum]
    else:
        all_metric_values = list(raw_scores.values())
        if all_metric_values:
            model_avg = [sum(col) / len(all_metric_values) for col in zip(*all_metric_values)]
            return [model_weight * s for s in model_avg]
    return None


def _compute_weighted_avg(
    weighted_scores_list: List[List[float]],
    output_scores: Dict,
    video_paths_or_images,
    is_image_input: bool,
) -> None:
    """Compute final weighted average and set output_scores['avg']."""
    if weighted_scores_list:
        output_scores['avg'] = [sum(col) for col in zip(*weighted_scores_list)]
        return
    all_values = [v for k, v in output_scores.items() if k != 'avg']
    if all_values:
        output_scores['avg'] = [sum(col) / len(all_values) for col in zip(*all_values)]
        return
    batch_size = video_paths_or_images.shape[0] if is_image_input else len(video_paths_or_images)
    output_scores['avg'] = [0.0] * batch_size


# ============================================================================
# Multi-Reward Interface
# ============================================================================

def multi_video_score(device, reward_config: Dict[str, Any]):
    """
    Unified multi-reward interface with flexible weighting.
    Only the **standard format** (Example 3) is supported; all other old formats are no longer supported.
    
    Args:
        device: Device to run models on
        reward_config: Dict with configuration
        
            Example:
            {
                "models": {
                    "videoalign_local": {
                        "weight": 1.0,
                        "sub_reward": {"VQ": 0.5, "MQ": 0.5, "TA": 1.0},
                    },
                    "video_align": {"weight": 1.0, "server_url": "..."},
                },
            }
            
            Constraints:
            - Must contain a "models" field.
            - Each model's config must be a dict and must contain at least the "weight" key.
            - The optional "sub_reward" field should be a dictionary mapping metric_name to weight.
            - If there is only one reward model, its weight must be exactly 1.0.
    
    Returns:
        reward_fn: Function (video_paths, prompts, metadata) -> (scores_dict, meta_dict)
    """
    # Must contain "models"
    if "models" not in reward_config:
        raise ValueError(
            "reward_config must contain a 'models' field with the standard format: "
            '{"models": {"model_name": {"weight": float, "sub_reward": {...}}}}'
        )
    
    model_weights = reward_config["models"]
    
    # Parse each model configuration (only standard format is supported)
    # model_configs: {model_name: {"model_weight": float, "metric_weights": dict or None}}
    model_configs: Dict[str, Dict[str, Any]] = {}
    for model_name, weight_config in model_weights.items():
        if not isinstance(weight_config, dict):
            raise ValueError(
                f"Model config for '{model_name}' must be a dict with at least 'weight' key, "
                f"got {type(weight_config)}"
            )
        
        if "weight" not in weight_config:
            raise ValueError(
                f"Model config for '{model_name}' must contain 'weight', "
                f"optionally with 'sub_reward', got keys: {list(weight_config.keys())}"
            )
        
        model_weight = float(weight_config["weight"])
        metric_weights = weight_config.get("sub_reward", None)
        
        if metric_weights is not None and not isinstance(metric_weights, dict):
            raise ValueError(
                f"'sub_reward' for model '{model_name}' must be a dict of metric_name -> weight, "
                f"got {type(metric_weights)}"
            )
        
        model_configs[model_name] = {
            "model_weight": model_weight,
            "metric_weights": metric_weights,
        }
    
    # When only one reward model is used, its weight must be 1.0
    if len(model_configs) == 1:
        only_model_name = next(iter(model_configs.keys()))
        only_model_weight = model_configs[only_model_name]["model_weight"]
        if abs(only_model_weight - 1.0) > 1e-6:
            raise ValueError(
                f"When only one reward model is used ('{only_model_name}'), "
                f"its weight must be 1.0, got {only_model_weight}"
            )
    
    # Initialize all reward functions
    reward_fns = {}
    for model_name in model_configs.keys():
        if model_name == "videoalign_local":
            checkpoint_mode = reward_config.get("reward_checkpoint_mode", "none")
            reward_fns[model_name] = videoalign_local_score(device, checkpoint_mode)
        elif model_name in ["video_align", "video_score2"]:
            server_url = reward_config.get("server_url", reward_config.get("remote_reward_url"))
            if server_url is None:
                raise ValueError(f"server_url required for {model_name}")
            reward_fns[model_name] = video_align_remote_score(server_url, model=model_name)
        else:
            raise ValueError(
                f"Unknown reward model: {model_name}. "
                "Only videoalign_local and video_align/video_score2 are supported."
            )
    
    def _fn(video_paths_or_images: Union[List[str], torch.Tensor], prompts: List[str], metadata: List[Dict] = None):
        """
        Compute rewards for videos/images.
        
        Args:
            video_paths_or_images: Either a list of video file paths (for video models)
                                   or a torch.Tensor of images (for image models)
            prompts: List of text prompts
            metadata: Optional metadata for each sample
        
        Weighting logic:
        1. Each video model's metrics are first weighted by their metric weights
        2. Then the weighted sum of each model's metrics is multiplied by model weight
        3. Image models are directly multiplied by their model weight
        4. Final avg = sum of all weighted scores
        
        Returns scores_dict with keys like:
        - "videoalign_local_vq", "videoalign_local_mq", "videoalign_local_ta" (video model metrics)
        - "aesthetic" (image model)
        - "tencent_remote_vq", "tencent_remote_mq" (another video model)
        - "avg" (weighted average)
        """
        is_image_input = isinstance(video_paths_or_images, torch.Tensor)
        metadata = _init_reward_metadata(video_paths_or_images, metadata, is_image_input)

        all_scores = {}
        all_meta = {}
        model_raw_scores = {}

        for model_name in model_configs.keys():
            if _should_skip_model(model_name, is_image_input):
                continue
            reward_fn = reward_fns[model_name]
            scores, meta = reward_fn(video_paths_or_images, prompts, metadata)
            _store_model_scores(model_name, scores, all_scores, model_raw_scores)
            all_meta.update(meta)

        output_scores = all_scores.copy()
        weighted_scores_list = []

        for model_name, config in model_configs.items():
            if model_name not in model_raw_scores:
                continue
            model_weight = config["model_weight"]
            per_model_metric_weights = config["metric_weights"]
            raw_scores = model_raw_scores[model_name]

            if isinstance(raw_scores, dict):
                weighted = _compute_video_model_weighted_scores(
                    raw_scores, per_model_metric_weights, model_weight
                )
                if weighted is not None:
                    weighted_scores_list.append(weighted)
            else:
                weighted_scores_list.append([model_weight * s for s in raw_scores])

        _compute_weighted_avg(
            weighted_scores_list, output_scores, video_paths_or_images, is_image_input
        )
        assert 'avg' in output_scores.keys(), f"avg not in output_scores: {output_scores.keys()}"
        return output_scores, all_meta

    return _fn


def get_reward_fn(args, device, logger):
    """
    Factory function to create reward function from args.
    
    Args:
        args: Training arguments
        device: Device
        logger: Logger
        
    Returns:
        reward_fn: Unified reward function
    """
    # External config has highest priority
    if hasattr(args, "reward_config") and isinstance(args.reward_config, dict) and args.reward_config:
        reward_config = args.reward_config.copy()
        # Add reward_checkpoint_mode if not in config
        if hasattr(args, "reward_checkpoint_mode"):
            reward_config.setdefault("reward_checkpoint_mode", args.reward_checkpoint_mode)
        if hasattr(args, "remote_reward_url"):
            reward_config.setdefault("server_url", args.remote_reward_url)
        logger.info(f"Using external reward_config: {reward_config}")
    else:
        # Fallback: Build reward_config from reward_model and optional reward_weights
        logger.warning("reward_config not provided, falling back to default config with reward_model only")
        
        reward_model = getattr(args, "reward_model", "videoalign_local")
        if reward_model == "auto":
            reward_model = "videoalign_local"
        
        # Use reward_weights from args if provided (e.g. from post_train --reward_weights), else default
        rw = getattr(args, "reward_weights", None)
        if isinstance(rw, dict) and rw:
            sub_reward = rw
        elif isinstance(rw, str) and rw.strip():
            try:
                parsed = json.loads(rw.strip())
                sub_reward = parsed if isinstance(parsed, dict) and parsed else {"VQ": 1.0, "MQ": 1.0, "TA": 1.0}
            except (json.JSONDecodeError, TypeError):
                sub_reward = {"VQ": 1.0, "MQ": 1.0, "TA": 1.0}
        else:
            sub_reward = {"VQ": 1.0, "MQ": 1.0, "TA": 1.0}
        
        # Build standard format reward_config
        reward_config = {
            "models": {
                reward_model: {
                    "weight": 1.0,
                    "sub_reward": sub_reward
                }
            }
        }
        
        # Model-specific config
        if reward_model in ["video_align", "video_score2"]:
            reward_config["server_url"] = getattr(args, "remote_reward_url", None)
        elif reward_model == "videoalign_local":
            reward_config["reward_checkpoint_mode"] = getattr(args, "reward_checkpoint_mode", "v2")
            # reward_config["reward_checkpoint_mode"] = "v3"
        
        if hasattr(args, "remote_reward_url"):
            reward_config.setdefault("server_url", args.remote_reward_url)
        if hasattr(args, "reward_checkpoint_mode"):
            reward_config.setdefault("reward_checkpoint_mode", args.reward_checkpoint_mode)
    
    logger.info(f"Creating reward function with config: {reward_config}")
    
    return multi_video_score(device, reward_config)


def create_reward_fn_from_config(config: Dict[str, Any], device, logger=None):
    """
    Create reward function from config dict.
    
    Args:
        config: Reward configuration
        device: Device
        logger: Optional logger
        
    Returns:
        reward_fn: Unified reward function
    """
    if logger:
        logger.info(f"Creating reward function from config: {config}")
    
    return multi_video_score(device, config)
