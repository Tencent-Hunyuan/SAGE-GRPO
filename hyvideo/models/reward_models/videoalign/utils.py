import os
import glob
from dataclasses import dataclass, field
import datasets
from typing import List, Literal, Optional

import safetensors
import torch
import torch.nn as nn
from transformers import TrainingArguments, Qwen2VLForConditionalGeneration, AutoProcessor

from peft import LoraConfig, get_peft_model
from trl import get_kbit_device_map, get_quantization_config

from typing import List, Optional
from transformers.trainer import (
    is_torch_xla_available,
)

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
else:
    IS_XLA_FSDPV2_POST_2_2 = False

########## Model Config ##########

class Qwen2VLRewardModelBT(Qwen2VLForConditionalGeneration):
    def __init__(self, config, output_dim=4, reward_token="last", special_token_ids=None):
        super().__init__(config)
        # pdb.set_trace()
        self.output_dim = output_dim
        self.rm_head = nn.Linear(config.hidden_size, output_dim, bias=False)
        self.reward_token = reward_token

        self.special_token_ids = special_token_ids
        if self.special_token_ids is not None:
            self.reward_token = "special"
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
    ):
        ## modified from the origin class Qwen2VLForConditionalGeneration
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # pdb.set_trace()
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]  # [B, L, D]

        logits = self.rm_head(hidden_states)    # [B, L, N]
        
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        ## get sequence length
        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        ## get the last token's logits
        if self.reward_token == "last":
            pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        elif self.reward_token == "mean":
            ## get the mean of all valid tokens' logits
            valid_lengths = torch.clamp(sequence_lengths, min=0, max=logits.size(1) - 1)
            pooled_logits = torch.stack([logits[i, :valid_lengths[i]].mean(dim=0) for i in range(batch_size)])
        elif self.reward_token == "special":
            # special_token_ids = self.tokenizer.convert_tokens_to_ids(self.special_tokens)
            # create a mask for special tokens
            special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for special_token_id in self.special_token_ids:
                special_token_mask = special_token_mask | (input_ids == special_token_id)
            pooled_logits = logits[special_token_mask, ...]
            pooled_logits = pooled_logits.view(batch_size, 3, -1)   # [B, 3, N] assert 3 attributes
            if self.output_dim == 3:
                pooled_logits = pooled_logits.diagonal(dim1=1, dim2=2)
            pooled_logits = pooled_logits.view(batch_size, -1)

            # pdb.set_trace()
        else:
            raise ValueError("Invalid reward_token")
        
        return {"logits": pooled_logits}


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

########## DataClass For Configure ##########

@dataclass
class TrainingConfig(TrainingArguments):
    max_length: Optional[int] = None
    dataset_num_proc: Optional[int] = None
    center_rewards_coefficient: Optional[float] = None
    disable_flash_attn2: bool = field(default=False)

    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    special_token_lr: Optional[float] = None

    conduct_eval: Optional[bool] = True
    load_from_pretrained: str = None
    load_from_pretrained_step: int = None
    logging_epochs: Optional[float] = None
    eval_epochs: Optional[float] = None
    save_epochs: Optional[float] = None
    remove_unused_columns: Optional[bool] = False

    save_full_model: Optional[bool] = False

@dataclass
class PEFTLoraConfig:
    lora_enable: bool = False
    vision_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None
    lora_namespan_exclude: Optional[List[str]] = None
    lora_modules_to_save: Optional[List[str]] = None
    lora_task_type: str = "CAUSAL_LM"
    use_rslora: bool = False
    num_lora_modules: int = -1

    def __post_init__(self):
        if isinstance(self.lora_target_modules, list) and len(self.lora_target_modules) == 1:
            self.lora_target_modules = self.lora_target_modules[0]

        if isinstance(self.lora_namespan_exclude, list) and len(self.lora_namespan_exclude) == 1:
            self.lora_namespan_exclude = self.lora_namespan_exclude[0]

@dataclass
class ModelConfig:
    model_name_or_path: Optional[str] = None
    model_revision: str = "main"

    output_dim: int = 1

    use_special_tokens: bool = False

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    tune_merger: bool = field(default=False)

    torch_dtype: Optional[Literal["auto", "bfloat16", "float16", "float32"]] = None
    trust_remote_code: bool = False
    attn_implementation: Optional[str] = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    bnb_4bit_quant_type: Literal["fp4", "nf4"] = "nf4"
    use_bnb_nested_quant: bool = False
    reward_token: Literal["last", "mean", "special"] = "last"
    loss_type: Literal["bt", "reg", "btt", "margin", "constant_margin", "scaled"] = "regular"

    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("You can't use 8 bit and 4 bit precision at the same time")

        # if isinstance(self.lora_target_modules, list) and len(self.lora_target_modules) == 1:
        #     self.lora_target_modules = self.lora_target_modules[0]

        # if isinstance(self.lora_namespan_exclude, list) and len(self.lora_namespan_exclude) == 1:
        #     self.lora_namespan_exclude = self.lora_namespan_exclude[0]


########## Load Models From Folder ##########

def _insert_adapter_name_into_state_dict(
    state_dict: dict[str, torch.Tensor], adapter_name: str, parameter_prefix: str
) -> dict[str, torch.Tensor]:
    """Utility function to remap the state_dict keys to fit the PEFT model by inserting the adapter name."""
    peft_model_state_dict = {}
    for key, val in state_dict.items():
        if parameter_prefix in key:
            suffix = key.split(parameter_prefix)[1]
            if "." in suffix:
                suffix_to_replace = ".".join(suffix.split(".")[1:])
                key = key.replace(suffix_to_replace, f"{adapter_name}.{suffix_to_replace}")
            else:
                key = f"{key}.{adapter_name}"
            peft_model_state_dict[key] = val
        else:
            peft_model_state_dict[key] = val
    return peft_model_state_dict


def _fix_checkpoint_key_mapping_v2(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Fix key mapping mismatch between checkpoint and model state dict."""
    fixed_state_dict = {}
    for key, val in state_dict.items():
        # Some checkpoints save the vision encoder weights with an extra "model" module.
        if key.startswith("base_model.model.model.visual."):
            new_key = key.replace("base_model.model.model.visual.", "base_model.model.visual.")
            fixed_state_dict[new_key] = val
        # Some checkpoints save the language model weights under ...model.model.language_model.*
        elif key.startswith("base_model.model.model.language_model."):
            new_key = key.replace("base_model.model.model.language_model.", "base_model.model.model.")
            fixed_state_dict[new_key] = val
        else:
            fixed_state_dict[key] = val
    return fixed_state_dict


def _fix_checkpoint_key_mapping_v3(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Normalize checkpoint keys to match current PEFT Qwen-VL layout."""
    fixed_state_dict = {}
    for key, val in state_dict.items():
        new_key = key

        # Visual tower variants -> base_model.model.model.visual.*
        if new_key.startswith("base_model.model.visual."):
            new_key = new_key.replace("base_model.model.visual.", "base_model.model.model.visual.")
        elif new_key.startswith("base_model.model.vision_tower."):
            new_key = new_key.replace("base_model.model.vision_tower.", "base_model.model.model.visual.")
        elif new_key.startswith("base_model.model.model.vision_tower."):
            new_key = new_key.replace("base_model.model.model.vision_tower.", "base_model.model.model.visual.")

        # Language model variants -> base_model.model.model.language_model.*
        if new_key.startswith("base_model.model.language_model."):
            new_key = new_key.replace("base_model.model.language_model.", "base_model.model.model.language_model.")
        elif new_key.startswith("base_model.model.model.embed_tokens") or new_key.startswith("base_model.model.model.layers") or new_key.startswith("base_model.model.model.norm"):
            new_key = new_key.replace("base_model.model.model.", "base_model.model.model.language_model.")

        fixed_state_dict[new_key] = val

    return fixed_state_dict


def _fix_checkpoint_key_mapping(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Fix key mapping mismatch between checkpoint and model state dict."""
    fixed_state_dict = {}
    for key, val in state_dict.items():
        # Fix visual keys: base_model.model.visual.* -> base_model.model.model.visual.*
        if key.startswith("base_model.model.visual."):
            new_key = key.replace("base_model.model.visual.", "base_model.model.model.visual.")
            fixed_state_dict[new_key] = val
        # Fix language model keys: base_model.model.model.* -> base_model.model.model.language_model.*
        elif key.startswith("base_model.model.model.embed_tokens") or key.startswith("base_model.model.model.layers") or key.startswith("base_model.model.model.norm"):
            new_key = key.replace("base_model.model.model.", "base_model.model.model.language_model.")
            fixed_state_dict[new_key] = val
        else:
            fixed_state_dict[key] = val
    return fixed_state_dict


def save_video(tensor, path):
    from torchvision.io import write_video
    tensor = tensor * 255.0
    tensor = tensor.permute(0, 2, 3, 1)
    tensor = tensor.clamp(0, 255).byte()
    write_video(path, tensor, 4, video_codec='h264')


def load_model_from_checkpoint(
    model, checkpoint_dir, checkpoint_step, reward_checkpoint_mode="v1"
):
    checkpoint_paths = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
    checkpoint_paths.sort(key=lambda x: int(x.split("-")[-1]), reverse=True)

    if checkpoint_step is None or checkpoint_step == -1:
        # get the latest checkpoint
        checkpoint_path = checkpoint_paths[0]
        print(f"===> Checkpoint step is not provided, using the latest checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{checkpoint_step}")
        if checkpoint_path not in checkpoint_paths:
            checkpoint_path = checkpoint_paths[0]
            print(f"===> Checkpoint step {checkpoint_step} not found, using the latest checkpoint: {checkpoint_path}")
        else:
            print(f"===> Checkpoint step {checkpoint_step} found, using the specified checkpoint: {checkpoint_path}")
    
    checkpoint_step = checkpoint_path.split("checkpoint-")[-1].split("/")[0]

    full_ckpt = os.path.join(checkpoint_path, "model.pth")
    lora_ckpt = os.path.join(checkpoint_path, "adapter_model.safetensors")
    non_lora_ckpt = os.path.join(checkpoint_path, "non_lora_state_dict.pth")
    if os.path.exists(full_ckpt):
        print(f"===> Loading full checkpoint from {full_ckpt}")
        model_state_dict = torch.load(full_ckpt, map_location="cpu")
        # Fix key mapping mismatch
        if reward_checkpoint_mode == "v1":
            model_state_dict = _fix_checkpoint_key_mapping(model_state_dict)
        elif reward_checkpoint_mode == "v2":
            model_state_dict = _fix_checkpoint_key_mapping_v2(model_state_dict)
        elif reward_checkpoint_mode == "v3":
            model_state_dict = _fix_checkpoint_key_mapping_v3(model_state_dict)
        else:
            raise ValueError(f"Unsupported reward checkpoint mode: {reward_checkpoint_mode}")
        model.load_state_dict(model_state_dict, strict=True)
    else:
        lora_state_dict = safetensors.torch.load_file(lora_ckpt)
        non_lora_state_dict = torch.load(non_lora_ckpt, map_location="cpu")

        # Fix key mapping mismatch for both state dicts
        if reward_checkpoint_mode == "v1":
            lora_state_dict = _fix_checkpoint_key_mapping(lora_state_dict)
            non_lora_state_dict = _fix_checkpoint_key_mapping(non_lora_state_dict)
        elif reward_checkpoint_mode == "v2":
            lora_state_dict = _fix_checkpoint_key_mapping_v2(lora_state_dict)
            non_lora_state_dict = _fix_checkpoint_key_mapping_v2(non_lora_state_dict)
        elif reward_checkpoint_mode == "v3":
            lora_state_dict = _fix_checkpoint_key_mapping_v3(lora_state_dict)
            non_lora_state_dict = _fix_checkpoint_key_mapping_v3(non_lora_state_dict)
        else:
            raise ValueError(f"Unsupported reward checkpoint mode: {reward_checkpoint_mode}")

        lora_state_dict = _insert_adapter_name_into_state_dict(lora_state_dict, adapter_name="default", parameter_prefix="lora_")
        
        model_state_dict = model.state_dict()
        model_state_dict.update(non_lora_state_dict)
        model_state_dict.update(lora_state_dict)
        model.load_state_dict(model_state_dict)

    return model, checkpoint_step


def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=False):
    """
    Find the target linear modules for LoRA.
    """
    linear_cls = torch.nn.Linear
    embedding_cls = torch.nn.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            # print(f"Excluding module: {name}")
            continue

        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)

    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names


def create_model_and_processor(
        model_config, peft_lora_config, training_args,
        cache_dir=None,
    ):
    # create model
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        use_cache=True if training_args.gradient_checkpointing else False,
    )
    # pdb.set_trace()

    # create processor and set padding
    processor = AutoProcessor.from_pretrained(model_config.model_name_or_path,
                                              padding_side="right",
                                              cache_dir=cache_dir)
    
    special_token_ids = None
    if model_config.use_special_tokens:
        special_tokens = ["<|VQ_reward|>", "<|MQ_reward|>", "<|TA_reward|>"]
        processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        special_token_ids = processor.tokenizer.convert_tokens_to_ids(special_tokens)

    model = Qwen2VLRewardModelBT.from_pretrained(
        model_config.model_name_or_path,
        output_dim=model_config.output_dim,
        reward_token=model_config.reward_token,
        special_token_ids=special_token_ids,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
        cache_dir=cache_dir,
        **model_kwargs
    )
    if model_config.use_special_tokens:
        model.resize_token_embeddings(len(processor.tokenizer)) 

    if training_args.bf16:
        model.to(torch.bfloat16)
    if training_args.fp16:
        model.to(torch.float16)

    # create lora and peft model
    if peft_lora_config.lora_enable:
        target_modules = find_target_linear_names(model,
            num_lora_modules=peft_lora_config.num_lora_modules,
            lora_namespan_exclude=peft_lora_config.lora_namespan_exclude)
        peft_config = LoraConfig(
            target_modules=target_modules,
            r=peft_lora_config.lora_r,
            lora_alpha=peft_lora_config.lora_alpha,
            lora_dropout=peft_lora_config.lora_dropout,
            task_type=peft_lora_config.lora_task_type,
            use_rslora=peft_lora_config.use_rslora,
            bias="none",
            modules_to_save=peft_lora_config.lora_modules_to_save,
        )
        model = get_peft_model(model, peft_config)
    else:
        peft_config = None

    model.config.tokenizer_padding_side = processor.tokenizer.padding_side
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    return model, processor, peft_config