# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Modified from diffusers==0.29.2
#
# ==============================================================================
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import torch
import torch.distributed as dist
import numpy as np
from PIL import Image
import pandas as pd
import re

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.utils import deprecate, replace_example_docstring
from torch.utils.data import Dataset, DataLoader

from hyvideo.commons import auto_offload_model


# kept from the migrated post-train stack.
from hyvideo.models.text_encoders.byT5.format_prompt import MultilingualPromptFormat
from hyvideo.commons import PRECISION_TO_TYPE, set_worker_seed_builder
from einops import rearrange
import random

from .hunyuan_video_pipeline import (
    HunyuanVideo_1_5_Pipeline,
    rescale_noise_cfg,
    retrieve_timesteps,
    _MULTILINGUAL_PROMPT_FORMAT_FONT_PATH,
    _MULTILINGUAL_PROMPT_FORMAT_COLOR_PATH,
)

from .data_samplers import SequentialSampler, RepeatRandomDistributedSampler

EXAMPLE_DOC_STRING = """"""


@dataclass
class PromptEncodingOutput:
    prompt_embeds: torch.Tensor
    prompt_mask: Optional[torch.Tensor]
    negative_prompt_embeds: Optional[torch.Tensor]
    negative_prompt_mask: Optional[torch.Tensor]
    prompt_embeds_2: Optional[torch.Tensor]
    prompt_mask_2: Optional[torch.Tensor]
    negative_prompt_embeds_2: Optional[torch.Tensor]
    negative_prompt_mask_2: Optional[torch.Tensor]
    do_classifier_free_guidance: bool
    extra_kwargs: Dict[str, Any]


class HunyuanVideoGRPOPipeline(HunyuanVideo_1_5_Pipeline):
    r"""
    Pipeline for text-to-video generation using HunyuanVideo with GRPO modifications.

    This pipeline inherits from HunyuanVideoPipeline and provides custom modifications
    for GRPO (Group Relative Policy Optimization) training.
    """

    @property
    def do_classifier_free_guidance(self):
        # Keep compatibility for GRPO utility paths that call pipeline helpers directly
        # before entering __call__, where _guidance_scale would normally be initialized.
        guidance_scale = getattr(self, "_guidance_scale", None)
        return guidance_scale is not None and guidance_scale > 1

    @classmethod
    def create_pipeline(cls, *args, grpo_args=None, **kwargs):
        """
        Create a GRPO pipeline by inheriting components from `HunyuanVideo_1_5_Pipeline.create_pipeline`.

        This matches the "simplified codebase" convention where the vision encoder is created inside
        `create_pipeline()`, and downstream GRPO usage should reuse that instance rather than constructing
        a separate VisionEncoder with diverging path configs.
        """
        base = HunyuanVideo_1_5_Pipeline.create_pipeline(*args, **kwargs)

        # Carry over only init-relevant configs to keep behavior consistent with the base pipeline.
        base_cfg = getattr(base, "config", {}) or {}
        passthrough_keys = {
            "flow_shift",
            "guidance_scale",
            "num_inference_steps",
            "embedded_guidance_scale",
            "vision_num_semantic_tokens",
            "vision_states_dim",
            "glyph_byT5_v2",
            "byt5_max_length",
        }
        init_cfg = {k: base_cfg[k] for k in passthrough_keys if k in base_cfg}

        byt5_kwarg = None
        if getattr(base, "byt5_model", None) is not None or getattr(base, "byt5_tokenizer", None) is not None:
            byt5_kwarg = {
                "byt5_model": getattr(base, "byt5_model", None),
                "byt5_tokenizer": getattr(base, "byt5_tokenizer", None),
                "byt5_max_length": getattr(base, "byt5_max_length", init_cfg.get("byt5_max_length", 256)),
            }

        extra_model = {"vision_encoder": getattr(base, "vision_encoder", None)}

        return cls(
            vae=base.vae,
            text_encoder=base.text_encoder,
            text_encoder_2=getattr(base, "text_encoder_2", None),
            transformer=base.transformer,
            scheduler=base.scheduler,
            args=grpo_args,
            byt5_kwarg=byt5_kwarg,
            extra_model=extra_model,
            vision_encoder=getattr(base, "vision_encoder", None),
            prompt_format=getattr(base, "prompt_format", None),
            execution_device=str(getattr(base, "execution_device", "cuda")),
            enable_offloading=getattr(base, "enable_offloading", False),
            **init_cfg,
        )

    def __init__(
        self,
        *,
        vae,
        text_encoder,
        transformer,
        scheduler,
        text_encoder_2=None,
        args=None,
        byt5_kwarg: Optional[Dict[str, Any]] = None,
        extra_model: Optional[Dict[str, Any]] = None,
        vision_encoder=None,
        prompt_format=None,
        execution_device=None,
        enable_offloading: bool = False,
        **kwargs,
    ):
        """
        GRPO pipeline wrapper around `HunyuanVideo_1_5_Pipeline`.

        Why this exists:
        - Post-train code constructs `HunyuanVideoGRPOPipeline(..., args=..., byt5_kwarg=..., extra_model=...)`.
        - The upstream open-source `HunyuanVideo_1_5_Pipeline.__init__` does NOT accept these fields.
        - We therefore adapt inputs, then call `super().__init__` so vision encoder + byT5 wiring follows
          the latest simplified codebase conventions.
        """
        self.args = args
        self.extra_model = extra_model or {}
        self.byt5_kwarg = byt5_kwarg

        if vision_encoder is None and self.extra_model is not None:
            vision_encoder = self.extra_model.get("vision_encoder", None)

        byt5_model = None
        byt5_tokenizer = None
        byt5_max_length = None
        if byt5_kwarg is not None:
            byt5_model = byt5_kwarg.get("byt5_model", None)
            byt5_tokenizer = byt5_kwarg.get("byt5_tokenizer", None)
            byt5_max_length = byt5_kwarg.get("byt5_max_length", None)

        # Default execution device to transformer.device if possible (distributed-safe).
        if execution_device is None:
            execution_device = str(getattr(transformer, "device", "cuda"))

        # Ensure prompt_format is set when byT5/glyph is used (required for _process_single_byt5_prompt).
        if prompt_format is None and byt5_model is not None:
            font_path = _MULTILINGUAL_PROMPT_FORMAT_FONT_PATH
            color_path = _MULTILINGUAL_PROMPT_FORMAT_COLOR_PATH
            if font_path and color_path:
                prompt_format = MultilingualPromptFormat(font_path=font_path, color_path=color_path)

        # Keep these consistent with args if present, otherwise fall back to upstream defaults.
        vision_num_semantic_tokens = getattr(args, "vision_num_semantic_tokens", None) if args is not None else None
        vision_states_dim = getattr(args, "vision_states_dim", None) if args is not None else None

        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
            text_encoder_2=text_encoder_2,
            byt5_model=byt5_model,
            byt5_tokenizer=byt5_tokenizer,
            byt5_max_length=byt5_max_length if byt5_max_length is not None else kwargs.pop("byt5_max_length", 256),
            prompt_format=prompt_format,
            execution_device=execution_device,
            vision_encoder=vision_encoder,
            enable_offloading=enable_offloading,
            vision_num_semantic_tokens=vision_num_semantic_tokens if vision_num_semantic_tokens is not None else kwargs.pop("vision_num_semantic_tokens", 729),
            vision_states_dim=vision_states_dim if vision_states_dim is not None else kwargs.pop("vision_states_dim", 1152),
            **kwargs,
        )

    def prepare_prompt_embeddings(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
        num_videos_per_prompt: int,
        *,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        prompt_embeds_2: Optional[torch.Tensor] = None,
        attention_mask_2: Optional[torch.Tensor] = None,
        negative_prompt_embeds_2: Optional[torch.Tensor] = None,
        negative_attention_mask_2: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
        data_type: str = "video",
        semantic_images: Optional[torch.Tensor] = None,
        text_encoder: Optional[Any] = None,
        text_encoder_2: Optional[Any] = None,
        do_classifier_free_guidance: bool = False,
        use_glyph_byT5: bool = False,
    ) -> PromptEncodingOutput:
        primary_text_encoder = text_encoder if text_encoder is not None else self.text_encoder

        encode_primary_kwargs = {
            "prompt_embeds": prompt_embeds,
            "attention_mask": attention_mask,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_attention_mask": negative_attention_mask,
            "lora_scale": lora_scale,
            "clip_skip": clip_skip,
            "data_type": data_type,
            "semantic_images": semantic_images,
            "text_encoder": primary_text_encoder,
        }
        encode_primary_kwargs = self.prepare_extra_func_kwargs(self.encode_prompt, encode_primary_kwargs)
        (
            prompt_embeds,
            negative_prompt_embeds,
            prompt_mask,
            negative_prompt_mask,
        ) = self.encode_prompt(
            prompt,
            device,
            num_videos_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            **encode_primary_kwargs,
        )

        secondary_text_encoder = (
            text_encoder_2 if text_encoder_2 is not None else getattr(self, "text_encoder_2", None)
        )
        prompt_embeds_2_out: Optional[torch.Tensor] = None
        prompt_mask_2_out: Optional[torch.Tensor] = None
        negative_prompt_embeds_2_out: Optional[torch.Tensor] = None
        negative_prompt_mask_2_out: Optional[torch.Tensor] = None
        if secondary_text_encoder is not None:
            encode_secondary_kwargs = {
                "prompt_embeds": prompt_embeds_2,
                "attention_mask": attention_mask_2,
                "negative_prompt_embeds": negative_prompt_embeds_2,
                "negative_attention_mask": negative_attention_mask_2,
                "lora_scale": lora_scale,
                "clip_skip": clip_skip,
                "text_encoder": secondary_text_encoder,
                "data_type": data_type,
                "semantic_images": None,
            }
            encode_secondary_kwargs = self.prepare_extra_func_kwargs(self.encode_prompt, encode_secondary_kwargs)
            (
                prompt_embeds_2_out,
                negative_prompt_embeds_2_out,
                prompt_mask_2_out,
                negative_prompt_mask_2_out,
            ) = self.encode_prompt(
                prompt,
                device,
                num_videos_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
                **encode_secondary_kwargs,
            )

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            if prompt_mask is not None:
                prompt_mask = torch.cat([negative_prompt_mask, prompt_mask])
            if prompt_embeds_2_out is not None:
                prompt_embeds_2_out = torch.cat([negative_prompt_embeds_2_out, prompt_embeds_2_out])
            if prompt_mask_2_out is not None:
                prompt_mask_2_out = torch.cat([negative_prompt_mask_2_out, prompt_mask_2_out])

        extra_kwargs: Dict[str, Any] = {}
        if use_glyph_byT5:
            byt5_kwargs = {"do_classifier_free_guidance": do_classifier_free_guidance}
            byt5_kwargs = self.prepare_extra_func_kwargs(self._prepare_byt5_embeddings, byt5_kwargs)
            extra_kwargs = self._prepare_byt5_embeddings(prompt, device, **byt5_kwargs)

        return PromptEncodingOutput(
            prompt_embeds=prompt_embeds,
            prompt_mask=prompt_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_mask=negative_prompt_mask,
            prompt_embeds_2=prompt_embeds_2_out,
            prompt_mask_2=prompt_mask_2_out,
            negative_prompt_embeds_2=negative_prompt_embeds_2_out,
            negative_prompt_mask_2=negative_prompt_mask_2_out,
            do_classifier_free_guidance=do_classifier_free_guidance,
            extra_kwargs=extra_kwargs,
        )

    def denoise_step(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_2: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        vision_states: Optional[torch.Tensor] = None,
        *,
        mask_type: Optional[str] = None,
        extra_kwargs: Optional[Dict[str, Any]] = None,
        embedded_guidance_scale: Optional[float] = None,
        guidance_scale: float = 1.0,
        guidance_rescale: float = 0.0,
        use_dynamic_cfg_scale: bool = False,
        step_index: Optional[int] = None,
        total_steps: Optional[int] = None,
        guidance_scale_min: Optional[float] = None,
        guidance_scale_max: Optional[float] = None,
        guidance_scale_max_step: Optional[int] = None,
    ) -> Tuple[torch.Tensor, float]:
        if extra_kwargs is None:
            extra_kwargs = {}

        do_classifier_free_guidance = (guidance_scale > 1.0) or use_dynamic_cfg_scale

        latent_model_input = (
            torch.cat([latents, latents], dim=0) if do_classifier_free_guidance else latents
        )

        vision_states_input = vision_states
        # Vision states may already be expanded for CFG depending on upstream preparation.
        # Only repeat if it still matches the non-CFG batch size.
        if (
            do_classifier_free_guidance
            and vision_states is not None
            and vision_states.shape[0] == latents.shape[0]
        ):
            vision_states_input = vision_states.repeat(2, 1, 1)

        latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep)

        # Expand timestep to match latent_model_input batch size
        # Handle two cases:
        # 1. timestep is a scalar tensor (from __call__): shape [] or [1]
        # 2. timestep is a batch tensor (from grpo_one_step): shape [batch_size]
        if do_classifier_free_guidance:
            # CFG: latent_model_input is [batch_size * 2, ...]
            if timestep.dim() == 0 or (timestep.dim() == 1 and timestep.shape[0] == 1):
                # Case 1: scalar tensor from __call__, repeat for all samples
                # [t] -> [t, t, t, t, t, t, t, t] for batch_size=4
                t_expand = timestep.repeat(latent_model_input.shape[0])
            else:
                # Case 2: batch tensor from grpo_one_step, shape [batch_size]
                # Use repeat_interleave to repeat each element: [t0, t1, t2, t3] -> [t0, t0, t1, t1, t2, t2, t3, t3]
                t_expand = timestep.repeat_interleave(2, dim=0)
        else:
            # No CFG: latent_model_input is [batch_size, ...]
            if timestep.dim() == 0 or (timestep.dim() == 1 and timestep.shape[0] == 1):
                # Case 1: scalar tensor from __call__, repeat for all samples
                t_expand = timestep.repeat(latent_model_input.shape[0])
            else:
                # Case 2: batch tensor from grpo_one_step, already matches
                t_expand = timestep
        target_dtype = PRECISION_TO_TYPE[self.args.precision]
        autocast_enabled = (target_dtype != torch.float32) and not self.args.val_disable_autocast

        if embedded_guidance_scale is not None:
            guidance_expand = (
                torch.tensor(
                    [embedded_guidance_scale] * latent_model_input.shape[0],
                    dtype=torch.float32,
                    device=latents.device,
                ).to(target_dtype)
                * 1000.0
            )
        else:
            guidance_expand = None

        # print(f'Rank: {dist.get_rank()}, latent_model_input: {latent_model_input.shape}, t_expand: {t_expand.shape}, prompt_embeds: {prompt_embeds.shape}, vision_states_input: {vision_states_input.shape}, extra_kwargs["byt5_text_states"]: {extra_kwargs["byt5_text_states"].shape}, extra_kwargs["byt5_text_mask"]: {extra_kwargs["byt5_text_mask"].shape}')
        with torch.autocast(device_type="cuda", dtype=target_dtype, enabled=autocast_enabled):
            output = self.transformer(
                latent_model_input,
                t_expand,
                prompt_embeds,
                prompt_embeds_2,
                prompt_mask,
                vision_states=vision_states_input,
                mask_type=mask_type,
                guidance=guidance_expand,
                return_dict=False,
                extra_kwargs=extra_kwargs,
            )
            noise_pred = output[0]

        current_guidance_scale = guidance_scale
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

            if use_dynamic_cfg_scale:
                current_guidance_scale = self.get_dynamic_guidance_scale(
                    step_index,
                    total_steps,
                    guidance_scale_min,
                    guidance_scale_max,
                    guidance_scale_max_step,
                )

            noise_pred = noise_pred_uncond + current_guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            if guidance_rescale > 0.0:
                noise_pred = rescale_noise_cfg(
                    noise_pred,
                    noise_pred_text,
                    guidance_rescale=guidance_rescale,
                )

        return noise_pred, current_guidance_scale

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: int,
        width: int,
        video_length: int,
        eta: float,
        data_type: str = "video",
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        use_dynamic_cfg_scale: bool = False,
        guidance_scale_min: Optional[float] = None,
        guidance_scale_max: Optional[float] = None,
        guidance_scale_max_step: Optional[float] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pt",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        vae_ver: str = "88-4c-sd",
        enable_tiling: bool = False,
        n_tokens: Optional[int] = None,
        embedded_guidance_scale: Optional[float] = None,
        multitask_mask_training_type: Optional[str] = None,
        multitask_type: Optional[str] = None,
        cond_latents: Optional[torch.Tensor] = None,
        uncond_latents: Optional[torch.Tensor] = None,
        multitask_mask: Optional[torch.Tensor] = None,
        semantic_images: Optional[torch.Tensor] = None,
        reference_image=None,
        vision_num_semantic_tokens: Optional[int] = None,
        vision_states_dim: Optional[int] = None,
        bucket_hw_base_size=None,
        bucket_hw_bucket_stride=None,
        sde_type: str = "dance_grpo",
        determistic: Union[bool, List[bool]] = False,  # Used in GRPO for progressive training (note: typo kept for consistency)
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`):
                The height in pixels of the generated image.
            width (`int`):
                The width in pixels of the generated image.
            video_length (`int`):
                The number of frames in the generated video.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.

            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`HunyuanVideoPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~HunyuanVideoPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`HunyuanVideoPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)
        target_dtype = PRECISION_TO_TYPE[self.args.precision]
        self.use_dynamic_cfg_scale = use_dynamic_cfg_scale

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Default height and width to unet
        # height = height or self.transformer.config.sample_size * self.vae_scale_factor
        # width = width or self.transformer.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        # self.check_inputs(
        #     prompt,
        #     height,
        #     width,
        #     video_length,
        #     callback_steps,
        #     negative_prompt,
        #     prompt_embeds,
        #     negative_prompt_embeds,
        #     callback_on_step_end_tensor_inputs,
        #     vae_ver=vae_ver,
        # )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # device = (
            # torch.device(f"cuda:{dist.get_rank()}")
            # if dist.is_initialized()
            # else self._execution_device
        # )
        device = self.transformer.device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None)
            if self.cross_attention_kwargs is not None
            else None
        )

        with auto_offload_model(self.text_encoder, device, enabled=self.enable_offloading):
            if getattr(self, "text_encoder_2", None) is not None:
                text_encoder_2_ctx = auto_offload_model(self.text_encoder_2, device, enabled=self.enable_offloading)
            else:
                text_encoder_2_ctx = None
            if getattr(self, "byt5_model", None) is not None:
                byt5_ctx = auto_offload_model(self.byt5_model, device, enabled=self.enable_offloading)
            else:
                byt5_ctx = None

            if text_encoder_2_ctx is not None and byt5_ctx is not None:
                with text_encoder_2_ctx, byt5_ctx:
                    prompt_bundle = self.prepare_prompt_embeddings(
                        prompt,
                        device,
                        num_videos_per_prompt,
                        negative_prompt=negative_prompt,
                        prompt_embeds=prompt_embeds,
                        attention_mask=attention_mask,
                        negative_prompt_embeds=negative_prompt_embeds,
                        negative_attention_mask=negative_attention_mask,
                        lora_scale=lora_scale,
                        clip_skip=self.clip_skip,
                        data_type=data_type,
                        semantic_images=semantic_images,
                        text_encoder=self.text_encoder,
                        text_encoder_2=self.text_encoder_2,
                        do_classifier_free_guidance=self.do_classifier_free_guidance,
                        use_glyph_byT5=self.args.glyph_byT5_v2,
                    )
            elif text_encoder_2_ctx is not None:
                with text_encoder_2_ctx:
                    prompt_bundle = self.prepare_prompt_embeddings(
                        prompt,
                        device,
                        num_videos_per_prompt,
                        negative_prompt=negative_prompt,
                        prompt_embeds=prompt_embeds,
                        attention_mask=attention_mask,
                        negative_prompt_embeds=negative_prompt_embeds,
                        negative_attention_mask=negative_attention_mask,
                        lora_scale=lora_scale,
                        clip_skip=self.clip_skip,
                        data_type=data_type,
                        semantic_images=semantic_images,
                        text_encoder=self.text_encoder,
                        text_encoder_2=self.text_encoder_2,
                        do_classifier_free_guidance=self.do_classifier_free_guidance,
                        use_glyph_byT5=self.args.glyph_byT5_v2,
                    )
            elif byt5_ctx is not None:
                with byt5_ctx:
                    prompt_bundle = self.prepare_prompt_embeddings(
                        prompt,
                        device,
                        num_videos_per_prompt,
                        negative_prompt=negative_prompt,
                        prompt_embeds=prompt_embeds,
                        attention_mask=attention_mask,
                        negative_prompt_embeds=negative_prompt_embeds,
                        negative_attention_mask=negative_attention_mask,
                        lora_scale=lora_scale,
                        clip_skip=self.clip_skip,
                        data_type=data_type,
                        semantic_images=semantic_images,
                        text_encoder=self.text_encoder,
                        text_encoder_2=self.text_encoder_2,
                        do_classifier_free_guidance=self.do_classifier_free_guidance,
                        use_glyph_byT5=self.args.glyph_byT5_v2,
                    )
            else:
                prompt_bundle = self.prepare_prompt_embeddings(
                    prompt,
                    device,
                    num_videos_per_prompt,
                    negative_prompt=negative_prompt,
                    prompt_embeds=prompt_embeds,
                    attention_mask=attention_mask,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_attention_mask=negative_attention_mask,
                    lora_scale=lora_scale,
                    clip_skip=self.clip_skip,
                    data_type=data_type,
                    semantic_images=semantic_images,
                    text_encoder=self.text_encoder,
                    text_encoder_2=self.text_encoder_2,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    use_glyph_byT5=self.args.glyph_byT5_v2,
                )
        prompt_embeds = prompt_bundle.prompt_embeds
        prompt_mask = prompt_bundle.prompt_mask
        negative_prompt_embeds = prompt_bundle.negative_prompt_embeds
        negative_prompt_mask = prompt_bundle.negative_prompt_mask
        prompt_embeds_2 = prompt_bundle.prompt_embeds_2
        prompt_mask_2 = prompt_bundle.prompt_mask_2
        negative_prompt_embeds_2 = prompt_bundle.negative_prompt_embeds_2
        negative_prompt_mask_2 = prompt_bundle.negative_prompt_mask_2
        extra_kwargs = prompt_bundle.extra_kwargs

        # 4. Prepare timesteps
        extra_set_timesteps_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.set_timesteps, {"n_tokens": n_tokens}
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            **extra_set_timesteps_kwargs,
        )

        # Keep pixel sizes for VAE / vision preprocessing; convert to latent sizes for transformer later.
        pixel_height, pixel_width = height, width

        # Align to simplified hyvideo semantics: task_type is driven by whether a reference image is provided,
        # unless explicitly specified.
        task_type = multitask_type if multitask_type in ("t2v", "i2v") else ("i2v" if reference_image is not None else "t2v")

        # Map bucket_hw_base_size to a hyvideo target resolution string used by `_prepare_vision_states`.
        target_resolution = None
        if bucket_hw_base_size is not None and hasattr(self, "target_size_config"):
            for res, cfg in self.target_size_config.items():
                if cfg.get("bucket_hw_base_size", None) == bucket_hw_base_size:
                    target_resolution = res
                    break
        if target_resolution is None:
            target_resolution = getattr(self, "ideal_resolution", None) or "480p"

        # Convert pixel-space sizes into latent-space sizes
        latent_video_length, latent_height, latent_width = self.get_latent_size(video_length, height, width)

        # Keep the downstream code using the latent-space `video_length` (consistent with VAE).
        video_length = latent_video_length
        height = latent_height
        width = latent_width

        if n_tokens is None:
            n_tokens = int(video_length) * int(height) * int(width)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            video_length,
            # prompt_embeds.dtype,
            target_dtype,
            device,
            generator,
            latents,
        )

        # ------------------------------------------------------------------
        # Non-GRPO parts: keep consistent with simplified hyvideo pipelines.
        # - compute multitask_mask & cond_latents via base helpers
        # - compute vision_states via base helper under auto_offload_model
        # ------------------------------------------------------------------
        multitask_mask = self.get_task_mask(task_type, video_length) if multitask_mask is None else multitask_mask

        # Accept both PIL and numpy reference images from legacy callsites.
        ref_image_pil = None
        if reference_image is not None:
            if isinstance(reference_image, Image.Image):
                ref_image_pil = reference_image
            elif isinstance(reference_image, np.ndarray):
                ref_np = reference_image
                if ref_np.ndim == 4:
                    ref_np = ref_np[0]
                ref_image_pil = Image.fromarray(ref_np.astype("uint8"))
            else:
                ref_image_pil = reference_image

        with auto_offload_model(self.vae, device, enabled=self.enable_offloading):
            image_cond = self.get_image_condition_latents(task_type, ref_image_pil, pixel_height, pixel_width)
        cond_latents = self._prepare_cond_latents(task_type, image_cond, latents, multitask_mask)

        semantic_images_np = None
        if ref_image_pil is not None and isinstance(ref_image_pil, Image.Image):
            semantic_images_np = np.array(ref_image_pil)

        with auto_offload_model(self.vision_encoder, device, enabled=self.enable_offloading):
            vision_states = self._prepare_vision_states(semantic_images_np, target_resolution, latents, device)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.step, {"generator": generator, "eta": eta},
        )

        target_dtype = PRECISION_TO_TYPE[self.args.precision]
        autocast_enabled = (
            target_dtype != torch.float32
        ) and not self.args.val_disable_autocast
        vae_dtype = PRECISION_TO_TYPE[self.args.vae_precision]
        vae_autocast_enabled = (
            vae_dtype != torch.float32
        ) and not self.args.val_disable_autocast

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        # Legacy hint for old transformer branches; kept for backward-compatibility.
        extra_kwargs["is_token_replace"] = bool(multitask_mask_training_type == "token_replace" and task_type == "i2v")

        all_latents = [latents]
        all_log_probs = []
        all_prev_means = []
        all_std_devs = []
        
        # Get rank to control progress bar display (only show on rank 0)
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        # if is_progress_bar:
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            import tqdm
            with auto_offload_model(self.transformer, device, enabled=self.enable_offloading):
                for i, t in tqdm.tqdm(enumerate(timesteps), disable=rank >= 1):
                    if self.interrupt:
                        continue

                    # Handle determistic parameter: can be a list (per-timestep) or a single bool
                    if isinstance(determistic, list):
                        assert len(determistic) == num_inference_steps, \
                            f"determistic list length ({len(determistic)}) must match num_inference_steps ({num_inference_steps})"
                        determistic_i = determistic[i]
                    else:
                        determistic_i = determistic

                    # Follow hyvideo convention: concatenate condition (image + mask) in channel dim.
                    latent_model_input = torch.concat([latents, cond_latents], dim=1)

                    noise_pred, updated_guidance_scale = self.denoise_step(
                        latent_model_input,
                        t,
                        prompt_embeds,
                        prompt_embeds_2,
                        prompt_mask,
                        vision_states=vision_states,
                        mask_type=task_type,
                        extra_kwargs=extra_kwargs,
                        embedded_guidance_scale=embedded_guidance_scale,
                        guidance_scale=self._guidance_scale,
                        guidance_rescale=self.guidance_rescale,
                        use_dynamic_cfg_scale=self.use_dynamic_cfg_scale,
                        step_index=i,
                        total_steps=len(timesteps),
                        guidance_scale_min=guidance_scale_min,
                        guidance_scale_max=guidance_scale_max,
                        guidance_scale_max_step=guidance_scale_max_step,
                    )
                    self._guidance_scale = updated_guidance_scale
                    
                    latents, pred_latents_original, log_prob, prev_mean, std_dev_t = self.scheduler.sde_step_with_logprob(
                        noise_pred, latents, t, eta=eta, prev_sample=None, grpo=True, sde_solver=True, sde_type=sde_type, determistic=determistic_i
                    )
                    all_latents.append(latents)
                    all_log_probs.append(log_prob)
                    all_prev_means.append(prev_mean)
                    all_std_devs.append(std_dev_t)

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop(
                            "negative_prompt_embeds", negative_prompt_embeds
                        )

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                    ):
                        if progress_bar is not None:
                            progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            step_idx = i // getattr(self.scheduler, "order", 1)
                            callback(step_idx, t, latents)


        # pred_latents_original = pred_latents_original.to(torch.float32) / self.vae.config.scaling_factor
        latents = latents.to(torch.float32) / self.vae.config.scaling_factor
        all_latents = torch.stack(all_latents, dim=1)  # (batch_size, num_steps + 1, ...)
        all_log_probs = torch.stack(all_log_probs, dim=1)  # (batch_size, num_steps, 1)
        all_prev_means = torch.stack(all_prev_means, dim=1)  # (batch_size, num_steps, ...)
        all_std_devs = torch.stack(all_std_devs, dim=0)  # (batch_size, num_steps, ...)
        
        if len(latents.shape) == 4:
            latents = latents.unsqueeze(2)
        elif len(latents.shape) == 5:
            pass
        else:
            raise ValueError(
                f"Only support latents with shape (b, c, h, w) or (b, c, f, h, w), but got {latents.shape}."
            )

        if (hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor):
            latents = (latents / self.vae.config.scaling_factor + self.vae.config.shift_factor)
        else:
            latents = latents / self.vae.config.scaling_factor
        
        with auto_offload_model(self.vae, device, enabled=self.enable_offloading), torch.autocast(
            device_type="cuda", dtype=vae_dtype, enabled=vae_autocast_enabled
        ):
            self.vae.enable_tiling()
            videos = self.vae.decode(latents, return_dict=False, generator=generator)[0]
            self.vae.disable_tiling()
        
        videos = (videos / 2 + 0.5).clamp(0, 1)
        videos = videos.cpu().float()

        return videos, all_latents, all_log_probs, all_prev_means, all_std_devs 


class VideoPromptDataset(Dataset):
    def __init__(self, args, logger,         
        text_encoder=None,
        text_encoder_2=None,
        byt5_tokenizer=None, 
        seed=42
    ):
        super().__init__()
        self.args = args
        self.text_encoder=text_encoder
        self.text_encoder_2=text_encoder_2
        self.byt5_tokenizer=byt5_tokenizer
        self.logger = logger
        self.seed = seed
        self._repeat_random_map = None
        self.load_data()

    def load_data(self):
        self.data = []
        for csv_path in self.args.train_video_csv:
            df = pd.read_csv(csv_path)
            required_cols = ['prompt']
            missing_cols = [col for col in required_cols if col not in df.columns]
            assert missing_cols == [], f"Missing required columns: {missing_cols}"

            # Handle index column
            if 'index' not in df.columns:
                df['index'] = range(len(df))
            
            # Handle seed column
            if 'seed' not in df.columns:
                df['seed'] = self.seed
            
            # Ensure stable column order
            required_cols = ['index', 'prompt', 'seed']
            if 'ref_image_path' in df.columns:
                required_cols.append('ref_image_path')
            df = df[required_cols]
            self.data.extend(df.to_dict('records'))
        self.total_length = len(self.data)
                    
    def __len__(self):
        if hasattr(self, '_repeat_random_map') and self._repeat_random_map is not None:
            return len(self._repeat_random_map)
        return len(self.data)
         
    def get_text_tokens(self, text_encoder, description, text_len):
        text_inputs = text_encoder.text2tokens(description, data_type='video', max_length=text_len)
        text_ids = text_inputs["input_ids"].squeeze(0)
        text_mask = text_inputs["attention_mask"].squeeze(0)
        return text_ids, text_mask

    def get_byt5_text_tokens(self, byt5_tokenizer, byt5_max_length, text_prompt):
        byt5_text_inputs = byt5_tokenizer(
            text_prompt,
            padding="max_length",
            max_length=byt5_max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        byt5_text_ids = byt5_text_inputs.input_ids
        byt5_text_mask = byt5_text_inputs.attention_mask

        return byt5_text_ids, byt5_text_mask

    def get_train_byt5_text_tokens(self, args, byt5_tokenizer, prompt):
        # get byt5 text token id
        byt5_text_ids = torch.zeros((args.byt5_max_length), dtype=torch.int64)
        byt5_text_mask = torch.zeros((args.byt5_max_length), dtype=torch.int64)
        byt5_text_valid = torch.tensor(0, dtype=torch.int64)
        if args.glyph_byT5_v2:
            prompt_format = MultilingualPromptFormat(
                font_path=args.multilingual_prompt_format_font_path,
                color_path=args.multilingual_prompt_format_color_path)
            pattern = r'\"(.*?)\"|“(.*?)”'
            matches = re.findall(pattern, prompt)
            glyph_byT5_text_list = [match[0] or match[1] for match in matches]

            if len(glyph_byT5_text_list) > 0:
                if random.random() < args.video_uncond_p_byt5:
                    glyph_byT5_text_list = ["" for _ in range(len(glyph_byT5_text_list))]

                text_prompt_style_list = [{'color': None, 'font-family': None} for _ in range(len(glyph_byT5_text_list))]
                glyph_byT5_text_formatted = prompt_format.format_prompt(glyph_byT5_text_list, text_prompt_style_list)

                byt5_text_ids, byt5_text_mask = self.get_byt5_text_tokens(
                    byt5_tokenizer, args.byt5_max_length, glyph_byT5_text_formatted)
                byt5_text_ids = byt5_text_ids.squeeze(0)
                byt5_text_mask = byt5_text_mask.squeeze(0)
                byt5_text_valid = torch.tensor(1, dtype=torch.int64)
        return byt5_text_ids, byt5_text_mask, byt5_text_valid


    def __getitem__(self, idx):
        if hasattr(self, '_repeat_random_map') and self._repeat_random_map is not None:
            return self._repeat_random_map[idx]
        # key: index, prompt, seed
        sample = self.data[idx]
        index = sample['index']
        prompt = sample['prompt']
        text_ids, text_mask = self.get_text_tokens(
            self.text_encoder, prompt, self.args.text_len
        )
        byt5_text_ids, byt5_text_mask, byt5_text_valid = self.get_train_byt5_text_tokens(self.args, self.byt5_tokenizer, prompt)
        
        result = (
            sample['index'],
            sample['prompt'],
            int(sample['seed']) if isinstance(sample['seed'], str) else sample['seed'],
            sample.get('ref_image_path', ""),
            text_ids.clone(),
            text_mask.clone(),
            byt5_text_ids.clone(),
            byt5_text_mask.clone(),
            byt5_text_valid,
            {
                "text": prompt,
                "videoid": index,
                "type": "video",
            },
        )
        return result 

    def prepare_repeat_random_slots(self, num_generations, world_size, global_seed=42):
        # get all data
        prompt_list = [x['prompt'] for x in self.data]
        index_list = [x['index'] for x in self.data]
        seed_list = [int(x['seed']) for x in self.data]
        total = len(prompt_list)
        slot_list = []
        # slot: prompt_id * num_generations + sample_id
        for pidx in range(total):
            for s in range(num_generations):
                slot_list.append({
                    "prompt_index": pidx,
                    "index": index_list[pidx],
                    "prompt": prompt_list[pidx],
                    "seed": seed_list[pidx] + s * 10000,   # strong consistency offset, avoid different slot seed conflict
                    "sample_slot": s,
                })
        # pad to world_size
        if len(slot_list) < world_size:
            if not slot_list:
                slot_list = [None] * world_size  # 或者根据业务需求设置其他默认值
            else:
                slot_list += [slot_list[-1]] * (world_size - len(slot_list))
        self._repeat_random_map = slot_list

def get_post_train_video_dataloader(args, logger, text_encoder, text_encoder_2, dp_degree, dp_rank, byt5_kwarg, local_seed=None):
    if args.post_train_type == "grpo":
        video_dataset = VideoPromptDataset(args, logger, text_encoder, text_encoder_2,
            byt5_tokenizer=byt5_kwarg["byt5_tokenizer"] if byt5_kwarg is not None else None)
    else:
        raise ValueError(f"Invalid post-train type: {args.post_train_type}")

        # Use RepeatRandomDistributedSampler: split data to each DP rank to execute, reduce peak memory on single card
        # mini_repeat_count should be num_generations so that each prompt is repeated num_generations times
        # and distributed across different ranks for parallel generation
    num_generations = getattr(args, 'num_generations', 1)
    video_batch_size = args.video_micro_batch_size[-1]
    
    # CRITICAL: Validate configuration to ensure RepeatRandomDistributedSampler works correctly with group logic
    # For RepeatRandomDistributedSampler to work correctly with our group logic,
    # video_batch_size and num_generations must satisfy one of these conditions:
    # 1. video_batch_size >= num_generations and video_batch_size % num_generations == 0
    #    -> Each rank has complete group(s), ranks_per_group = 1
    # 2. video_batch_size < num_generations and num_generations % video_batch_size == 0
    #    -> Multiple ranks form a group, ranks_per_group > 1
    #
    # Otherwise, samples within a rank may come from different prompts, breaking group structure!
    if video_batch_size >= num_generations:
        # Case 1: Each rank should have complete group(s)
        if video_batch_size % num_generations != 0:
            raise ValueError(
                f"Invalid configuration: video_batch_size ({video_batch_size}) must be divisible by "
                f"num_generations ({num_generations}) when video_batch_size >= num_generations. "
                f"This ensures each rank processes complete group(s). "
                f"Current remainder: {video_batch_size % num_generations}. "
                f"Please adjust video_micro_batch_size or num_generations."
            )
        ranks_per_group = 1
        num_groups_per_rank = video_batch_size // num_generations
        logger.info(f"[Config Validation] video_batch_size={video_batch_size}, num_generations={num_generations}, "
                   f"ranks_per_group={ranks_per_group}, num_groups_per_rank={num_groups_per_rank}")
    else:
        # Case 2: Multiple ranks should form a group
        if num_generations % video_batch_size != 0:
            raise ValueError(
                f"Invalid configuration: num_generations ({num_generations}) must be divisible by "
                f"video_batch_size ({video_batch_size}) when video_batch_size < num_generations. "
                f"This ensures multiple ranks can form complete group(s). "
                f"Current remainder: {num_generations % video_batch_size}. "
                f"Please adjust video_micro_batch_size or num_generations."
            )
        ranks_per_group = num_generations // video_batch_size
        num_groups_per_rank = 1
        logger.info(f"[Config Validation] video_batch_size={video_batch_size}, num_generations={num_generations}, "
                   f"ranks_per_group={ranks_per_group}, num_groups_per_rank={num_groups_per_rank}")
    
    video_sampler = RepeatRandomDistributedSampler(
        video_dataset,
        num_replicas=dp_degree,
        rank=dp_rank,
        shuffle=True,
        seed=args.global_seed,
        drop_last=True,
        batch_size=video_batch_size,
        repeat_count=1,
        mini_repeat_count=num_generations,  # Each prompt repeated num_generations times, distributed across ranks
    )

    video_dataloader = DataLoader(
        video_dataset,
        batch_size=args.video_micro_batch_size[-1],
        shuffle=False,
        sampler=video_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=None if args.num_workers == 0 else args.prefetch_factor,
        worker_init_fn=set_worker_seed_builder(dp_rank),
        persistent_workers=True if args.num_workers > 0 else False,
    )
    logger.info('Dataloader init done')
    return video_dataset, video_sampler, video_dataloader
