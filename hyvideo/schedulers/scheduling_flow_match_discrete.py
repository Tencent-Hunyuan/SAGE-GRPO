# Copyright 2024 Stability AI, Katherine Crowson and The HuggingFace Team. All rights reserved.
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
# Modified by @jarvizhang
# Modified from diffusers==0.29.2
#
# ==============================================================================

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, logging
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils.torch_utils import randn_tensor


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class FlowMatchDiscreteSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor


class FlowMatchDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    Euler scheduler.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        timestep_spacing (`str`, defaults to `"linspace"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        shift (`float`, defaults to 1.0):
            The shift value for the timestep schedule.
        reverse (`bool`, defaults to `True`):
            Whether to reverse the timestep schedule.
    """

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        reverse: bool = True,
        solver: str = "euler",
        use_flux_shift: bool = False,
        flux_base_shift: float = 0.5,
        flux_max_shift: float = 1.15,
        n_tokens: Optional[int] = None,
        flux_base_token=256.,
        flux_max_token=4096.,
        flux_shift_factor=1.0,
    ):
        sigmas = torch.linspace(1, 0, num_train_timesteps + 1)

        if not reverse:
            sigmas = sigmas.flip(0)

        self.sigmas = sigmas
        # the value fed to model
        self.timesteps = (sigmas[:-1] * num_train_timesteps).to(dtype=torch.float32)

        self._step_index = None
        self._begin_index = None

        self.supported_solver = [
            "euler"
        ]
        if solver not in self.supported_solver:
            raise ValueError(f"Solver {solver} not supported. Supported solvers: {self.supported_solver}")

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index
    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def _sigma_to_t(self, sigma):
        return sigma * self.config.num_train_timesteps

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None,
                      n_tokens: int = None):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            n_tokens (`int`, *optional*):
                Number of tokens in the input sequence.
        """
        self.num_inference_steps = num_inference_steps

        sigmas = torch.linspace(1, 0, num_inference_steps + 1)

        # Apply timestep shift
        if self.config.use_flux_shift:
            assert isinstance(n_tokens, int), "n_tokens should be provided for flux shift"
            mu = self.get_lin_function(x1=self.config.flux_base_token, x2=self.config.flux_max_token,
                                       y1=self.config.flux_base_shift * self.config.flux_shift_factor,
                                       y2=self.config.flux_max_shift * self.config.flux_shift_factor)(n_tokens)
            sigmas = self.flux_time_shift(mu, 1.0, sigmas)
        elif self.config.shift != 1.:
            sigmas = self.sd3_time_shift(sigmas)

        if not self.config.reverse:
            sigmas = 1 - sigmas

        self.sigmas = sigmas
        self.timesteps = (sigmas[:-1] * self.config.num_train_timesteps).to(dtype=torch.float32, device=device)

        # Reset step index
        self._step_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        return sample

    @staticmethod
    def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15):
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return lambda x: m * x + b

    @staticmethod
    def flux_time_shift(mu: float, sigma: float, t: torch.Tensor):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def sd3_time_shift(self, t: torch.Tensor):
        return (self.config.shift * t) / (1 + (self.config.shift - 1) * t)

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        n_tokens: Optional[int] = None,
        return_dict: bool = True,
    ) -> Union[FlowMatchDiscreteSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            n_tokens (`int`, *optional*):
                Number of tokens in the input sequence.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or
                tuple.

        Returns:
            [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] is
                returned, otherwise a tuple is returned where the first element is the sample tensor.
        """

        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        dt = self.sigmas[self.step_index + 1] - self.sigmas[self.step_index]

        if self.config.solver == "euler":
            prev_sample = sample + model_output.float() * dt
        else:
            raise ValueError(f"Solver {self.config.solver} not supported. Supported solvers: {self.supported_solver}")

        # Cast sample back to model compatible dtype
        # prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return FlowMatchDiscreteSchedulerOutput(prev_sample=prev_sample)


    def sde_step_with_logprob(
        self,
        model_output: torch.FloatTensor,
        sample: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        eta: float,
        prev_sample: Optional[torch.FloatTensor] = None,
        grpo: bool = True,
        sde_solver: bool = False,
        sde_type: str = "dance_grpo",
        generator: Optional[torch.Generator] = None,
        return_sqrt_dt: Optional[bool] = False,
        determistic: bool = False,
    ):
        """
        SDE step with log probability calculation for GRPO training.
        
        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            eta (`float`, defaults to 1.0):
                The noise scale factor.
            prev_sample (`torch.FloatTensor`, *optional*):
                The previous sample for log probability calculation. If None and grpo=True, will be sampled.
            grpo (`bool`, defaults to True):
                Whether to compute log probability for GRPO.
            sde_solver (`bool`, defaults to False):
                Whether to use SDE solver with log term correction.
            sde_type (`str`, defaults to "dance_grpo"):
                The type of SDE to use. "dance_grpo" or "sage_grpo".
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_sqrt_dt (`bool`, *optional*, defaults to False):
                Whether to return sqrt(-dt) as an additional output.
            determistic (`bool`, defaults to False):
                Whether to use deterministic sampling (ODE) instead of stochastic (SDE).
                When True, no noise is added during sampling.
                
        Returns:
            If grpo=True: (prev_sample, pred_original_sample, log_prob)
            If grpo=False: (prev_sample, pred_original_sample)
        """
        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        model_output=model_output.float()
        sample=sample.float()

        # Get sigma values
        sigma = self.sigmas[self.step_index]
        next_sigma = self.sigmas[self.step_index + 1]
        dt = next_sigma - sigma # next_sigma < sigma, dt<0

        # Compute predicted original sample，x0 clean sample
        pred_original_sample = sample - sigma * model_output.to(torch.float32)

        # print(f'sde_type: {sde_type}, self.sigmas: {self.sigmas}, self.step_index: {self.step_index}, dt: {dt}, sigma: {sigma}, next_sigma: {next_sigma}')
        if sde_type == 'flow_grpo':
            sigma_max = self.sigmas[1].item()
            std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma)))*eta

            prev_sample_mean = sample*(1+std_dev_t**2/(2*sigma)*dt)+model_output*(1+std_dev_t**2*(1-sigma)/(2*sigma))*dt

            # No noise is added during deterministic evaluation
            if determistic:
                prev_sample = sample + dt * model_output
            elif prev_sample is None:
                variance_noise = randn_tensor(
                    model_output.shape,
                    generator=generator,
                    device=model_output.device,
                    dtype=model_output.dtype,
                )
                # in diffusion, the sde diffusion part should be positive, so we use -dt to get the absolute value of dt
                prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1*dt) * variance_noise 

            log_prob = (
                -((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1*dt))**2))
                - torch.log(std_dev_t * torch.sqrt(-1*dt))
                - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
            )

        elif sde_type == 'cps':
            std_dev_t = next_sigma * math.sin(eta * math.pi / 2) # sigma_t in paper
            noise_estimate = sample + model_output * (1 - sigma) # predicted x_1 in paper
            prev_sample_mean = pred_original_sample * (1 - next_sigma) + noise_estimate * torch.sqrt(next_sigma**2 - std_dev_t**2)

            # No noise is added during deterministic evaluation
            if determistic:
                prev_sample = sample + dt * model_output
            elif prev_sample is None:
                variance_noise = randn_tensor(
                    model_output.shape,
                    generator=generator,
                    device=model_output.device,
                    dtype=model_output.dtype,
                )
                prev_sample = prev_sample_mean + std_dev_t * variance_noise

            # remove all constants
            log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2)

        elif sde_type == 'dance_grpo':
            # Compute deterministic mean
            prev_sample_mean = sample + dt * model_output.to(torch.float32)
            
            # Compute noise standard deviation
            delta_t = sigma - self.sigmas[self.step_index + 1]
            std_dev_t = eta * torch.sqrt(delta_t)
            
            # Apply SDE solver correction if enabled
            if sde_solver:
                score_estimate = -(sample - pred_original_sample * (1 - sigma)) / (sigma ** 2)
                log_term = -0.5 * eta ** 2 * score_estimate
                prev_sample_mean = prev_sample_mean + log_term * dt

            # No noise is added during deterministic evaluation
            if determistic:
                prev_sample = sample + dt * model_output
            elif prev_sample is None:
                # Sample prev_sample if not provided and grpo is enabled
                variance_noise = randn_tensor(
                    model_output.shape,
                    generator=generator,
                    device=model_output.device,
                    dtype=model_output.dtype,
                )
                prev_sample = prev_sample_mean + variance_noise * std_dev_t

            # Log probability of prev_sample given prev_sample_mean and std_dev_t
            # log P(x | mean, std) = -0.5 * ((x - mean)² / std²) - log(std) - log(√(2π))
            log_prob = (
                -((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2) / (2 * (std_dev_t ** 2))
                - torch.log(std_dev_t)
                - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
            )

        elif sde_type == 'sage_grpo':
            # Compute deterministic mean
            prev_sample_mean = sample + dt * model_output.to(torch.float32)
            
            # Compute noise standard deviation
            delta_t = sigma - next_sigma
            _sigma = torch.clamp(sigma, max=1-3e-3)
            std_dev_t_sq = eta**2 * (-delta_t + torch.log((1 - next_sigma) / (1 - _sigma)))
            std_dev_t = torch.sqrt(std_dev_t_sq)
            
            # Apply SDE solver correction if enabled
            if sde_solver:
                score_estimate = -(sample - pred_original_sample * (1 - sigma)) / (sigma ** 2)
                prev_sample_mean = prev_sample_mean + 0.5 * std_dev_t_sq * score_estimate

            # No noise is added during deterministic evaluation
            if determistic:
                prev_sample = sample + dt * model_output
            elif prev_sample is None:
                # Sample prev_sample if not provided and grpo is enabled
                variance_noise = randn_tensor(
                    model_output.shape,
                    generator=generator,
                    device=model_output.device,
                    dtype=model_output.dtype,
                )
                prev_sample = prev_sample_mean + variance_noise * std_dev_t
            
            # Log probability of prev_sample given prev_sample_mean and std_dev_t
            # log P(x | mean, std) = -0.5 * ((x - mean)² / std²) - log(std) - log(√(2π))
            log_prob = (
                -((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2) / (2 * (std_dev_t ** 2))
                - torch.log(std_dev_t)
                - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
            )
        
        # Mean along all but batch dimension
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        
        # Increment step index
        self._step_index += 1
        
        if return_sqrt_dt:
            return prev_sample, pred_original_sample, log_prob, prev_sample_mean, std_dev_t, torch.sqrt(-1*dt)
        return prev_sample, pred_original_sample, log_prob, prev_sample_mean, std_dev_t


    def compute_log_prob_grad_norms(self, eta, sde_type: str = "dance_grpo", sde_solver: bool = False, device=None, sigmas=None):
        """
        Estimate per-timestep ||∂ log_prob / ∂ model_output|| using the analytic Jacobians in
        ``sde_step_with_logprob``. If ``sigmas`` is not provided, this uses ``self.sigmas``.
        """
        sigma_tensor = torch.as_tensor(sigmas if sigmas is not None else self.sigmas, dtype=torch.float32)
        if sigma_tensor.ndim != 1:
            sigma_tensor = sigma_tensor.flatten()
        sigma_tensor = sigma_tensor.cpu()

        if sigma_tensor.numel() < 2:
            result = torch.ones(1, dtype=torch.float32)
            return result.to(device) if device is not None else result

        grad_norms = []
        for idx in range(sigma_tensor.numel() - 1):
            sigma = float(sigma_tensor[idx].item())
            next_sigma = float(sigma_tensor[idx + 1].item())
            dt = next_sigma - sigma
            delta_t = sigma - next_sigma

            if sde_type == "dance_grpo":
                std = eta * math.sqrt(max(delta_t, 1e-12))
                mean_jac = dt
                if sde_solver:
                    mean_jac *= 1 + 0.5 * eta ** 2 * ((1 - sigma) / max(sigma, 1e-12))
                grad_norm = abs(mean_jac) / max(std, 1e-12)

            elif sde_type == "sage_grpo":
                _sigma = min(sigma, 1 - 3e-3)
                std_sq = (eta ** 2) * (-delta_t + math.log((1 - next_sigma) / max(1 - _sigma, 1e-12)))
                if std_sq <= 0 or sigma == 0:
                    grad_norms.append(1.0)
                    continue
                std = math.sqrt(std_sq)
                mean_jac = dt
                if sde_solver:
                    mean_jac -= 0.5 * std_sq * ((1 - sigma) / sigma)
                grad_norm = abs(mean_jac) / max(std, 1e-12)

            elif sde_type == "flow_grpo":
                sigma_max = float(sigma_tensor[min(1, sigma_tensor.numel() - 1)].item())
                denom_sigma = sigma_max if sigma == 1 else sigma
                std_dev_t = math.sqrt(max(sigma, 1e-12) / max(1 - denom_sigma, 1e-12)) * eta
                noise_std = std_dev_t * math.sqrt(max(-dt, 1e-12))
                mean_jac = dt * (1 + (std_dev_t ** 2) * ((1 - sigma) / (2 * max(sigma, 1e-12))))
                grad_norm = abs(mean_jac) / max(noise_std, 1e-12)

            elif sde_type == "cps":
                std_dev_t = next_sigma * math.sin(eta * math.pi / 2)
                noise_std = max(std_dev_t, 1e-12)
                base = next_sigma ** 2 - std_dev_t ** 2
                sqrt_term = math.sqrt(max(base, 0.0))
                mean_jac = -sigma * (1 - next_sigma) + (1 - sigma) * sqrt_term
                grad_norm = abs(mean_jac) / noise_std

            else:
                raise ValueError(f"Unsupported sde_type: {sde_type}")

            if not math.isfinite(grad_norm) or grad_norm <= 0:
                grad_norm = 1.0
            grad_norms.append(grad_norm)

        grad_norms = torch.tensor(grad_norms, dtype=torch.float32)
        return grad_norms.to(device) if device is not None else grad_norms


    def compute_grad_balancing_factors(self, eta, device, sde_type: str = "dance_grpo", sde_solver: bool = False, sigmas=None):
        """
        Compute per-step scaling factors to balance gradient norms across diffusion indices for any SDE type.
        """
        grad_norms = self.compute_log_prob_grad_norms(
            eta=eta, sde_type=sde_type, sde_solver=sde_solver, device=None, sigmas=sigmas
        )

        median_norm = torch.median(grad_norms)
        if median_norm <= 0 or not torch.isfinite(median_norm):
            median_norm = torch.tensor(1.0, dtype=grad_norms.dtype)
        scales = median_norm / grad_norms
        return scales.to(device)


    def __len__(self):
        return self.config.num_train_timesteps
