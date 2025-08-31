import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, logging
from diffusers.schedulers.scheduling_utils import SchedulerMixin


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
    ):
        sigmas = torch.linspace(1, 0, num_train_timesteps + 1)

        if not reverse:
            sigmas = sigmas.flip(0)

        self.sigmas = sigmas
        # the value fed to model
        self.timesteps = (sigmas[:-1] * num_train_timesteps).to(dtype=torch.float32)
        self.timesteps_full = (sigmas * num_train_timesteps).to(dtype=torch.float32)

        self._step_index = None
        self._begin_index = None

        self.supported_solver = [
            "euler",
            "heun-2", "midpoint-2",
            "kutta-4",
        ]
        if solver not in self.supported_solver:
            raise ValueError(f"Solver {solver} not supported. Supported solvers: {self.supported_solver}")

        # empty dt and derivative (for heun)
        self.derivative_1 = None
        self.derivative_2 = None
        self.derivative_3 = None
        self.dt = None

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

    @property
    def state_in_first_order(self):
        return self.derivative_1 is None

    @property
    def state_in_second_order(self):
        return self.derivative_2 is None

    @property
    def state_in_third_order(self):
        return self.derivative_3 is None

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
            mu = self.get_lin_function(y1=self.config.flux_base_shift, y2=self.config.flux_max_shift)(n_tokens)
            sigmas = self.flux_time_shift(mu, 1.0, sigmas)
        elif self.config.shift != 1.:
            sigmas = self.sd3_time_shift(sigmas)

        if not self.config.reverse:
            sigmas = 1 - sigmas

        self.sigmas = sigmas
        self.timesteps = (sigmas[:-1] * self.config.num_train_timesteps).to(dtype=torch.float32, device=device)
        self.timesteps_full = (sigmas * self.config.num_train_timesteps).to(dtype=torch.float32, device=device)

        # empty dt and derivative (for kutta)
        self.derivative_1 = None
        self.derivative_2 = None
        self.derivative_3 = None
        self.dt = None

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
            pred_uncond: torch.FloatTensor = None,
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
        model_output = model_output.to(torch.float32)
        pred_uncond = pred_uncond.to(torch.float32) if pred_uncond is not None else None

        # dt = self.sigmas[self.step_index + 1] - self.sigmas[self.step_index]
        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]

        last_inner_step = True
        if self.config.solver == "euler":
            derivative, dt, sample, last_inner_step = self.first_order_method(model_output, sigma, sigma_next, sample)
        elif self.config.solver in ["heun-2", "midpoint-2"]:
            derivative, dt, sample, last_inner_step = self.second_order_method(model_output, sigma, sigma_next, sample)
        elif self.config.solver == "kutta-4":
            derivative, dt, sample, last_inner_step = self.fourth_order_method(model_output, sigma, sigma_next, sample)
        else:
            raise ValueError(f"Solver {self.config.solver} not supported. Supported solvers: {self.supported_solver}")

        prev_sample = sample + derivative * dt

        # Cast sample back to model compatible dtype
        # prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        if last_inner_step:
            self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return FlowMatchDiscreteSchedulerOutput(prev_sample=prev_sample)

    def first_order_method(self, model_output, sigma, sigma_next, sample):
        derivative = model_output.float()
        dt = sigma_next - sigma
        return derivative, dt, sample, True

    def second_order_method(self, model_output, sigma, sigma_next, sample):
        if self.state_in_first_order:
            # store for 2nd order step
            self.derivative_1 = model_output
            self.dt = sigma_next - sigma
            self.sample = sample

            derivative = model_output
            if self.config.solver == 'heun-2':
                dt = self.dt
            elif self.config.solver == 'midpoint-2':
                dt = self.dt / 2
            else:
                raise NotImplementedError(f"Solver {self.config.solver} not supported.")
            last_inner_step = False

        else:
            if self.config.solver == 'heun-2':
                derivative = 0.5 * (self.derivative_1 + model_output)
            elif self.config.solver == 'midpoint-2':
                derivative = model_output
            else:
                raise NotImplementedError(f"Solver {self.config.solver} not supported.")

            # 3. take prev timestep & sample
            dt = self.dt
            sample = self.sample
            last_inner_step = True

            # free dt and derivative
            # Note, this puts the scheduler in "first order mode"
            self.derivative_1 = None
            self.dt = None
            self.sample = None

        return derivative, dt, sample, last_inner_step

    def fourth_order_method(self, model_output, sigma, sigma_next, sample):
        if self.state_in_first_order:
            self.derivative_1 = model_output
            self.dt = sigma_next - sigma
            self.sample = sample
            derivative = model_output
            dt = self.dt / 2
            last_inner_step = False

        elif self.state_in_second_order:
            self.derivative_2 = model_output
            derivative = model_output
            dt = self.dt / 2
            last_inner_step = False

        elif self.state_in_third_order:
            self.derivative_3 = model_output
            derivative = model_output
            dt = self.dt
            last_inner_step = False

        else:
            derivative = 1/6 * self.derivative_1 + 1/3 * self.derivative_2 + 1/3 * self.derivative_3 + 1/6 * model_output

            # 3. take prev timestep & sample
            dt = self.dt
            sample = self.sample
            last_inner_step = True

            # free dt and derivative
            # Note, this puts the scheduler in "first order mode"
            self.derivative_1 = None
            self.derivative_2 = None
            self.derivative_3 = None
            self.dt = None
            self.sample = None

        return derivative, dt, sample, last_inner_step

    def __len__(self):
        return self.config.num_train_timesteps
