import math
import time
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple

def compute_vector_norm(
    tensor: torch.Tensor, 
    dim: Optional[Tuple[int, ...]] = None, 
    keepdim: bool = True
) -> torch.Tensor:
    """
    Safely computes the Euclidean (L2) vector norm over an arbitrary number of dimensions.
    """
    if hasattr(torch.linalg, "vector_norm"):
        return torch.linalg.vector_norm(tensor, ord=2, dim=dim, keepdim=keepdim)
    if dim is None:
        return torch.sqrt(torch.sum(tensor ** 2))
    return torch.sqrt(torch.sum(tensor ** 2, dim=dim, keepdim=keepdim))

class SpectralGuidanceController:
    """
    Dedicated Frequency-Domain (Fourier) Latent Guidance Controller.
    
    Treats Classifier-Free Guidance (CFG) as an audio-style graphic equalizer:
    - Low-frequency band: Governs global lighting, overall composition, color temperature, and mood.
    - Mid-frequency band: Governs subject anatomy, silhouettes, structural geometry, and boundaries.
    - High-frequency band: Governs fine textures, skin pores, fabric weave, micro-edges, and sharpness.
    - Spectral Tilt: Controls overall energy decay slope across frequencies (1/f^alpha).
    - High-Shelf Damping: Smoothly rolls off extreme Nyquist frequencies to prevent digital ringing/frying.
    """
    def __init__(
        self,
        cfg_scale: float = 4.5,
        low_freq_gain: float = 1.0,
        mid_freq_gain: float = 1.0,
        high_freq_gain: float = 1.0,
        crossover_low: float = 0.12,
        crossover_high: float = 0.45,
        spectral_tilt: float = 0.0,
        high_shelf_damping: float = 0.15,
        rescale_factor: float = 0.70,
        rescale_mode: str = "std",
        schedule_mode: str = "linear",
        start_percent: float = 0.0,
        end_percent: float = 1.0,
        debug: bool = False,
    ):
        self.cfg_scale = cfg_scale
        self.low_freq_gain = low_freq_gain
        self.mid_freq_gain = mid_freq_gain
        self.high_freq_gain = high_freq_gain
        self.crossover_low = crossover_low
        self.crossover_high = crossover_high
        self.spectral_tilt = spectral_tilt
        self.high_shelf_damping = high_shelf_damping
        self.rescale_factor = rescale_factor
        self.rescale_mode = rescale_mode
        self.schedule_mode = schedule_mode
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.debug = debug

        # Trajectory tracking registers
        self.initial_sigma: Optional[float] = None
        self.last_sigma: Optional[float] = None
        self.last_call_time: float = 0.0
        self.prev_shape: Optional[Tuple[int, ...]] = None
        self.step_count: int = 0
        
        # Cached 2D frequency filter kernel
        self._cached_hw: Optional[Tuple[int, int]] = None
        self._cached_device: Optional[torch.device] = None
        self._cached_filter: Optional[torch.Tensor] = None

    def reset(self):
        """Resets controller state registers for a clean sampling run."""
        self.initial_sigma = None
        self.last_sigma = None
        self.prev_shape = None
        self.step_count = 0
        self._cached_hw = None
        self._cached_device = None
        self._cached_filter = None

    def _build_filter_kernel(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """
        Synthesizes a 2D radial frequency modulation mask with smooth sigmoidal
        crossovers to prevent Gibbs ringing artifacts.
        """
        if (
            self._cached_filter is not None and 
            self._cached_hw == (H, W) and 
            self._cached_device == device
        ):
            return self._cached_filter

        # Compute normalized frequencies in [0.0, 1.0]
        fy = torch.fft.fftfreq(H, device=device).unsqueeze(1)
        fx = torch.fft.rfftfreq(W, device=device).unsqueeze(0)
        radius = torch.sqrt((2.0 * fy) ** 2 + (2.0 * fx) ** 2).clamp(0.0, 1.0)

        # Smooth sigmoidal band transitions (slope factor = 16.0 for gentle crossover)
        low_mask = torch.sigmoid((self.crossover_low - radius) * 16.0)
        high_mask = torch.sigmoid((radius - self.crossover_high) * 16.0)
        mid_mask = (1.0 - low_mask - high_mask).clamp(min=0.0)

        # Base 3-band equalizer kernel
        kernel = (
            self.low_freq_gain * low_mask +
            self.mid_freq_gain * mid_mask +
            self.high_freq_gain * high_mask
        )

        # Optional Spectral Tilt: tilts the spectrum slope (+tilt boosts high, -tilt boosts low)
        if abs(self.spectral_tilt) > 1e-4:
            tilt_slope = torch.pow(radius + 0.08, self.spectral_tilt * 0.8)
            kernel = kernel * tilt_slope

        # High-Shelf Anti-Aliasing Damping (rolls off extreme boundary frequencies near Nyquist)
        if self.high_shelf_damping > 1e-4:
            shelf_mask = torch.sigmoid((radius - 0.82) * 18.0)
            damping_curve = 1.0 - (self.high_shelf_damping * shelf_mask)
            kernel = kernel * damping_curve

        # Cache filter [1, 1, H, W_half]
        self._cached_filter = kernel.unsqueeze(0).unsqueeze(0).float()
        self._cached_hw = (H, W)
        self._cached_device = device

        return self._cached_filter

    @torch.no_grad()
    def step(
        self,
        cond: torch.Tensor,
        uncond: torch.Tensor,
        current_sigma: float,
        cond_scale: float = 1.0
    ) -> torch.Tensor:
        if uncond is None:
            return cond

        # Ensure batch dimensions match
        if cond.shape[0] != uncond.shape[0]:
            if uncond.shape[0] == 1:
                uncond = uncond.repeat(cond.shape[0], *([1] * (cond.ndim - 1)))
            elif cond.shape[0] == 1:
                cond = cond.repeat(uncond.shape[0], *([1] * (uncond.ndim - 1)))

        now = time.time()
        is_fresh_run = (
            (self.last_sigma is None) or
            (current_sigma > self.last_sigma + 1e-4) or
            (now - self.last_call_time > 180.0) or
            (self.prev_shape is not None and cond.shape != self.prev_shape)
        )
        if is_fresh_run:
            self.reset()

        self.last_call_time = now
        self.prev_shape = cond.shape

        if self.initial_sigma is None:
            self.initial_sigma = max(current_sigma, 1e-4)

        self.last_sigma = current_sigma
        self.step_count += 1

        # Calculate progress [0.0, 1.0]
        if self.schedule_mode == "logarithmic" and self.initial_sigma > 1e-4 and current_sigma > 1e-4:
            target_min_sigma = 0.0292
            curr_log = math.log(max(current_sigma, target_min_sigma))
            init_log = math.log(max(self.initial_sigma, target_min_sigma + 1e-4))
            min_log = math.log(target_min_sigma)
            progress = max(0.0, min(1.0, (init_log - curr_log) / (init_log - min_log))) if init_log > min_log else 0.0
        else:
            progress = max(0.0, min(1.0, 1.0 - (current_sigma / self.initial_sigma)))

        in_window = (self.start_percent <= progress <= self.end_percent)
        effective_cfg = self.cfg_scale if in_window else 1.0

        # Raw guidance score gradient vector
        delta = (cond - uncond).detach()

        if in_window:
            orig_shape = delta.shape
            is_5d = (delta.ndim == 5)

            # Flatten 5D video [B, C, F, H, W] into 2D batch [B*F, C, H, W]
            if is_5d:
                B, C, F_len, H, W = orig_shape
                d_2d = delta.permute(0, 2, 1, 3, 4).reshape(B * F_len, C, H, W).float()
            else:
                H, W = orig_shape[-2], orig_shape[-1]
                d_2d = delta.float()

            # 2D Real FFT into spatial frequency spectrum
            fft_coeffs = torch.fft.rfft2(d_2d, norm="ortho")
            filter_kernel = self._build_filter_kernel(H, W, delta.device)

            # Apply frequency band gains
            modulated_fft = fft_coeffs * filter_kernel
            filtered_2d = torch.fft.irfft2(modulated_fft, s=(H, W), norm="ortho")

            # Restore original tensor shape and dtype
            if is_5d:
                guided_delta = filtered_2d.view(B, F_len, C, H, W).permute(0, 2, 1, 3, 4).to(delta.dtype)
            else:
                guided_delta = filtered_2d.to(delta.dtype)
        else:
            guided_delta = delta

        # Synthesize guided latent prediction
        guided_prediction = uncond + effective_cfg * guided_delta

        if self.rescale_factor > 0.0 and in_window:
            cond_f = cond.float()
            guided_f = guided_prediction.float()
            norm_dims = (1, 3, 4) if cond.ndim == 5 else tuple(range(1, cond.ndim))

            if self.rescale_mode == "std":
                cond_measure = torch.std(cond_f, dim=norm_dims, keepdim=True) + 1e-8
                guided_measure = torch.std(guided_f, dim=norm_dims, keepdim=True) + 1e-8
            else:
                cond_measure = compute_vector_norm(cond_f, dim=norm_dims, keepdim=True) + 1e-8
                guided_measure = compute_vector_norm(guided_f, dim=norm_dims, keepdim=True) + 1e-8

            ratio = torch.clamp(cond_measure / guided_measure, min=0.05, max=20.0)
            rescaled_output = (guided_f * ratio).to(cond.dtype)
            guided_prediction = (
                self.rescale_factor * rescaled_output + 
                (1.0 - self.rescale_factor) * guided_prediction
            )

        if self.debug and (self.step_count % 5 == 0 or self.step_count == 1):
            print(f"[Spectral EQ] Step {self.step_count:02d} | σ={current_sigma:.3f} | CFG={effective_cfg:.2f}")

        return guided_prediction

def create_spectral_sampler_cfg_function(controller: SpectralGuidanceController):
    """
    Wraps the Spectral Guidance Controller into ComfyUI's model sampler_cfg_function signature.
    """
    def spectral_cfg_function(args: Dict[str, Any]) -> torch.Tensor:
        cond = args["cond"]
        uncond = args["uncond"]
        sigma = args["sigma"]
        cond_scale = float(args.get("cond_scale", 1.0))
        current_sigma = float(sigma.flatten()[0].item()) if isinstance(sigma, torch.Tensor) else float(sigma)

        return controller.step(
            cond=cond,
            uncond=uncond,
            current_sigma=current_sigma,
            cond_scale=cond_scale
        )

    return spectral_cfg_function

class SpectralGuidanceEQNode:
    """
    ComfyUI Node: Spectral Guidance (Fourier EQ).
    Decomposes the CFG guidance vector into spatial frequency bands (Low, Mid, High),
    allowing independent control over global lighting/mood, geometry, and micro-textures.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "cfg_scale": (
                    "FLOAT", 
                    {
                        "default": 4.5, 
                        "min": 0.0, 
                        "max": 30.0, 
                        "step": 0.1, 
                        "tooltip": "Primary guidance scale (CFG equivalent)."
                    }
                ),
                "low_freq_gain": (
                    "FLOAT", 
                    {
                        "default": 1.0, 
                        "min": 0.0, 
                        "max": 4.0, 
                        "step": 0.05, 
                        "tooltip": "[Bass/Macro] Gain for global composition, lighting contrast, and macro color palette."
                    }
                ),
                "mid_freq_gain": (
                    "FLOAT", 
                    {
                        "default": 1.0, 
                        "min": 0.0, 
                        "max": 4.0, 
                        "step": 0.05, 
                        "tooltip": "[Mids/Form] Gain for silhouettes, structural boundaries, and subject anatomy."
                    }
                ),
                "high_freq_gain": (
                    "FLOAT", 
                    {
                        "default": 1.0, 
                        "min": 0.0, 
                        "max": 4.0, 
                        "step": 0.05, 
                        "tooltip": "[Treble/Detail] Gain for micro-textures, skin pores, fabric weave, and fine edges."
                    }
                ),
            },
            "optional": {
                "crossover_low": (
                    "FLOAT", 
                    {
                        "default": 0.12, 
                        "min": 0.02, 
                        "max": 0.35, 
                        "step": 0.01, 
                        "tooltip": "Normalized frequency boundary separating Low and Mid bands."
                    }
                ),
                "crossover_high": (
                    "FLOAT", 
                    {
                        "default": 0.45, 
                        "min": 0.20, 
                        "max": 0.85, 
                        "step": 0.01, 
                        "tooltip": "Normalized frequency boundary separating Mid and High bands."
                    }
                ),
                "spectral_tilt": (
                    "FLOAT", 
                    {
                        "default": 0.0, 
                        "min": -1.5, 
                        "max": 1.5, 
                        "step": 0.05, 
                        "tooltip": "Global spectral slope (1/f^alpha): negative tilts warmer/softer, positive tilts sharper/crisper."
                    }
                ),
                "high_shelf_damping": (
                    "FLOAT", 
                    {
                        "default": 0.15, 
                        "min": 0.0, 
                        "max": 1.0, 
                        "step": 0.05, 
                        "tooltip": "Rolls off extreme boundary frequencies near Nyquist to eliminate digital checkerboard ringing."
                    }
                ),
                "rescale_factor": (
                    "FLOAT", 
                    {
                        "default": 0.70, 
                        "min": 0.0, 
                        "max": 1.0, 
                        "step": 0.05, 
                        "tooltip": "Dynamic range compression (0.0 = unconstrained, 1.0 = strict standard deviation match)."
                    }
                ),
                "rescale_mode": (["std", "norm"], {"default": "std"}),
                "schedule_mode": (["linear", "logarithmic"], {"default": "linear"}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "debug_logging": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "apply_spectral_guidance"
    CATEGORY = "sampling/guidance"

    def apply_spectral_guidance(
        self,
        model,
        cfg_scale: float,
        low_freq_gain: float = 1.0,
        mid_freq_gain: float = 1.0,
        high_freq_gain: float = 1.0,
        crossover_low: float = 0.12,
        crossover_high: float = 0.45,
        spectral_tilt: float = 0.0,
        high_shelf_damping: float = 0.15,
        rescale_factor: float = 0.70,
        rescale_mode: str = "std",
        schedule_mode: str = "linear",
        start_percent: float = 0.0,
        end_percent: float = 1.0,
        debug_logging: bool = False,
    ):
        model_patched = model.clone()

        controller = SpectralGuidanceController(
            cfg_scale=cfg_scale,
            low_freq_gain=low_freq_gain,
            mid_freq_gain=mid_freq_gain,
            high_freq_gain=high_freq_gain,
            crossover_low=crossover_low,
            crossover_high=crossover_high,
            spectral_tilt=spectral_tilt,
            high_shelf_damping=high_shelf_damping,
            rescale_factor=rescale_factor,
            rescale_mode=rescale_mode,
            schedule_mode=schedule_mode,
            start_percent=start_percent,
            end_percent=end_percent,
            debug=debug_logging,
        )

        model_patched.set_model_sampler_cfg_function(
            create_spectral_sampler_cfg_function(controller),
            disable_cfg1_optimization=True
        )

        return (model_patched,)

NODE_CLASS_MAPPINGS = {
    "SpectralGuidanceEQ": SpectralGuidanceEQNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpectralGuidanceEQ": "Spectral Guidance (Fourier EQ)"
}

# Registers the static web asset directory for ComfyUI frontend extensions
WEB_DIRECTORY = "./web"