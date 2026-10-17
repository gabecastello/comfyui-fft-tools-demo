import math
import time
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List

# Attempt to import ComfyUI's internal PromptServer for real-time WebSocket telemetry
try:
    from server import PromptServer
except ImportError:
    PromptServer = None

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
    Decomposes guidance into spatial frequency bands with real-time FFT telemetry.
    """
    def __init__(
        self,
        node_id: str = "",
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
        send_telemetry: bool = True,
        debug: bool = False,
    ):
        self.node_id = str(node_id or "")
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
        self.send_telemetry = send_telemetry
        self.debug = debug

        self.initial_sigma: Optional[float] = None
        self.last_sigma: Optional[float] = None
        self.last_call_time: float = 0.0
        self.prev_shape: Optional[Tuple[int, ...]] = None
        self.step_count: int = 0
        
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
        if (
            self._cached_filter is not None and 
            self._cached_hw == (H, W) and 
            self._cached_device == device
        ):
            return self._cached_filter

        fy = torch.fft.fftfreq(H, device=device).unsqueeze(1)
        fx = torch.fft.rfftfreq(W, device=device).unsqueeze(0)
        radius = torch.sqrt((2.0 * fy) ** 2 + (2.0 * fx) ** 2).clamp(0.0, 1.0)

        low_mask = torch.sigmoid((self.crossover_low - radius) * 16.0)
        high_mask = torch.sigmoid((radius - self.crossover_high) * 16.0)
        mid_mask = (1.0 - low_mask - high_mask).clamp(min=0.0)

        kernel = (
            self.low_freq_gain * low_mask +
            self.mid_freq_gain * mid_mask +
            self.high_freq_gain * high_mask
        )

        if abs(self.spectral_tilt) > 1e-4:
            tilt_slope = torch.pow(radius + 0.08, self.spectral_tilt * 0.8)
            kernel = kernel * tilt_slope

        if self.high_shelf_damping > 1e-4:
            shelf_mask = torch.sigmoid((radius - 0.82) * 18.0)
            damping_curve = 1.0 - (self.high_shelf_damping * shelf_mask)
            kernel = kernel * damping_curve

        self._cached_filter = kernel.unsqueeze(0).unsqueeze(0).float()
        self._cached_hw = (H, W)
        self._cached_device = device

        return self._cached_filter

    def _broadcast_spectrum_data(
        self,
        d_2d: torch.Tensor,
        current_sigma: float,
        progress: float
    ):
        if PromptServer is None or PromptServer.instance is None or not self.send_telemetry:
            return

        try:
            H, W = d_2d.shape[-2], d_2d.shape[-1]
            
            # 1. 2D Real FFT for 1D Radial PSD
            rfft_coeffs = torch.fft.rfft2(d_2d, norm="ortho")
            mag_spatial = torch.mean(torch.abs(rfft_coeffs), dim=(0, 1))

            fy = torch.fft.fftfreq(H, device=d_2d.device).unsqueeze(1)
            fx = torch.fft.rfftfreq(W, device=d_2d.device).unsqueeze(0)
            radius = torch.sqrt((2.0 * fy) ** 2 + (2.0 * fx) ** 2).clamp(0.0, 1.0)

            num_bins = 64
            bin_idx = (radius * (num_bins - 1)).long().view(-1)
            mag_flat = mag_spatial.contiguous().view(-1)
            
            radial_accum = torch.zeros(num_bins, device=d_2d.device)
            bin_counts = torch.zeros(num_bins, device=d_2d.device)
            radial_accum.scatter_add_(0, bin_idx, mag_flat)
            bin_counts.scatter_add_(0, bin_idx, torch.ones_like(mag_flat))
            
            psd_1d = (radial_accum / (bin_counts + 1e-8)).clamp(min=1e-7)
            log_psd = torch.log10(psd_1d + 1e-5)
            min_p, max_p = log_psd.min(), log_psd.max()
            norm_psd = ((log_psd - min_p) / (max_p - min_p + 1e-6)).clamp(0.0, 1.0)
            psd_list = [round(float(v), 4) for v in norm_psd.cpu().tolist()]

            # 2. Centered 2D Spatial FFT Heatmap
            fft2_full = torch.fft.fft2(d_2d, norm="ortho")
            mag_centered = torch.fft.fftshift(torch.abs(fft2_full), dim=(-2, -1))
            mag_2d_mean = torch.mean(mag_centered, dim=(0, 1), keepdim=True)
            downsampled = F.adaptive_avg_pool2d(mag_2d_mean, (32, 32)).squeeze().log1p()
            d_min, d_max = downsampled.min(), downsampled.max()
            norm_2d = ((downsampled - d_min) / (d_max - d_min + 1e-6)).clamp(0.0, 1.0)
            heatmap_list = [[round(float(val), 3) for val in row] for row in norm_2d.cpu().tolist()]

            # 3. Band Energy Allocation
            low_e = float(mag_spatial[radius < self.crossover_low].sum().item())
            mid_e = float(mag_spatial[(radius >= self.crossover_low) & (radius <= self.crossover_high)].sum().item())
            high_e = float(mag_spatial[radius > self.crossover_high].sum().item())
            total_e = low_e + mid_e + high_e + 1e-8

            payload = {
                "node_id": str(self.node_id or ""),
                "step": self.step_count,
                "sigma": round(float(current_sigma), 4),
                "progress": round(float(progress), 3),
                "psd": psd_list,
                "heatmap": heatmap_list,
                "bands": {
                    "low": round(low_e / total_e, 4),
                    "mid": round(mid_e / total_e, 4),
                    "high": round(high_e / total_e, 4)
                },
                "eq": {
                    "low_gain": self.low_freq_gain,
                    "mid_gain": self.mid_freq_gain,
                    "high_gain": self.high_freq_gain,
                    "cross_low": self.crossover_low,
                    "cross_high": self.crossover_high,
                    "tilt": self.spectral_tilt,
                    "damping": self.high_shelf_damping,
                }
            }
            PromptServer.instance.send_sync("spectral_spectrum_step", payload)
        except Exception as e:
            if self.debug:
                print(f"[Spectral EQ] Telemetry broadcast error: {e}")

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

        delta = (cond - uncond).detach()

        orig_shape = delta.shape
        is_5d = (delta.ndim == 5)
        if is_5d:
            B, C, F_len, H, W = orig_shape
            d_2d = delta.permute(0, 2, 1, 3, 4).reshape(B * F_len, C, H, W).float()
        else:
            H, W = orig_shape[-2], orig_shape[-1]
            d_2d = delta.float()

        self._broadcast_spectrum_data(d_2d, current_sigma, progress)

        if in_window:
            fft_coeffs = torch.fft.rfft2(d_2d, norm="ortho")
            filter_kernel = self._build_filter_kernel(H, W, delta.device)

            modulated_fft = fft_coeffs * filter_kernel
            filtered_2d = torch.fft.irfft2(modulated_fft, s=(H, W), norm="ortho")

            if is_5d:
                guided_delta = filtered_2d.view(B, F_len, C, H, W).permute(0, 2, 1, 3, 4).to(delta.dtype)
            else:
                guided_delta = filtered_2d.to(delta.dtype)
        else:
            guided_delta = delta

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

        return guided_prediction

def create_spectral_sampler_cfg_function(controller: SpectralGuidanceController):
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
    Equalizes latent guidance across spatial frequency bands.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "cfg_scale": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 30.0, "step": 0.1}),
                "low_freq_gain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "mid_freq_gain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "high_freq_gain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
            },
            "optional": {
                "crossover_low": ("FLOAT", {"default": 0.12, "min": 0.02, "max": 0.35, "step": 0.01}),
                "crossover_high": ("FLOAT", {"default": 0.45, "min": 0.20, "max": 0.85, "step": 0.01}),
                "spectral_tilt": ("FLOAT", {"default": 0.0, "min": -1.5, "max": 1.5, "step": 0.05}),
                "high_shelf_damping": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05}),
                "rescale_factor": ("FLOAT", {"default": 0.70, "min": 0.0, "max": 1.0, "step": 0.05}),
                "rescale_mode": (["std", "norm"], {"default": "std"}),
                "schedule_mode": (["linear", "logarithmic"], {"default": "linear"}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "debug_logging": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
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
        unique_id: str = "",
        **kwargs
    ):
        model_patched = model.clone()

        # Telemetry is handled by SpectralSpectrumVisualizerNode to avoid duplicate broadcasts
        controller = SpectralGuidanceController(
            node_id=str(unique_id or ""),
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
            send_telemetry=False,
            debug=debug_logging,
        )

        model_patched.set_model_sampler_cfg_function(
            create_spectral_sampler_cfg_function(controller),
            disable_cfg1_optimization=True
        )

        return (model_patched,)

class SpectralSpectrumVisualizerNode:
    """
    ComfyUI Node: Spectral Spectrum Monitor & Scrubber.
    A non-destructive oscilloscope tap that streams real-time frequency spectra to the UI
    without overriding upstream guidance or sampler behavior.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "target_signal": (
                    ["guidance_delta (cond - uncond)", "conditioned (cond)", "unconditioned (uncond)"],
                    {"default": "guidance_delta (cond - uncond)"}
                ),
            },
            "optional": {
                "crossover_low": ("FLOAT", {"default": 0.12, "min": 0.02, "max": 0.35, "step": 0.01}),
                "crossover_high": ("FLOAT", {"default": 0.45, "min": 0.20, "max": 0.85, "step": 0.01}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "attach_spectrum_monitor"
    CATEGORY = "sampling/guidance"

    def attach_spectrum_monitor(
        self,
        model,
        target_signal: str = "guidance_delta (cond - uncond)",
        crossover_low: float = 0.12,
        crossover_high: float = 0.45,
        unique_id: str = "",
        **kwargs
    ):
        model_patched = model.clone()

        # Preserve any upstream sampler_cfg_function (e.g. from SpectralGuidanceEQ)
        existing_cfg_function = None
        if hasattr(model_patched, "model_options"):
            existing_cfg_function = model_patched.model_options.get("sampler_cfg_function", None)

        state = {
            "step": 0,
            "last_sigma": None,
            "initial_sigma": None,
            "last_time": 0.0
        }

        def spectrum_monitor_hook(args: Dict[str, Any]) -> torch.Tensor:
            cond = args["cond"]
            uncond = args["uncond"]
            sigma = args["sigma"]
            cond_scale = float(args.get("cond_scale", 1.0))
            current_sigma = float(sigma.flatten()[0].item()) if isinstance(sigma, torch.Tensor) else float(sigma)

            now = time.time()
            if (
                state["last_sigma"] is None or
                current_sigma > state["last_sigma"] + 1e-4 or
                now - state["last_time"] > 180.0
            ):
                state["step"] = 0
                state["initial_sigma"] = max(current_sigma, 1e-4)

            state["last_sigma"] = current_sigma
            state["last_time"] = now
            state["step"] += 1

            if "unconditioned" in target_signal:
                sig = uncond
            elif "conditioned" in target_signal:
                sig = cond
            else:
                sig = cond - uncond if uncond is not None else cond

            if sig is not None and PromptServer is not None and PromptServer.instance is not None:
                try:
                    s_float = sig.detach().float()
                    if s_float.ndim == 5:
                        B, C, F_len, H, W = s_float.shape
                        d_2d = s_float.permute(0, 2, 1, 3, 4).reshape(B * F_len, C, H, W)
                    else:
                        H, W = s_float.shape[-2], s_float.shape[-1]
                        d_2d = s_float

                    rfft_c = torch.fft.rfft2(d_2d, norm="ortho")
                    mag_sp = torch.mean(torch.abs(rfft_c), dim=(0, 1))

                    fy = torch.fft.fftfreq(H, device=d_2d.device).unsqueeze(1)
                    fx = torch.fft.rfftfreq(W, device=d_2d.device).unsqueeze(0)
                    radius = torch.sqrt((2.0 * fy) ** 2 + (2.0 * fx) ** 2).clamp(0.0, 1.0)

                    num_bins = 64
                    bin_idx = (radius * (num_bins - 1)).long().view(-1)
                    mag_flat = mag_sp.contiguous().view(-1)
                    
                    radial_accum = torch.zeros(num_bins, device=d_2d.device)
                    bin_counts = torch.zeros(num_bins, device=d_2d.device)
                    radial_accum.scatter_add_(0, bin_idx, mag_flat)
                    bin_counts.scatter_add_(0, bin_idx, torch.ones_like(mag_flat))
                    
                    psd_1d = (radial_accum / (bin_counts + 1e-8)).clamp(min=1e-7)
                    log_psd = torch.log10(psd_1d + 1e-5)
                    min_p, max_p = log_psd.min(), log_psd.max()
                    norm_psd = ((log_psd - min_p) / (max_p - min_p + 1e-6)).clamp(0.0, 1.0)
                    psd_list = [round(float(v), 4) for v in norm_psd.cpu().tolist()]

                    fft2_full = torch.fft.fft2(d_2d, norm="ortho")
                    mag_centered = torch.fft.fftshift(torch.abs(fft2_full), dim=(-2, -1))
                    mag_2d_mean = torch.mean(mag_centered, dim=(0, 1), keepdim=True)
                    downsampled = F.adaptive_avg_pool2d(mag_2d_mean, (32, 32)).squeeze().log1p()
                    d_min, d_max = downsampled.min(), downsampled.max()
                    norm_2d = ((downsampled - d_min) / (d_max - d_min + 1e-6)).clamp(0.0, 1.0)
                    heatmap_list = [[round(float(val), 3) for val in row] for row in norm_2d.cpu().tolist()]

                    low_e = float(mag_sp[radius < crossover_low].sum().item())
                    mid_e = float(mag_sp[(radius >= crossover_low) & (radius <= crossover_high)].sum().item())
                    high_e = float(mag_sp[radius > crossover_high].sum().item())
                    tot = low_e + mid_e + high_e + 1e-8

                    progress = max(0.0, min(1.0, 1.0 - (current_sigma / (state["initial_sigma"] or 1.0))))

                    payload = {
                        "node_id": str(unique_id or ""),
                        "step": state["step"],
                        "sigma": round(float(current_sigma), 4),
                        "progress": round(float(progress), 3),
                        "psd": psd_list,
                        "heatmap": heatmap_list,
                        "bands": {
                            "low": round(low_e / tot, 4),
                            "mid": round(mid_e / tot, 4),
                            "high": round(high_e / tot, 4)
                        },
                        "eq": {
                            "low_gain": 1.0,
                            "mid_gain": 1.0,
                            "high_gain": 1.0,
                            "cross_low": crossover_low,
                            "cross_high": crossover_high,
                            "tilt": 0.0,
                            "damping": 0.0,
                        }
                    }
                    PromptServer.instance.send_sync("spectral_spectrum_step", payload)
                except Exception:
                    pass

            # Chain execution: if an upstream custom guidance function was set, delegate to it
            if existing_cfg_function is not None:
                return existing_cfg_function(args)

            if uncond is None:
                return cond
            return uncond + cond_scale * (cond - uncond)

        model_patched.set_model_sampler_cfg_function(
            spectrum_monitor_hook,
            disable_cfg1_optimization=True
        )

        return (model_patched,)

NODE_CLASS_MAPPINGS = {
    "SpectralGuidanceEQ": SpectralGuidanceEQNode,
    "SpectralSpectrumVisualizer": SpectralSpectrumVisualizerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpectralGuidanceEQ": "Spectral Guidance (Fourier EQ)",
    "SpectralSpectrumVisualizer": "Spectral Spectrum Monitor & Scrubber",
}

WEB_DIRECTORY = "./web"
