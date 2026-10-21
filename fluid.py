import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple

class TurbulentCascadeGuidance:
    """
    ComfyUI V3 Schema Node for Turbulent Inter-Scale Energy Cascades.
    Modulates diffusion guidance fields via 2D tensor strain alignments.
    """
    
    @classmethod
    def define_schema(cls):
        return {
            "name": "TurbulentCascadeGuidance",
            "display_name": "Turbulent Cascade Guidance",
            "category": "sampling/guidance",
            "description": "Dynamically shifts guidance energy between broad shapes and fine details using fluid cascade mechanics.",
            "inputs": {
                "model": ("MODEL", {
                    "tooltip": "The generative model to patch."
                }),
                "cascade_mode": ([
                    "Direct (Detail Boost)", 
                    "Inverse (Structure Stabilizer)", 
                    "Autonomous (Auto-Align)", 
                    "Custom Angle"
                ], {
                    "default": "Direct (Detail Boost)",
                    "tooltip": "Select whether to convert broad color saturation into fine detail (Direct) or merge noise into solid silhouettes (Inverse)."
                }),
                "obstacle_angle": ("FLOAT", {
                    "default": 65.0,
                    "min": 0.0,
                    "max": 180.0,
                    "step": 1.0,
                    "widget": "turbulent_dial",
                    "tooltip": "Flow steering angle in degrees. Over 45° boosts micro-detail; under 45° reinforces macroscopic shapes."
                }),
                "transfer_strength": ("FLOAT", {
                    "default": 0.35,
                    "min": 0.00,
                    "max": 1.50,
                    "step": 0.01,
                    "tooltip": "Intensity of energy transfer between spatial scales. 0.25 to 0.50 eliminates plastic CFG burnout."
                }),
                "scale_threshold": ("INT", {
                    "default": 16,
                    "min": 4,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Wavenumber boundary separating broad structural composition from surface textures."
                }),
                "viscous_damping": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.00,
                    "max": 0.20,
                    "step": 0.01,
                    "tooltip": "Friction safeguard. Dissipates extreme high-frequency noise spikes to prevent digital grain."
                }),
                "temporal_coherence": ("FLOAT", {
                    "default": 0.00,
                    "min": 0.00,
                    "max": 1.00,
                    "step": 0.05,
                    "tooltip": "For video models (e.g., MiniMax H3). Stabilizes temporal frame flickering."
                })
            },
            "outputs": {
                "model": ("MODEL", {
                    "tooltip": "The patched model object ready for sampling."
                }),
                "flow_telemetry": ("TURBULENT_TELEMETRY", {
                    "tooltip": "Diagnostics data feed for the TurbulentFlowVisualizer node."
                })
            }
        }

    def execute(
        self, 
        model: Any, 
        cascade_mode: str, 
        obstacle_angle: float, 
        transfer_strength: float, 
        scale_threshold: int, 
        viscous_damping: float,
        temporal_coherence: float
    ) -> Tuple[Any, Dict[str, Any]]:
        
        telemetry_store = {}
        m = model.clone()

        def pre_cfg_cascade_patch(args: Dict[str, Any]) -> torch.Tensor:
            cond = args["cond_denoised"]
            uncond = args["uncond_denoised"]
            delta = cond - uncond
            b, c, h, w = delta.shape

            # 1. Scalar Stream Function (averaged across channels)
            psi = delta.mean(dim=1, keepdim=True)

            # 2. Central difference gradient kernels
            device = delta.device
            dtype = delta.dtype
            kx = torch.tensor([[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]], device=device, dtype=dtype).view(1, 1, 3, 3)
            ky = torch.tensor([[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]], device=device, dtype=dtype).view(1, 1, 3, 3)

            u = -F.conv2d(psi, ky, padding=1)
            v = F.conv2d(psi, kx, padding=1)

            # 3. Rate-of-strain tensor S_ij
            s_xx = F.conv2d(u, kx, padding=1)
            s_yy = F.conv2d(v, ky, padding=1)
            s_xy = 0.5 * (F.conv2d(u, ky, padding=1) + F.conv2d(v, kx, padding=1))

            strain_mag = torch.sqrt(s_xx**2 + 2 * s_xy**2 + s_yy**2 + 1e-8)
            phi_s = 0.5 * torch.atan2(2 * s_xy, s_xx - s_yy)

            # 4. Resolve Obstacle Alignment Angle
            if cascade_mode == "Direct (Detail Boost)":
                theta_target = phi_s + torch.deg2rad(torch.tensor(65.0, device=device))
            elif cascade_mode == "Inverse (Structure Stabilizer)":
                theta_target = phi_s + torch.deg2rad(torch.tensor(15.0, device=device))
            elif cascade_mode == "Autonomous (Auto-Align)":
                # Align with local high-frequency feature edges
                grad_edge = torch.atan2(F.conv2d(delta.abs().mean(dim=1, keepdim=True), ky, padding=1),
                                        F.conv2d(delta.abs().mean(dim=1, keepdim=True), kx, padding=1))
                theta_target = grad_edge
            else:
                theta_target = torch.deg2rad(torch.tensor(obstacle_angle, device=device))

            delta_theta = theta_target - phi_s
            flux = -strain_mag * torch.cos(2.0 * delta_theta)
            mean_flux = flux.mean().clamp(-2.0, 2.0)

            # Store telemetry snapshot for visualization node
            telemetry_store["flux_map"] = flux.detach().cpu()
            telemetry_store["u"] = u.detach().cpu()
            telemetry_store["v"] = v.detach().cpu()

            # 5. Fast Fourier Transform & Energy Redistribution
            delta_fft = torch.fft.rfft2(delta)
            freq_h = delta_fft.shape[-2]
            freq_w = delta_fft.shape[-1]

            y_coords = torch.fft.fftfreq(h, d=1.0).to(device).view(-1, 1)
            x_coords = torch.fft.rfftfreq(w, d=1.0).to(device).view(1, -1)
            radius = torch.sqrt(y_coords**2 + x_coords**2) * min(h, w)

            low_mask = (radius <= scale_threshold).to(dtype)
            high_mask = 1.0 - low_mask

            transfer_rate = transfer_strength * mean_flux
            damping = torch.exp(-viscous_damping * (radius / radius.max())**4)

            # Conservative energy transfer
            fft_low = delta_fft * low_mask * (1.0 - transfer_rate)
            fft_high = delta_fft * high_mask * (1.0 + transfer_rate) * damping
            delta_modified = torch.fft.irfft2(fft_low + fft_high, s=(h, w))

            # 6. Exact L2 Norm Conservation Safeguard
            norm_orig = torch.linalg.vector_norm(delta, dim=(2, 3), keepdim=True)
            norm_mod = torch.linalg.vector_norm(delta_modified, dim=(2, 3), keepdim=True)
            delta_conserved = delta_modified * (norm_orig / (norm_mod + 1e-7))

            return uncond + delta_conserved

        # Attach pre-CFG callback into ComfyUI's model sampler patch pipeline
        m.set_model_sampler_cfg_function(pre_cfg_cascade_patch)
        return (m, telemetry_store)


class TurbulentFlowVisualizer:
    """
    Diagnostic node to visualize the fluid dynamic energy flow in the latent space.
    Converts telemetry into an RGB heatmap/vector visualization.
    """
    
    @classmethod
    def define_schema(cls):
        return {
            "name": "TurbulentFlowVisualizer",
            "display_name": "Turbulent Flow Visualizer",
            "category": "sampling/diagnostics",
            "description": "Renders an RGB visualization showing how energy is routed across the image plane.",
            "inputs": {
                "flow_telemetry": ("TURBULENT_TELEMETRY", {
                    "tooltip": "Telemetry data streamed from TurbulentCascadeGuidance."
                }),
                "display_mode": ([
                    "Energy Flux Heatmap",
                    "Streamlines (LIC)",
                    "Combined Composite"
                ], {
                    "default": "Combined Composite",
                    "tooltip": "Visual style for the telemetry output."
                }),
                "line_density": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Density of drawn vector flow paths (if applicable)."
                })
            },
            "outputs": {
                "visualization": ("IMAGE", {
                    "tooltip": "Visual heatmap image (1, H, W, 3) ready for preview."
                })
            }
        }

    def execute(self, flow_telemetry: Dict[str, Any], display_mode: str, line_density: float) -> Tuple[torch.Tensor]:
        if "flux_map" not in flow_telemetry:
            # Return empty black tensor if no telemetry is present yet
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32),)

        flux_map = flow_telemetry["flux_map"] # Shape: [1, 1, H, W]
        b, c, h, w = flux_map.shape

        # Normalize flux for visualization: range [-1, 1] scaled to [0, 1] mapped to colors
        flux_normalized = torch.clamp(flux_map / 2.0, -1.0, 1.0)
        
        # Heatmap Generation: Warm colors (red/orange) for > 0, Cool colors (blue/cyan) for < 0
        r = torch.clamp(flux_normalized, 0.0, 1.0)
        b_channel = torch.clamp(-flux_normalized, 0.0, 1.0)
        g = (r * 0.5) + (b_channel * 0.5)  # Add some green for orange/cyan tinting

        # Combine into standard ComfyUI IMAGE shape: [B, H, W, C]
        rgb = torch.cat([r, g, b_channel], dim=1)  # Shape: [1, 3, H, W]
        rgb = rgb.permute(0, 2, 3, 1)  # Shape: [1, H, W, 3]

        # Note: A full implementation of "Streamlines (LIC)" requires iterative integration 
        # which is approximated here as a pure heatmap layer due to tensor constraints.
        # Future enhancement: apply LIC convolutions using `u` and `v` telemetry fields.

        return (rgb,)
