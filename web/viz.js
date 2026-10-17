import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

/**
 * ComfyUI Frontend Extension: Live Fourier Spectrum Visualizer & Timeline Scrubber.
 * Features:
 * - Real-time step telemetry recording via WebSocket
 * - Interactive step scrubber to scrub back and forth through denoising history
 * - 1D Radial Power Spectral Density (PSD) plot with natural 1/f² reference slope
 * - Interactive transfer function / EQ curve overlay toggle
 * - 2D Spatial FFT Heatmap (radar/spatial frequency orientation)
 * - 3-Band Energy VU-Meter Gauges (Low / Mid / High)
 */
app.registerExtension({
  name: "SpectralGuidanceEQ.SpectrumMonitor",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (
      nodeData.name !== "SpectralGuidanceEQ" &&
      nodeData.name !== "SpectralSpectrumVisualizer"
    ) {
      return;
    }

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated
        ? onNodeCreated.apply(this, arguments)
        : undefined;
      const node = this;

      node.spectrumHistory = [];
      node.currentStepIdx = 0;
      node.activeView = "psd"; // "psd" | "heatmap" | "bands"
      node.showEqOverlay = true;
      node.isLive = true;

      const getVal = (name, fallback) => {
        const w = node.widgets?.find((w) => w.name === name);
        return w && typeof w.value === "number" ? w.value : fallback;
      };

      if (!window.__spectral_telemetry_attached) {
        window.__spectral_telemetry_attached = true;
        api.addEventListener("spectral_spectrum_step", (event) => {
          const data = event.detail;
          if (!data || !data.node_id) return;

          const targetNode = app.graph._nodes.find(
            (n) => String(n.id) === String(data.node_id),
          );
          if (targetNode && targetNode.onStepTelemetryReceived) {
            targetNode.onStepTelemetryReceived(data);
          }
        });
      }

      node.onStepTelemetryReceived = function (data) {
        // If a new sampling run starts, reset history
        if (
          data.step <= 1 ||
          (node.spectrumHistory.length > 0 &&
            data.step <
              node.spectrumHistory[node.spectrumHistory.length - 1].step)
        ) {
          node.spectrumHistory = [];
        }

        node.spectrumHistory.push(data);
        if (node.isLive) {
          node.currentStepIdx = node.spectrumHistory.length - 1;
        }
        node.setDirtyCanvas(true, true);
      };

      const monitorWidget = {
        type: "custom_canvas",
        name: "spectral_monitor_canvas",
        size: [300, 215],

        draw(ctx, parentNode, widgetWidth, yPos) {
          const margin = 8;
          const w = Math.max(260, widgetWidth - margin * 2);
          const h = 205;
          const x0 = margin;
          const y0 = yPos + 2;

          ctx.save();
          // Outer Background
          ctx.fillStyle = "#0e1015";
          ctx.beginPath();
          ctx.roundRect(x0, y0, w, h, 6);
          ctx.fill();

          ctx.strokeStyle = "#212530";
          ctx.lineWidth = 1;
          ctx.stroke();

          const headerH = 24;
          ctx.fillStyle = "#151821";
          ctx.beginPath();
          ctx.roundRect(x0, y0, w, headerH, [6, 6, 0, 0]);
          ctx.fill();

          // View Mode Tabs: [1D PSD] [2D FFT] [BANDS]
          const tabs = [
            { id: "psd", label: "1D PSD" },
            { id: "heatmap", label: "2D FFT" },
            { id: "bands", label: "BANDS" },
          ];

          let tabX = x0 + 6;
          for (const tab of tabs) {
            const isSel = node.activeView === tab.id;
            ctx.fillStyle = isSel ? "#3b82f6" : "#262b3a";
            ctx.beginPath();
            ctx.roundRect(tabX, y0 + 4, 46, 16, 3);
            ctx.fill();

            ctx.fillStyle = isSel ? "#ffffff" : "#94a3b8";
            ctx.font = "bold 8.5px sans-serif";
            ctx.fillText(tab.label, tabX + 6, y0 + 15);
            tabX += 50;
          }

          // EQ Overlay Pill Toggle
          const eqPillX = tabX + 4;
          ctx.fillStyle = node.showEqOverlay ? "#ea580c" : "#262b3a";
          ctx.beginPath();
          ctx.roundRect(eqPillX, y0 + 4, 38, 16, 3);
          ctx.fill();
          ctx.fillStyle = node.showEqOverlay ? "#ffffff" : "#64748b";
          ctx.font = "bold 8px sans-serif";
          ctx.fillText(
            node.showEqOverlay ? "EQ: ON" : "EQ: OFF",
            eqPillX + 4,
            y0 + 15,
          );

          // Live / Step status text
          const currentData =
            node.spectrumHistory && node.spectrumHistory.length > 0
              ? node.spectrumHistory[
                  Math.min(node.currentStepIdx, node.spectrumHistory.length - 1)
                ]
              : null;

          ctx.font = "9px monospace";
          ctx.fillStyle = node.isLive ? "#10b981" : "#f59e0b";
          const statusText = currentData
            ? `S:${currentData.step} σ:${currentData.sigma.toFixed(2)}`
            : "IDLE";
          ctx.fillText(statusText, x0 + w - 75, y0 + 15);

          const plotY = y0 + headerH + 4;
          const plotH = 135;
          const plotW = w - 12;
          const plotX = x0 + 6;

          // Plot Inner Box
          ctx.fillStyle = "#090a0e";
          ctx.beginPath();
          ctx.roundRect(plotX, plotY, plotW, plotH, 4);
          ctx.fill();
          ctx.strokeStyle = "#1b1e28";
          ctx.stroke();

          const crossLow = getVal(
            "crossover_low",
            currentData?.eq?.cross_low ?? 0.12,
          );
          const crossHigh = getVal(
            "crossover_high",
            currentData?.eq?.cross_high ?? 0.45,
          );

          // VIEW 1: 1D Radial Power Spectrum (PSD) Curve
          if (node.activeView === "psd") {
            // Band Region Shading
            const xCrossL = plotX + crossLow * plotW;
            const xCrossH = plotX + crossHigh * plotW;

            ctx.fillStyle = "rgba(99, 102, 241, 0.08)";
            ctx.fillRect(plotX, plotY, xCrossL - plotX, plotH);
            ctx.fillStyle = "rgba(245, 158, 11, 0.06)";
            ctx.fillRect(xCrossH, plotY, plotX + plotW - xCrossH, plotH);

            // Crossover Divider Lines
            ctx.strokeStyle = "rgba(99, 102, 241, 0.35)";
            ctx.beginPath();
            ctx.moveTo(xCrossL, plotY);
            ctx.lineTo(xCrossL, plotY + plotH);
            ctx.stroke();

            ctx.strokeStyle = "rgba(245, 158, 11, 0.35)";
            ctx.beginPath();
            ctx.moveTo(xCrossH, plotY);
            ctx.lineTo(xCrossH, plotY + plotH);
            ctx.stroke();

            // Natural 1/f² Reference Power Slope (Dashed Line)
            ctx.strokeStyle = "#2d3345";
            ctx.setLineDash([2, 3]);
            ctx.beginPath();
            ctx.moveTo(plotX + 4, plotY + 12);
            ctx.quadraticCurveTo(
              plotX + plotW * 0.4,
              plotY + plotH * 0.75,
              plotX + plotW - 4,
              plotY + plotH - 12,
            );
            ctx.stroke();
            ctx.setLineDash([]);

            // Draw Live Measured Latent Guidance Spectrum
            if (currentData && currentData.psd && currentData.psd.length > 0) {
              const pts = currentData.psd;
              const stepW = plotW / (pts.length - 1);

              ctx.beginPath();
              ctx.moveTo(plotX, plotY + plotH - pts[0] * (plotH - 16) - 8);
              for (let i = 1; i < pts.length; i++) {
                const px = plotX + i * stepW;
                const py = plotY + plotH - pts[i] * (plotH - 16) - 8;
                ctx.lineTo(px, py);
              }

              // Glowing Gradient Fill Under Spectrum
              const fillGrad = ctx.createLinearGradient(
                plotX,
                plotY,
                plotX + plotW,
                plotY,
              );
              fillGrad.addColorStop(0.0, "rgba(56, 189, 248, 0.28)");
              fillGrad.addColorStop(0.6, "rgba(129, 140, 248, 0.20)");
              fillGrad.addColorStop(1.0, "rgba(236, 72, 153, 0.15)");
              ctx.strokeStyle = "#38bdf8";
              ctx.lineWidth = 1.8;
              ctx.stroke();

              ctx.lineTo(plotX + plotW, plotY + plotH);
              ctx.lineTo(plotX, plotY + plotH);
              ctx.closePath();
              ctx.fillStyle = fillGrad;
              ctx.fill();
            } else {
              ctx.fillStyle = "#475569";
              ctx.font = "10px sans-serif";
              ctx.fillText(
                "Awaiting sampling step data...",
                plotX + plotW / 2 - 68,
                plotY + plotH / 2,
              );
            }

            // Overlay Parametric EQ Curve if enabled
            if (node.showEqOverlay) {
              const lowG = getVal(
                "low_freq_gain",
                currentData?.eq?.low_gain ?? 1.0,
              );
              const midG = getVal(
                "mid_freq_gain",
                currentData?.eq?.mid_gain ?? 1.0,
              );
              const highG = getVal(
                "high_freq_gain",
                currentData?.eq?.high_gain ?? 1.0,
              );
              const tilt = getVal(
                "spectral_tilt",
                currentData?.eq?.tilt ?? 0.0,
              );
              const damping = getVal(
                "high_shelf_damping",
                currentData?.eq?.damping ?? 0.15,
              );

              const maxG = Math.max(3.2, lowG, midG, highG);
              const eqPoints = [];
              const samples = 48;

              for (let s = 0; s <= samples; s++) {
                const rNorm = s / samples;
                const lowMask =
                  1.0 / (1.0 + Math.exp(-((crossLow - rNorm) * 16.0)));
                const highMask =
                  1.0 / (1.0 + Math.exp(-((rNorm - crossHigh) * 16.0)));
                const midMask = Math.max(0.0, 1.0 - lowMask - highMask);

                let gain = lowG * lowMask + midG * midMask + highG * highMask;
                if (Math.abs(tilt) > 1e-4) {
                  gain *= Math.pow(rNorm + 0.08, tilt * 0.8);
                }
                if (damping > 1e-4) {
                  const shelfMask =
                    1.0 / (1.0 + Math.exp(-((rNorm - 0.82) * 18.0)));
                  gain *= 1.0 - damping * shelfMask;
                }

                const px = plotX + rNorm * plotW;
                const py =
                  plotY +
                  plotH -
                  (Math.min(maxG, gain) / maxG) * (plotH - 24) -
                  12;
                eqPoints.push({ x: px, y: py });
              }

              ctx.strokeStyle = "#f97316";
              ctx.lineWidth = 1.5;
              ctx.setLineDash([3, 2]);
              ctx.beginPath();
              ctx.moveTo(eqPoints[0].x, eqPoints[0].y);
              for (let i = 1; i < eqPoints.length; i++) {
                ctx.lineTo(eqPoints[i].x, eqPoints[i].y);
              }
              ctx.stroke();
              ctx.setLineDash([]);
            }

            // Labels
            ctx.font = "8px sans-serif";
            ctx.fillStyle = "#818cf8";
            ctx.fillText("LOW (Macro)", plotX + 4, plotY + 11);
            ctx.fillStyle = "#38bdf8";
            ctx.fillText(
              "MID (Form)",
              (xCrossL + xCrossH) / 2 - 18,
              plotY + 11,
            );
            ctx.fillStyle = "#fbbf24";
            ctx.fillText("HIGH (Detail)", plotX + plotW - 48, plotY + 11);
          }

          // VIEW 2: 2D Spatial FFT Magnitude Heatmap
          else if (node.activeView === "heatmap") {
            if (
              currentData &&
              currentData.heatmap &&
              currentData.heatmap.length > 0
            ) {
              const grid = currentData.heatmap;
              const rows = grid.length;
              const cols = grid[0].length;
              const cellW = plotW / cols;
              const cellH = plotH / rows;

              for (let r = 0; r < rows; r++) {
                for (let c = 0; c < cols; c++) {
                  const val = grid[r][c]; // [0.0, 1.0]
                  // Cyberpunk Heatmap: dark violet -> blue -> cyan -> yellow
                  const red = Math.floor(Math.min(255, val * 320));
                  const green = Math.floor(
                    Math.min(255, Math.pow(val, 1.3) * 260),
                  );
                  const blue = Math.floor(
                    Math.min(255, (1.0 - val) * 120 + val * 240),
                  );
                  ctx.fillStyle = `rgb(${red},${green},${blue})`;
                  ctx.fillRect(
                    plotX + c * cellW,
                    plotY + r * cellH,
                    cellW + 0.4,
                    cellH + 0.4,
                  );
                }
              }

              // Concentric Frequency Radius Rings (Centered DC)
              const cx = plotX + plotW / 2;
              const cy = plotY + plotH / 2;
              const maxR = Math.min(plotW, plotH) / 2;

              ctx.strokeStyle = "rgba(255, 255, 255, 0.35)";
              ctx.setLineDash([2, 3]);
              ctx.lineWidth = 1;

              // Low crossover ring
              ctx.beginPath();
              ctx.arc(cx, cy, crossLow * maxR, 0, Math.PI * 2);
              ctx.stroke();

              // High crossover ring
              ctx.beginPath();
              ctx.arc(cx, cy, crossHigh * maxR, 0, Math.PI * 2);
              ctx.stroke();
              ctx.setLineDash([]);

              ctx.font = "8px monospace";
              ctx.fillStyle = "#ffffff";
              ctx.fillText("DC (Center)", cx - 24, cy - 3);
              ctx.fillText("Nyquist (Edges)", plotX + 4, plotY + plotH - 4);
            } else {
              ctx.fillStyle = "#475569";
              ctx.font = "10px sans-serif";
              ctx.fillText(
                "Awaiting 2D FFT telemetry...",
                plotX + plotW / 2 - 62,
                plotY + plotH / 2,
              );
            }
          }

          // VIEW 3: 3-Band Energy Distribution Gauges
          else if (node.activeView === "bands") {
            const bands = currentData?.bands || {
              low: 0.33,
              mid: 0.33,
              high: 0.34,
            };
            const barW = (plotW - 32) / 3;
            const meterH = plotH - 36;
            const meterY = plotY + 16;

            const items = [
              { name: "LOW / MACRO", ratio: bands.low, color: "#6366f1" },
              { name: "MID / ANATOMY", ratio: bands.mid, color: "#38bdf8" },
              { name: "HIGH / TEXTURE", ratio: bands.high, color: "#f59e0b" },
            ];

            items.forEach((item, idx) => {
              const bx = plotX + 8 + idx * (barW + 8);
              // Bar background slot
              ctx.fillStyle = "#161922";
              ctx.beginPath();
              ctx.roundRect(bx, meterY, barW, meterH, 4);
              ctx.fill();

              // Active Fill
              const fillH = Math.max(4, item.ratio * meterH);
              const fillY = meterY + meterH - fillH;
              ctx.fillStyle = item.color;
              ctx.beginPath();
              ctx.roundRect(bx, fillY, barW, fillH, 4);
              ctx.fill();

              // Percentage Label
              ctx.font = "bold 9px monospace";
              ctx.fillStyle = "#ffffff";
              const pctText = `${Math.round(item.ratio * 100)}%`;
              ctx.fillText(
                pctText,
                bx + barW / 2 - 10,
                fillY - 4 > meterY + 10 ? fillY - 4 : meterY + 14,
              );

              ctx.font = "7.5px sans-serif";
              ctx.fillStyle = "#94a3b8";
              ctx.fillText(item.name, bx + 2, meterY + meterH + 12);
            });
          }

          const scrubY = y0 + h - 30;
          const scrubH = 22;
          const scrubW = w - 12;
          const scrubX = x0 + 6;

          ctx.fillStyle = "#12151d";
          ctx.beginPath();
          ctx.roundRect(scrubX, scrubY, scrubW, scrubH, 4);
          ctx.fill();
          ctx.strokeStyle = "#252b3b";
          ctx.stroke();

          // Step Prev / Next Buttons
          ctx.fillStyle = "#262c3e";
          ctx.beginPath();
          ctx.roundRect(scrubX + 3, scrubY + 3, 16, 16, 3);
          ctx.roundRect(scrubX + 22, scrubY + 3, 16, 16, 3);
          ctx.fill();

          ctx.fillStyle = "#94a3b8";
          ctx.font = "bold 9px monospace";
          ctx.fillText("◀", scrubX + 7, scrubY + 14);
          ctx.fillText("▶", scrubX + 26, scrubY + 14);

          // Scrubber Track
          const trackX = scrubX + 46;
          const trackW = scrubW - 90;
          const trackY = scrubY + 9;

          ctx.fillStyle = "#232838";
          ctx.beginPath();
          ctx.roundRect(trackX, trackY, trackW, 4, 2);
          ctx.fill();

          const totalHistory = node.spectrumHistory.length;
          const activeIndex = Math.min(
            node.currentStepIdx,
            Math.max(0, totalHistory - 1),
          );

          if (totalHistory > 1) {
            const progressRatio = activeIndex / (totalHistory - 1);
            const thumbX = trackX + progressRatio * trackW;

            // Filled Track Progress
            ctx.fillStyle = "#38bdf8";
            ctx.beginPath();
            ctx.roundRect(trackX, trackY, progressRatio * trackW, 4, 2);
            ctx.fill();

            // Thumb Playhead
            ctx.fillStyle = "#ffffff";
            ctx.beginPath();
            ctx.arc(thumbX, trackY + 2, 5, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = "#38bdf8";
            ctx.lineWidth = 1.5;
            ctx.stroke();
          }

          // Scrubber Step Badge
          ctx.font = "8.5px monospace";
          ctx.fillStyle = node.isLive ? "#10b981" : "#f59e0b";
          const scrubLabel =
            totalHistory > 0 ? `${activeIndex + 1}/${totalHistory}` : "0/0";
          ctx.fillText(scrubLabel, scrubX + scrubW - 38, scrubY + 14);

          ctx.restore();
        },

        computeSize() {
          return [300, 225];
        },
      };

      node.onMouseDown = function (e, localPos) {
        const margin = 8;
        const lx = localPos[0];
        const ly = localPos[1];
        const yPos = 30; // widget vertical offset

        // Header Tab Hit Tests
        const tabY = yPos + 4;
        if (ly >= tabY && ly <= tabY + 20) {
          if (lx >= margin + 6 && lx <= margin + 52) {
            node.activeView = "psd";
            node.setDirtyCanvas(true, true);
            return true;
          }
          if (lx >= margin + 56 && lx <= margin + 102) {
            node.activeView = "heatmap";
            node.setDirtyCanvas(true, true);
            return true;
          }
          if (lx >= margin + 106 && lx <= margin + 152) {
            node.activeView = "bands";
            node.setDirtyCanvas(true, true);
            return true;
          }
          if (lx >= margin + 156 && lx <= margin + 198) {
            node.showEqOverlay = !node.showEqOverlay;
            node.setDirtyCanvas(true, true);
            return true;
          }
        }

        // Scrubber Hit Tests
        const h = 205;
        const scrubY = yPos + h - 30;
        if (ly >= scrubY && ly <= scrubY + 24) {
          const scrubX = margin + 6;
          // Prev Button
          if (lx >= scrubX + 3 && lx <= scrubX + 19) {
            node.isLive = false;
            node.currentStepIdx = Math.max(0, node.currentStepIdx - 1);
            node.setDirtyCanvas(true, true);
            return true;
          }
          // Next Button
          if (lx >= scrubX + 22 && lx <= scrubX + 38) {
            node.currentStepIdx = Math.min(
              node.spectrumHistory.length - 1,
              node.currentStepIdx + 1,
            );
            if (node.currentStepIdx === node.spectrumHistory.length - 1)
              node.isLive = true;
            node.setDirtyCanvas(true, true);
            return true;
          }

          // Scrubber Track Drag
          const trackX = scrubX + 46;
          const trackW = node.size[0] - margin * 2 - 12 - 90;
          if (
            lx >= trackX &&
            lx <= trackX + trackW &&
            node.spectrumHistory.length > 0
          ) {
            const ratio = Math.max(0, Math.min(1.0, (lx - trackX) / trackW));
            node.currentStepIdx = Math.round(
              ratio * (node.spectrumHistory.length - 1),
            );
            node.isLive =
              node.currentStepIdx === node.spectrumHistory.length - 1;
            node.setDirtyCanvas(true, true);
            return true;
          }
        }

        return false;
      };

      node.addCustomWidget(monitorWidget);
      node.setSize([310, Math.max(node.size[1], 480)]);
      return r;
    };
  },
});
