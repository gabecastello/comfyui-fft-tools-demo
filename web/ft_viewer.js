import { app } from "../../../scripts/app.js";

/**
 * ComfyUI Frontend Extension: Spectral Guidance (Fourier EQ) Live Transfer Function Plotter.
 * Renders a real-time parametric frequency response curve inside the node body on canvas.
 */
app.registerExtension({
  name: "SpectralGuidanceEQ.Visualizer",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "SpectralGuidanceEQ") {
      return;
    }

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated
        ? onNodeCreated.apply(this, arguments)
        : undefined;
      const node = this;

      // Helper to safely read current numeric widget value
      const getVal = (name, fallback) => {
        const w = node.widgets?.find((w) => w.name === name);
        return w && typeof w.value === "number" ? w.value : fallback;
      };

      const eqWidget = {
        type: "custom_canvas",
        name: "spectral_eq_graph",
        size: [280, 115],

        draw(ctx, parentNode, widgetWidth, yPos) {
          const margin = 10;
          const w = Math.max(220, widgetWidth - margin * 2);
          const h = 95;
          const x0 = margin;
          const y0 = yPos + 4;

          // Pull current slider settings live from the node
          const lowGain = getVal("low_freq_gain", 1.0);
          const midGain = getVal("mid_freq_gain", 1.0);
          const highGain = getVal("high_freq_gain", 1.0);
          const crossLow = getVal("crossover_low", 0.12);
          const crossHigh = getVal("crossover_high", 0.45);
          const tilt = getVal("spectral_tilt", 0.0);
          const damping = getVal("high_shelf_damping", 0.15);

          // Background
          ctx.save();
          ctx.fillStyle = "#121318";
          ctx.beginPath();
          ctx.roundRect(x0, y0, w, h, 6);
          ctx.fill();

          // Subtle border
          ctx.strokeStyle = "#252733";
          ctx.lineWidth = 1;
          ctx.stroke();

          // Max display gain (vertical axis scale)
          const maxGain = Math.max(3.2, lowGain, midGain, highGain);

          // Neutral 1.0x gain unity baseline
          const unityY = y0 + h - (1.0 / maxGain) * (h - 18) - 10;
          ctx.strokeStyle = "#2e3346";
          ctx.setLineDash([3, 3]);
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(x0, unityY);
          ctx.lineTo(x0 + w, unityY);
          ctx.stroke();
          ctx.setLineDash([]);

          const xCrossL = x0 + crossLow * w;
          const xCrossH = x0 + crossHigh * w;

          // Band region shading
          // Low Band (Indigo)
          ctx.fillStyle = "rgba(99, 102, 241, 0.08)";
          ctx.fillRect(x0, y0, xCrossL - x0, h);

          // High Band (Amber)
          ctx.fillStyle = "rgba(245, 158, 11, 0.06)";
          ctx.fillRect(xCrossH, y0, x0 + w - xCrossH, h);

          // Crossover line markers
          ctx.strokeStyle = "rgba(99, 102, 241, 0.35)";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(xCrossL, y0);
          ctx.lineTo(xCrossL, y0 + h);
          ctx.stroke();

          ctx.strokeStyle = "rgba(245, 158, 11, 0.35)";
          ctx.beginPath();
          ctx.moveTo(xCrossH, y0);
          ctx.lineTo(xCrossH, y0 + h);
          ctx.stroke();

          const numSamples = 80;
          const points = [];

          for (let i = 0; i <= numSamples; i++) {
            const rNorm = i / numSamples; // Normalized radius in [0.0, 1.0]

            // Sigmoidal band crossover masks (mirrors Python kernel logic)
            const lowMask =
              1.0 / (1.0 + Math.exp(-((crossLow - rNorm) * 16.0)));
            const highMask =
              1.0 / (1.0 + Math.exp(-((rNorm - crossHigh) * 16.0)));
            const midMask = Math.max(0.0, 1.0 - lowMask - highMask);

            let gain =
              lowGain * lowMask + midGain * midMask + highGain * highMask;

            // Spectral Tilt slope
            if (Math.abs(tilt) > 1e-4) {
              gain *= Math.pow(rNorm + 0.08, tilt * 0.8);
            }

            // High-Shelf Damping
            if (damping > 1e-4) {
              const shelfMask =
                1.0 / (1.0 + Math.exp(-((rNorm - 0.82) * 18.0)));
              gain *= 1.0 - damping * shelfMask;
            }

            // Map to pixel coordinates
            const px = x0 + rNorm * w;
            const py =
              y0 + h - (Math.min(maxGain, gain) / maxGain) * (h - 18) - 10;
            points.push({ x: px, y: py });
          }

          // Gradient Fill under curve
          const fillGrad = ctx.createLinearGradient(x0, y0, x0 + w, y0);
          fillGrad.addColorStop(0.0, "rgba(99, 102, 241, 0.30)"); // Low: Indigo
          fillGrad.addColorStop(0.5, "rgba(56, 189, 248, 0.25)"); // Mid: Sky Blue
          fillGrad.addColorStop(1.0, "rgba(245, 158, 11, 0.25)"); // High: Amber

          ctx.fillStyle = fillGrad;
          ctx.beginPath();
          ctx.moveTo(points[0].x, y0 + h - 6);
          for (let pt of points) {
            ctx.lineTo(pt.x, pt.y);
          }
          ctx.lineTo(points[points.length - 1].x, y0 + h - 6);
          ctx.closePath();
          ctx.fill();

          // Response Curve Stroke
          const strokeGrad = ctx.createLinearGradient(x0, y0, x0 + w, y0);
          strokeGrad.addColorStop(0.0, "#818cf8");
          strokeGrad.addColorStop(0.45, "#38bdf8");
          strokeGrad.addColorStop(1.0, "#fbbf24");

          ctx.strokeStyle = strokeGrad;
          ctx.lineWidth = 2.0;
          ctx.beginPath();
          ctx.moveTo(points[0].x, points[0].y);
          for (let i = 1; i < points.length; i++) {
            ctx.lineTo(points[i].x, points[i].y);
          }
          ctx.stroke();

          ctx.font = "9px sans-serif";
          ctx.fillStyle = "#818cf8";
          ctx.fillText("LOW", x0 + 6, y0 + 13);

          ctx.fillStyle = "#38bdf8";
          ctx.fillText("MID", (xCrossL + xCrossH) / 2 - 8, y0 + 13);

          ctx.fillStyle = "#fbbf24";
          ctx.fillText("HIGH", x0 + w - 30, y0 + 13);

          ctx.fillStyle = "#52576b";
          ctx.fillText("1.0x", x0 + 4, unityY - 3);
          ctx.fillText("Nyquist", x0 + w - 38, y0 + h - 3);

          ctx.restore();
        },

        computeSize() {
          return [280, 105];
        },
      };

      node.addCustomWidget(eqWidget);

      // Re-render immediately when any slider changes
      node.onWidgetChanged = function () {
        node.setDirtyCanvas(true, true);
      };

      // Set default node dimensions to fit the visualizer cleanly
      node.setSize([310, Math.max(node.size[1], 460)]);
      return r;
    };
  },
});
