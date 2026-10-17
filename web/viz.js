import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// Inject lightweight CSS for interactive DOM widgets in Nodes 2.0 & Legacy Canvas
(function injectStyles() {
    if (document.getElementById("spectral-monitor-styles")) return;
    const style = document.createElement("style");
    style.id = "spectral-monitor-styles";
    style.textContent = `
        .spec-container {
            display: flex;
            flex-direction: column;
            width: 100%;
            box-sizing: border-box;
            background: #090b10;
            border-radius: 8px;
            border: 1px solid #1c2230;
            padding: 8px;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            user-select: none;
            -webkit-user-select: none;
            gap: 6px;
        }
        .spec-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 4px;
        }
        .spec-btn-group {
            display: flex;
            align-items: center;
            gap: 3px;
        }
        .spec-btn {
            background: #141822;
            border: 1px solid #222a3d;
            color: #94a3b8;
            font-size: 8.5px;
            font-weight: 700;
            font-family: inherit;
            padding: 3px 6px;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        .spec-btn:hover {
            background: #1e2638;
            color: #f1f5f9;
            border-color: #38bdf8;
        }
        .spec-btn.active {
            background: #2563eb;
            color: #ffffff;
            border-color: #60a5fa;
        }
        .spec-btn.eq-active {
            background: #ea580c;
            color: #ffffff;
            border-color: #fb923c;
        }
        .spec-badge {
            font-size: 8.5px;
            font-weight: 700;
            padding: 2px 6px;
            border-radius: 4px;
            background: #11141c;
            color: #10b981;
            border: 1px solid #1a2233;
        }
        .spec-badge.paused {
            color: #f59e0b;
        }
        .spec-canvas {
            width: 100%;
            height: 135px;
            background: #06070a;
            border-radius: 5px;
            border: 1px solid #151a24;
            display: block;
        }
        .spec-scrubber-bar {
            display: flex;
            align-items: center;
            gap: 5px;
            background: #0f121a;
            padding: 3px 6px;
            border-radius: 5px;
            border: 1px solid #1a202e;
        }
        .spec-slider {
            -webkit-appearance: none;
            appearance: none;
            flex: 1;
            height: 5px;
            background: #22293a;
            border-radius: 3px;
            outline: none;
            cursor: pointer;
            margin: 0;
        }
        .spec-slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 13px;
            height: 13px;
            border-radius: 50%;
            background: #38bdf8;
            cursor: pointer;
            box-shadow: 0 0 6px rgba(56, 189, 248, 0.55);
            border: none;
        }
        .spec-slider::-moz-range-thumb {
            width: 13px;
            height: 13px;
            border-radius: 50%;
            background: #38bdf8;
            cursor: pointer;
            box-shadow: 0 0 6px rgba(56, 189, 248, 0.55);
            border: none;
        }
        .spec-step-txt {
            font-size: 8px;
            color: #94a3b8;
            min-width: 38px;
            text-align: right;
        }
    `;
    document.head.appendChild(style);
})();

// Global registry for direct dispatch across Nodes 2.0 Vue instances
const visualizerRegistry = new Map();
const allVisualizers = new Set();

app.registerExtension({
    name: "SpectralGuidanceEQ.SpectrumMonitor",

    async setup(app) {
        api.addEventListener("spectral_spectrum_step", (event) => {
            const data = event.detail;
            if (!data) return;

            const targetId = data.node_id != null ? String(data.node_id) : "";
            let handled = false;

            if (targetId && visualizerRegistry.has(targetId)) {
                visualizerRegistry.get(targetId).onStepTelemetry(data);
                handled = true;
            }

            if (!handled) {
                for (const viz of allVisualizers) {
                    viz.onStepTelemetry(data);
                }
            }
        });
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        const isEQ = nodeData.name === "SpectralGuidanceEQ";
        const isVisualizer = nodeData.name === "SpectralSpectrumVisualizer";

        if (!isEQ && !isVisualizer) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            const node = this;

            const getVal = (name, fallback) => {
                const w = node.widgets?.find((w) => w.name === name);
                return (w && typeof w.value === "number") ? w.value : fallback;
            };

            // -------------------------------------------------------------
            // NODE 1: SpectralGuidanceEQ -> Compact Parametric EQ Response
            // -------------------------------------------------------------
            if (isEQ) {
                const eqContainer = document.createElement("div");
                eqContainer.className = "spec-container";

                // Header
                const eqHeader = document.createElement("div");
                eqHeader.className = "spec-header";
                eqHeader.innerHTML = `
                    <span style="font-size: 9px; font-weight: 700; color: #38bdf8;">PARAMETRIC LATENT EQ RESPONSE</span>
                    <span style="font-size: 8px; color: #64748b;">TRANSFER FUNCTION</span>
                `;
                eqContainer.appendChild(eqHeader);

                // Canvas
                const eqCanvas = document.createElement("canvas");
                eqCanvas.className = "spec-canvas";
                eqCanvas.style.height = "105px";
                eqContainer.appendChild(eqCanvas);

                function drawEQCurve() {
                    const rect = eqCanvas.getBoundingClientRect();
                    const w = rect.width > 20 ? rect.width : 280;
                    const h = 105;
                    const dpr = window.devicePixelRatio || 1;

                    if (eqCanvas.width !== Math.floor(w * dpr) || eqCanvas.height !== Math.floor(h * dpr)) {
                        eqCanvas.width = Math.floor(w * dpr);
                        eqCanvas.height = Math.floor(h * dpr);
                    }

                    const ctx = eqCanvas.getContext("2d");
                    ctx.save();
                    ctx.scale(dpr, dpr);
                    ctx.clearRect(0, 0, w, h);

                    const lowG = getVal("low_freq_gain", 1.0);
                    const midG = getVal("mid_freq_gain", 1.0);
                    const highG = getVal("high_freq_gain", 1.0);
                    const crossLow = getVal("crossover_low", 0.12);
                    const crossHigh = getVal("crossover_high", 0.45);
                    const tilt = getVal("spectral_tilt", 0.0);
                    const damping = getVal("high_shelf_damping", 0.15);

                    const pX = 6;
                    const pY = 4;
                    const pW = w - 12;
                    const pH = h - 8;

                    // Band shading
                    const xCL = pX + crossLow * pW;
                    const xCH = pX + crossHigh * pW;

                    ctx.fillStyle = "rgba(99, 102, 241, 0.12)";
                    ctx.fillRect(pX, pY, xCL - pX, pH);
                    ctx.fillStyle = "rgba(56, 189, 248, 0.07)";
                    ctx.fillRect(xCL, pY, xCH - xCL, pH);
                    ctx.fillStyle = "rgba(245, 158, 11, 0.10)";
                    ctx.fillRect(xCH, pY, (pX + pW) - xCH, pH);

                    // Dividers
                    ctx.strokeStyle = "rgba(99, 102, 241, 0.40)";
                    ctx.beginPath();
                    ctx.moveTo(xCL, pY);
                    ctx.lineTo(xCL, pY + pH);
                    ctx.stroke();

                    ctx.strokeStyle = "rgba(245, 158, 11, 0.40)";
                    ctx.beginPath();
                    ctx.moveTo(xCH, pY);
                    ctx.lineTo(xCH, pY + pH);
                    ctx.stroke();

                    // Reference line 1.0
                    const maxG = Math.max(3.2, lowG, midG, highG);
                    const neutralY = pY + pH - (1.0 / maxG) * (pH - 16) - 8;
                    ctx.strokeStyle = "#232838";
                    ctx.setLineDash([2, 3]);
                    ctx.beginPath();
                    ctx.moveTo(pX, neutralY);
                    ctx.lineTo(pX + pW, neutralY);
                    ctx.stroke();
                    ctx.setLineDash([]);

                    // Compute points
                    const samples = 48;
                    const pts = [];
                    for (let s = 0; s <= samples; s++) {
                        const rNorm = s / samples;
                        const lowMask = 1.0 / (1.0 + Math.exp(-((crossLow - rNorm) * 16.0)));
                        const highMask = 1.0 / (1.0 + Math.exp(-((rNorm - crossHigh) * 16.0)));
                        const midMask = Math.max(0.0, 1.0 - lowMask - highMask);

                        let gain = (lowG * lowMask) + (midG * midMask) + (highG * highMask);
                        if (Math.abs(tilt) > 1e-4) {
                            gain *= Math.pow(rNorm + 0.08, tilt * 0.8);
                        }
                        if (damping > 1e-4) {
                            const shelfMask = 1.0 / (1.0 + Math.exp(-((rNorm - 0.82) * 18.0)));
                            gain *= (1.0 - damping * shelfMask);
                        }

                        const px = pX + rNorm * pW;
                        const py = pY + pH - (Math.min(maxG, gain) / maxG) * (pH - 16) - 8;
                        pts.push({ x: px, y: py });
                    }

                    // Fill
                    const grad = ctx.createLinearGradient(pX, pY, pX + pW, pY);
                    grad.addColorStop(0.0, "rgba(99, 102, 241, 0.32)");
                    grad.addColorStop(0.5, "rgba(56, 189, 248, 0.22)");
                    grad.addColorStop(1.0, "rgba(245, 158, 11, 0.26)");

                    ctx.beginPath();
                    ctx.moveTo(pts[0].x, pts[0].y);
                    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
                    ctx.lineTo(pX + pW, pY + pH);
                    ctx.lineTo(pX, pY + pH);
                    ctx.closePath();
                    ctx.fillStyle = grad;
                    ctx.fill();

                    // Stroke
                    ctx.strokeStyle = "#38bdf8";
                    ctx.lineWidth = 1.8;
                    ctx.beginPath();
                    ctx.moveTo(pts[0].x, pts[0].y);
                    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
                    ctx.stroke();

                    // Labels
                    ctx.font = "8px sans-serif";
                    ctx.fillStyle = "#818cf8";
                    ctx.fillText(`LOW x${lowG.toFixed(1)}`, pX + 4, pY + 10);
                    ctx.fillStyle = "#38bdf8";
                    ctx.fillText(`MID x${midG.toFixed(1)}`, (xCL + xCH) / 2 - 14, pY + 10);
                    ctx.fillStyle = "#fbbf24";
                    ctx.fillText(`HIGH x${highG.toFixed(1)}`, pX + pW - 46, pY + 10);

                    ctx.restore();
                }

                // Prevent pointer bubbling
                ["pointerdown", "mousedown", "touchstart", "wheel"].forEach((ev) => {
                    eqContainer.addEventListener(ev, (e) => e.stopPropagation());
                });

                // Attach to slider changes
                if (node.widgets) {
                    node.widgets.forEach((w) => {
                        const origCallback = w.callback;
                        w.callback = function () {
                            const res = origCallback ? origCallback.apply(this, arguments) : undefined;
                            drawEQCurve();
                            return res;
                        };
                    });
                }

                node.addDOMWidget("spectral_eq_preview_dom", "custom", eqContainer, {
                    getMinHeight: () => 140,
                    hideOnZoom: false,
                    serialize: false,
                });

                setTimeout(drawEQCurve, 100);
                return r;
            }

            // -------------------------------------------------------------
            // NODE 2: SpectralSpectrumVisualizer -> Oscilloscope & Scrubber
            // -------------------------------------------------------------
            const container = document.createElement("div");
            container.className = "spec-container";

            // Prevent event bubbling so Nodes 2.0 does not drag node or zoom graph
            ["pointerdown", "mousedown", "touchstart", "click", "dblclick", "wheel"].forEach((ev) => {
                container.addEventListener(ev, (e) => e.stopPropagation());
            });

            // Widget state
            const state = {
                history: [],
                currentStepIdx: 0,
                activeView: "psd", // "psd" | "heatmap" | "bands"
                showEqOverlay: true,
                isLive: true,
            };

            // 1. Header Bar
            const header = document.createElement("div");
            header.className = "spec-header";

            const btnGroup = document.createElement("div");
            btnGroup.className = "spec-btn-group";

            const tabs = [
                { id: "psd", label: "1D PSD" },
                { id: "heatmap", label: "2D FFT" },
                { id: "bands", label: "BANDS" },
            ];

            const tabButtons = {};
            tabs.forEach((tab) => {
                const btn = document.createElement("button");
                btn.className = `spec-btn ${state.activeView === tab.id ? "active" : ""}`;
                btn.innerText = tab.label;
                btn.onclick = (e) => {
                    e.stopPropagation();
                    state.activeView = tab.id;
                    Object.values(tabButtons).forEach((b) => b.classList.remove("active"));
                    btn.classList.add("active");
                    render();
                };
                tabButtons[tab.id] = btn;
                btnGroup.appendChild(btn);
            });

            const btnEq = document.createElement("button");
            btnEq.className = `spec-btn ${state.showEqOverlay ? "eq-active" : ""}`;
            btnEq.innerText = "EQ: ON";
            btnEq.onclick = (e) => {
                e.stopPropagation();
                state.showEqOverlay = !state.showEqOverlay;
                btnEq.innerText = state.showEqOverlay ? "EQ: ON" : "EQ: OFF";
                btnEq.className = `spec-btn ${state.showEqOverlay ? "eq-active" : ""}`;
                render();
            };
            btnGroup.appendChild(btnEq);
            header.appendChild(btnGroup);

            const badge = document.createElement("div");
            badge.className = "spec-badge";
            badge.innerText = "IDLE";
            header.appendChild(badge);
            container.appendChild(header);

            // 2. Telemetry Canvas
            const canvas = document.createElement("canvas");
            canvas.className = "spec-canvas";
            container.appendChild(canvas);

            // 3. Timeline Scrubber Bar
            const scrubberBar = document.createElement("div");
            scrubberBar.className = "spec-scrubber-bar";

            const btnPrev = document.createElement("button");
            btnPrev.className = "spec-btn";
            btnPrev.innerText = "◀";
            btnPrev.onclick = (e) => {
                e.stopPropagation();
                state.isLive = false;
                state.currentStepIdx = Math.max(0, state.currentStepIdx - 1);
                slider.value = String(state.currentStepIdx);
                render();
            };

            const btnNext = document.createElement("button");
            btnNext.className = "spec-btn";
            btnNext.innerText = "▶";
            btnNext.onclick = (e) => {
                e.stopPropagation();
                state.currentStepIdx = Math.min(state.history.length - 1, state.currentStepIdx + 1);
                state.isLive = (state.currentStepIdx === state.history.length - 1);
                slider.value = String(state.currentStepIdx);
                render();
            };

            const slider = document.createElement("input");
            slider.type = "range";
            slider.className = "spec-slider";
            slider.min = "0";
            slider.max = "0";
            slider.value = "0";

            slider.addEventListener("input", (e) => {
                e.stopPropagation();
                const idx = parseInt(e.target.value, 10);
                state.currentStepIdx = idx;
                state.isLive = (idx === state.history.length - 1);
                render();
            });

            const stepTxt = document.createElement("span");
            stepTxt.className = "spec-step-txt";
            stepTxt.innerText = "0/0";

            const btnLive = document.createElement("button");
            btnLive.className = "spec-btn active";
            btnLive.innerText = "LIVE";
            btnLive.onclick = (e) => {
                e.stopPropagation();
                state.isLive = true;
                if (state.history.length > 0) {
                    state.currentStepIdx = state.history.length - 1;
                    slider.value = String(state.currentStepIdx);
                }
                render();
            };

            scrubberBar.appendChild(btnPrev);
            scrubberBar.appendChild(btnNext);
            scrubberBar.appendChild(slider);
            scrubberBar.appendChild(stepTxt);
            scrubberBar.appendChild(btnLive);
            container.appendChild(scrubberBar);

            function render() {
                const total = state.history.length;
                const activeIdx = Math.min(state.currentStepIdx, Math.max(0, total - 1));
                const currentData = total > 0 ? state.history[activeIdx] : null;

                // Update Header Status Badge
                if (currentData) {
                    badge.innerText = `S:${currentData.step} σ:${currentData.sigma.toFixed(2)}`;
                    badge.className = `spec-badge ${state.isLive ? "" : "paused"}`;
                    stepTxt.innerText = `${activeIdx + 1}/${total}`;
                } else {
                    badge.innerText = "AWAITING RUN";
                    badge.className = "spec-badge";
                    stepTxt.innerText = "0/0";
                }

                btnLive.className = `spec-btn ${state.isLive ? "active" : ""}`;

                // Paint Canvas
                const rect = canvas.getBoundingClientRect();
                const w = rect.width > 20 ? rect.width : 280;
                const h = 135;
                const dpr = window.devicePixelRatio || 1;

                if (canvas.width !== Math.floor(w * dpr) || canvas.height !== Math.floor(h * dpr)) {
                    canvas.width = Math.floor(w * dpr);
                    canvas.height = Math.floor(h * dpr);
                }

                const ctx = canvas.getContext("2d");
                ctx.save();
                ctx.scale(dpr, dpr);
                ctx.clearRect(0, 0, w, h);

                const crossLow = getVal("crossover_low", currentData?.eq?.cross_low ?? 0.12);
                const crossHigh = getVal("crossover_high", currentData?.eq?.cross_high ?? 0.45);

                const plotX = 6;
                const plotY = 6;
                const plotW = w - 12;
                const plotH = h - 12;

                // ------------------ VIEW: 1D PSD ------------------
                if (state.activeView === "psd") {
                    const xCrossL = plotX + crossLow * plotW;
                    const xCrossH = plotX + crossHigh * plotW;

                    ctx.fillStyle = "rgba(99, 102, 241, 0.10)";
                    ctx.fillRect(plotX, plotY, xCrossL - plotX, plotH);
                    ctx.fillStyle = "rgba(245, 158, 11, 0.08)";
                    ctx.fillRect(xCrossH, plotY, (plotX + plotW) - xCrossH, plotH);

                    ctx.strokeStyle = "rgba(99, 102, 241, 0.45)";
                    ctx.beginPath();
                    ctx.moveTo(xCrossL, plotY);
                    ctx.lineTo(xCrossL, plotY + plotH);
                    ctx.stroke();

                    ctx.strokeStyle = "rgba(245, 158, 11, 0.45)";
                    ctx.beginPath();
                    ctx.moveTo(xCrossH, plotY);
                    ctx.lineTo(xCrossH, plotY + plotH);
                    ctx.stroke();

                    // Natural prior 1/f² reference slope
                    ctx.strokeStyle = "#252b3b";
                    ctx.setLineDash([2, 3]);
                    ctx.beginPath();
                    ctx.moveTo(plotX + 4, plotY + 10);
                    ctx.quadraticCurveTo(plotX + plotW * 0.4, plotY + plotH * 0.75, plotX + plotW - 4, plotY + plotH - 10);
                    ctx.stroke();
                    ctx.setLineDash([]);

                    // Real-Time Measured PSD Curve
                    if (currentData && currentData.psd && currentData.psd.length > 0) {
                        const pts = currentData.psd;
                        const stepW = plotW / (pts.length - 1);

                        ctx.beginPath();
                        ctx.moveTo(plotX, plotY + plotH - pts[0] * (plotH - 14) - 7);
                        for (let i = 1; i < pts.length; i++) {
                            const px = plotX + i * stepW;
                            const py = plotY + plotH - pts[i] * (plotH - 14) - 7;
                            ctx.lineTo(px, py);
                        }

                        const fillGrad = ctx.createLinearGradient(plotX, plotY, plotX + plotW, plotY);
                        fillGrad.addColorStop(0.0, "rgba(56, 189, 248, 0.32)");
                        fillGrad.addColorStop(0.6, "rgba(129, 140, 248, 0.22)");
                        fillGrad.addColorStop(1.0, "rgba(236, 72, 153, 0.16)");

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
                        ctx.font = "9.5px sans-serif";
                        ctx.fillText("Sampling step telemetry will stream here live...", plotX + plotW / 2 - 92, plotY + plotH / 2);
                    }

                    // Parametric EQ Overlay
                    if (state.showEqOverlay) {
                        const lowG = getVal("low_freq_gain", currentData?.eq?.low_gain ?? 1.0);
                        const midG = getVal("mid_freq_gain", currentData?.eq?.mid_gain ?? 1.0);
                        const highG = getVal("high_freq_gain", currentData?.eq?.high_gain ?? 1.0);
                        const tilt = getVal("spectral_tilt", currentData?.eq?.tilt ?? 0.0);
                        const damping = getVal("high_shelf_damping", currentData?.eq?.damping ?? 0.15);

                        const maxG = Math.max(3.2, lowG, midG, highG);
                        const eqPts = [];
                        const samples = 48;

                        for (let s = 0; s <= samples; s++) {
                            const rNorm = s / samples;
                            const lowMask = 1.0 / (1.0 + Math.exp(-((crossLow - rNorm) * 16.0)));
                            const highMask = 1.0 / (1.0 + Math.exp(-((rNorm - crossHigh) * 16.0)));
                            const midMask = Math.max(0.0, 1.0 - lowMask - highMask);

                            let gain = (lowG * lowMask) + (midG * midMask) + (highG * highMask);
                            if (Math.abs(tilt) > 1e-4) gain *= Math.pow(rNorm + 0.08, tilt * 0.8);
                            if (damping > 1e-4) {
                                const shelfMask = 1.0 / (1.0 + Math.exp(-((rNorm - 0.82) * 18.0)));
                                gain *= (1.0 - damping * shelfMask);
                            }

                            const px = plotX + rNorm * plotW;
                            const py = plotY + plotH - (Math.min(maxG, gain) / maxG) * (plotH - 20) - 10;
                            eqPts.push({ x: px, y: py });
                        }

                        ctx.strokeStyle = "#f97316";
                        ctx.lineWidth = 1.5;
                        ctx.setLineDash([3, 2]);
                        ctx.beginPath();
                        ctx.moveTo(eqPts[0].x, eqPts[0].y);
                        for (let i = 1; i < eqPts.length; i++) ctx.lineTo(eqPts[i].x, eqPts[i].y);
                        ctx.stroke();
                        ctx.setLineDash([]);
                    }

                    // Region Labels
                    ctx.font = "8px sans-serif";
                    ctx.fillStyle = "#818cf8";
                    ctx.fillText("LOW (Macro)", plotX + 4, plotY + 11);
                    ctx.fillStyle = "#38bdf8";
                    ctx.fillText("MID (Form)", (xCrossL + xCrossH) / 2 - 16, plotY + 11);
                    ctx.fillStyle = "#fbbf24";
                    ctx.fillText("HIGH (Detail)", plotX + plotW - 48, plotY + 11);

                // ------------------ VIEW: 2D FFT HEATMAP ------------------
                } else if (state.activeView === "heatmap") {
                    if (currentData && currentData.heatmap && currentData.heatmap.length > 0) {
                        const grid = currentData.heatmap;
                        const rows = grid.length;
                        const cols = grid[0].length;
                        const cellW = plotW / cols;
                        const cellH = plotH / rows;

                        for (let r = 0; r < rows; r++) {
                            for (let c = 0; c < cols; c++) {
                                const val = grid[r][c];
                                const red = Math.floor(Math.min(255, val * 320));
                                const green = Math.floor(Math.min(255, Math.pow(val, 1.3) * 260));
                                const blue = Math.floor(Math.min(255, (1.0 - val) * 120 + val * 240));
                                ctx.fillStyle = `rgb(${red},${green},${blue})`;
                                ctx.fillRect(plotX + c * cellW, plotY + r * cellH, cellW + 0.4, cellH + 0.4);
                            }
                        }

                        const cx = plotX + plotW / 2;
                        const cy = plotY + plotH / 2;
                        const maxR = Math.min(plotW, plotH) / 2;

                        ctx.strokeStyle = "rgba(255, 255, 255, 0.40)";
                        ctx.setLineDash([2, 3]);
                        ctx.lineWidth = 1;

                        ctx.beginPath();
                        ctx.arc(cx, cy, crossLow * maxR, 0, Math.PI * 2);
                        ctx.stroke();

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
                        ctx.font = "9.5px sans-serif";
                        ctx.fillText("Awaiting 2D FFT telemetry...", plotX + plotW / 2 - 62, plotY + plotH / 2);
                    }

                // ------------------ VIEW: 3-BAND ENERGY METERS ------------------
                } else if (state.activeView === "bands") {
                    const bands = currentData?.bands || { low: 0.33, mid: 0.33, high: 0.34 };
                    const barW = (plotW - 24) / 3;
                    const meterH = plotH - 32;
                    const meterY = plotY + 12;

                    const items = [
                        { name: "LOW / MACRO", ratio: bands.low, color: "#6366f1" },
                        { name: "MID / ANATOMY", ratio: bands.mid, color: "#38bdf8" },
                        { name: "HIGH / TEXTURE", ratio: bands.high, color: "#f59e0b" },
                    ];

                    items.forEach((item, idx) => {
                        const bx = plotX + 6 + idx * (barW + 6);
                        ctx.fillStyle = "#141722";
                        ctx.beginPath();
                        ctx.roundRect(bx, meterY, barW, meterH, 4);
                        ctx.fill();

                        const fillH = Math.max(4, item.ratio * meterH);
                        const fillY = meterY + meterH - fillH;
                        ctx.fillStyle = item.color;
                        ctx.beginPath();
                        ctx.roundRect(bx, fillY, barW, fillH, 4);
                        ctx.fill();

                        ctx.font = "bold 9px monospace";
                        ctx.fillStyle = "#ffffff";
                        const pctText = `${Math.round(item.ratio * 100)}%`;
                        ctx.fillText(pctText, bx + barW / 2 - 9, fillY - 4 > meterY + 10 ? fillY - 4 : meterY + 14);

                        ctx.font = "7.5px sans-serif";
                        ctx.fillStyle = "#94a3b8";
                        ctx.fillText(item.name, bx + 2, meterY + meterH + 12);
                    });
                }

                ctx.restore();
            }

            const visualizerController = {
                onStepTelemetry(data) {
                    if (data.step <= 1 || (state.history.length > 0 && data.step < state.history[state.history.length - 1].step)) {
                        state.history = [];
                    }
                    state.history.push(data);
                    slider.max = String(Math.max(0, state.history.length - 1));

                    if (state.isLive) {
                        state.currentStepIdx = state.history.length - 1;
                        slider.value = String(state.currentStepIdx);
                    }

                    render();
                }
            };

            allVisualizers.add(visualizerController);
            if (node.id != null) {
                visualizerRegistry.set(String(node.id), visualizerController);
            }

            const onRemoved = node.onRemoved;
            node.onRemoved = function () {
                allVisualizers.delete(visualizerController);
                if (node.id != null) visualizerRegistry.delete(String(node.id));
                return onRemoved ? onRemoved.apply(this, arguments) : undefined;
            };

            node.addDOMWidget("spectral_monitor_dom", "custom", container, {
                getMinHeight: () => 240,
                hideOnZoom: false,
                serialize: false,
            });

            node.setSize([320, Math.max(node.size ? node.size[1] : 320, 310)]);
            setTimeout(render, 100);

            return r;
        };
    }
});
