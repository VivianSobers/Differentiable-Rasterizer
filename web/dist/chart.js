/**
 * A small loss chart, drawn straight to a canvas.
 *
 * No charting library: the whole requirement is one line on a log axis,
 * redrawn every frame. A dependency would cost more to load than the entire
 * WebAssembly module.
 */
const SERIES = "#3987e5"; // Categorical slot 1, dark-surface step.
const GRID = "#2e2e2c";
const INK_MUTED = "#8a8a86";
export class LossChart {
    canvas;
    ctx;
    constructor(canvas) {
        this.canvas = canvas;
        const ctx = canvas.getContext("2d");
        if (!ctx)
            throw new Error("2D canvas context unavailable");
        this.ctx = ctx;
        this.resize();
    }
    /**
     * Match the backing store to the CSS size and device pixel ratio, so lines
     * are crisp on high-DPI displays instead of soft.
     */
    resize() {
        const dpr = window.devicePixelRatio || 1;
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = Math.max(1, Math.round(rect.width * dpr));
        this.canvas.height = Math.max(1, Math.round(rect.height * dpr));
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    draw(losses) {
        const rect = this.canvas.getBoundingClientRect();
        const w = rect.width;
        const h = rect.height;
        const ctx = this.ctx;
        ctx.clearRect(0, 0, w, h);
        if (losses.length < 2)
            return;
        // Log scale: a good fit spans two or three orders of magnitude, and on a
        // linear axis everything past the first few dozen iterations flattens onto
        // the floor and looks like nothing is happening.
        let min = Infinity;
        let max = -Infinity;
        for (const v of losses) {
            if (!Number.isFinite(v) || v <= 0)
                continue;
            if (v < min)
                min = v;
            if (v > max)
                max = v;
        }
        if (!Number.isFinite(min) || !Number.isFinite(max))
            return;
        const logMin = Math.log10(min);
        const logMax = Math.log10(max);
        const span = Math.max(logMax - logMin, 1e-6);
        const pad = 6;
        const x = (i) => pad + (i / (losses.length - 1)) * (w - pad * 2);
        const y = (v) => {
            const t = (Math.log10(Math.max(v, min)) - logMin) / span;
            return h - pad - t * (h - pad * 2);
        };
        // One gridline per decade — enough to read the scale, quiet enough to
        // stay behind the data.
        ctx.strokeStyle = GRID;
        ctx.lineWidth = 1;
        for (let d = Math.ceil(logMin); d <= Math.floor(logMax); d++) {
            const gy = Math.round(y(Math.pow(10, d))) + 0.5;
            ctx.beginPath();
            ctx.moveTo(pad, gy);
            ctx.lineTo(w - pad, gy);
            ctx.stroke();
        }
        ctx.strokeStyle = SERIES;
        ctx.lineWidth = 2;
        ctx.lineJoin = "round";
        ctx.beginPath();
        for (let i = 0; i < losses.length; i++) {
            const v = losses[i] ?? min;
            const px = x(i);
            const py = y(v);
            if (i === 0)
                ctx.moveTo(px, py);
            else
                ctx.lineTo(px, py);
        }
        ctx.stroke();
        ctx.fillStyle = INK_MUTED;
        ctx.font = "11px ui-monospace, monospace";
        ctx.fillText(formatLoss(max), pad + 2, pad + 10);
        ctx.fillText(formatLoss(min), pad + 2, h - pad - 2);
    }
}
export function formatLoss(v) {
    if (!Number.isFinite(v))
        return "—";
    return v.toExponential(2);
}
