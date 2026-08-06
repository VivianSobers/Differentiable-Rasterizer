/**
 * Browser viewer: runs a real fit in WebAssembly and draws every iteration.
 *
 * Nothing here reimplements the algorithm — the rasterizer, the gradients, and
 * Adam all run in the same Rust code the CLI uses. This file owns what only a
 * browser can do: loading an image the user picked, pacing the loop against
 * the frame budget, and putting pixels on screen.
 */

import init, { WasmFitter } from "../pkg/diffrast_wasm.js";
import { LossChart, formatLoss } from "./chart.js";

/** Fit resolution. Small on purpose: this runs on one thread on whatever
 * device the visitor happens to have, and 160px converges in seconds. */
const FIT_SIZE = 160;

/** Milliseconds of fitting per frame. Below the ~16ms frame budget so the page
 * stays responsive while the optimizer runs. */
const FRAME_BUDGET_MS = 10;

interface Elements {
  stage: HTMLCanvasElement;
  target: HTMLCanvasElement;
  chart: HTMLCanvasElement;
  start: HTMLButtonElement;
  reset: HTMLButtonElement;
  download: HTMLButtonElement;
  file: HTMLInputElement;
  triangles: HTMLInputElement;
  iters: HTMLInputElement;
  trianglesOut: HTMLElement;
  itersOut: HTMLElement;
  status: HTMLElement;
  statIter: HTMLElement;
  statLoss: HTMLElement;
  statBest: HTMLElement;
  statSigma: HTMLElement;
}

function element<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing element #${id}`);
  return el as T;
}

class Viewer {
  private fitter: WasmFitter | null = null;
  private running = false;
  private rafId = 0;
  private targetRgba: Uint8Array | null = null;
  private chart: LossChart;
  private stageCtx: CanvasRenderingContext2D;

  constructor(private readonly el: Elements) {
    this.chart = new LossChart(el.chart);

    const ctx = el.stage.getContext("2d");
    if (!ctx) throw new Error("2D canvas context unavailable");
    this.stageCtx = ctx;
    // Nearest-neighbour: the fit is 160px shown large, and smoothing would
    // hide exactly the soft-edge behavior the project is about.
    this.stageCtx.imageSmoothingEnabled = false;

    el.start.addEventListener("click", () => this.toggle());
    el.reset.addEventListener("click", () => this.reset());
    el.download.addEventListener("click", () => void this.download());
    el.file.addEventListener("change", () => void this.onFile());
    el.triangles.addEventListener("input", () => {
      el.trianglesOut.textContent = el.triangles.value;
      this.reset();
    });
    el.iters.addEventListener("input", () => {
      el.itersOut.textContent = el.iters.value;
      this.reset();
    });
    window.addEventListener("resize", () => {
      this.chart.resize();
      if (this.fitter) this.chart.draw(this.fitter.losses());
    });
  }

  /** Load the built-in target, then make the page usable. */
  async begin(): Promise<void> {
    this.targetRgba = syntheticTarget(FIT_SIZE);
    this.paintTarget();
    this.reset();
    this.setStatus("ready — press Start");
  }

  private toggle(): void {
    if (!this.fitter) this.reset();
    this.running = !this.running;
    this.el.start.textContent = this.running ? "Pause" : "Start";
    if (this.running) {
      this.setStatus("fitting…");
      this.rafId = requestAnimationFrame(() => this.frame());
    } else {
      cancelAnimationFrame(this.rafId);
      this.setStatus("paused");
    }
  }

  private reset(): void {
    cancelAnimationFrame(this.rafId);
    this.running = false;
    this.el.start.textContent = "Start";

    if (!this.targetRgba) return;
    const triangles = Number(this.el.triangles.value);
    const iters = Number(this.el.iters.value);

    try {
      this.fitter = new WasmFitter(
        FIT_SIZE,
        FIT_SIZE,
        this.targetRgba,
        triangles,
        iters,
        0,
      );
    } catch (err) {
      // A construction failure is a real error the user can act on (usually a
      // bad image), so surface it rather than leaving a dead page.
      this.setStatus(`error: ${String(err)}`);
      this.fitter = null;
      return;
    }

    this.paintStage();
    this.chart.draw([]);
    this.updateStats();
    this.setStatus("ready");
  }

  /** One animation frame: fit for a fixed slice of time, then draw once. */
  private frame(): void {
    if (!this.running || !this.fitter) return;

    const deadline = performance.now() + FRAME_BUDGET_MS;
    let stepped = 0;
    // Step in small batches and re-check the clock: iteration cost varies with
    // triangle count, so a fixed batch size would either waste the budget or
    // blow through it.
    while (performance.now() < deadline) {
      const ran = this.fitter.step_many(4);
      stepped += ran;
      if (ran === 0) break;
    }

    this.paintStage();
    this.chart.draw(this.fitter.losses());
    this.updateStats();

    if (this.fitter.done || stepped === 0) {
      this.running = false;
      this.el.start.textContent = "Start";
      this.setStatus(`done — ${this.fitter.iter} iterations`);
      return;
    }
    this.rafId = requestAnimationFrame(() => this.frame());
  }

  private paintStage(): void {
    if (!this.fitter) return;
    const rgba = this.fitter.render_rgba();
    const image = new ImageData(new Uint8ClampedArray(rgba), FIT_SIZE, FIT_SIZE);
    // Round-trip through a bitmap-sized canvas so the result can be drawn
    // scaled: putImageData ignores transforms entirely.
    const scratch = document.createElement("canvas");
    scratch.width = FIT_SIZE;
    scratch.height = FIT_SIZE;
    scratch.getContext("2d")?.putImageData(image, 0, 0);

    const { stage } = this.el;
    this.stageCtx.imageSmoothingEnabled = false;
    this.stageCtx.clearRect(0, 0, stage.width, stage.height);
    this.stageCtx.drawImage(scratch, 0, 0, stage.width, stage.height);
  }

  private paintTarget(): void {
    if (!this.targetRgba) return;
    const { target } = this.el;
    target.width = FIT_SIZE;
    target.height = FIT_SIZE;
    const image = new ImageData(new Uint8ClampedArray(this.targetRgba), FIT_SIZE, FIT_SIZE);
    target.getContext("2d")?.putImageData(image, 0, 0);
  }

  private updateStats(): void {
    const f = this.fitter;
    this.el.statIter.textContent = f ? String(f.iter) : "—";
    this.el.statLoss.textContent = f ? formatLoss(f.loss) : "—";
    this.el.statBest.textContent = f ? formatLoss(f.best_loss) : "—";
    this.el.statSigma.textContent = f ? f.sigma.toFixed(4) : "—";
  }

  private setStatus(text: string): void {
    this.el.status.textContent = text;
  }

  /** Load a user-supplied image, letterboxed into the square fit canvas. */
  private async onFile(): Promise<void> {
    const file = this.el.file.files?.[0];
    if (!file) return;

    this.setStatus("loading image…");
    try {
      const bitmap = await createImageBitmap(file);
      const scratch = document.createElement("canvas");
      scratch.width = FIT_SIZE;
      scratch.height = FIT_SIZE;
      const ctx = scratch.getContext("2d");
      if (!ctx) throw new Error("2D canvas context unavailable");

      // Cover-crop rather than stretch: a squashed target would make the fit
      // look wrong for reasons that have nothing to do with the optimizer.
      const scale = Math.max(FIT_SIZE / bitmap.width, FIT_SIZE / bitmap.height);
      const w = bitmap.width * scale;
      const h = bitmap.height * scale;
      ctx.drawImage(bitmap, (FIT_SIZE - w) / 2, (FIT_SIZE - h) / 2, w, h);
      bitmap.close();

      this.targetRgba = new Uint8Array(ctx.getImageData(0, 0, FIT_SIZE, FIT_SIZE).data);
      this.paintTarget();
      this.reset();
      this.setStatus(`loaded ${file.name} — press Start`);
    } catch (err) {
      this.setStatus(`could not read that image: ${String(err)}`);
    }
  }

  /** Export the current scene at high resolution. */
  private async download(): Promise<void> {
    if (!this.fitter) return;
    const size = 1024;
    try {
      // Geometry is stored in normalized coordinates, so the fitted scene
      // re-renders at any resolution without being refitted.
      const rgba = this.fitter.render_rgba_at(size, size);
      const scratch = document.createElement("canvas");
      scratch.width = size;
      scratch.height = size;
      scratch
        .getContext("2d")
        ?.putImageData(new ImageData(new Uint8ClampedArray(rgba), size, size), 0, 0);

      const blob = await new Promise<Blob | null>((resolve) =>
        scratch.toBlob(resolve, "image/png"),
      );
      if (!blob) throw new Error("canvas produced no image");

      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `diffrast-${this.fitter.triangles}tris.png`;
      link.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      this.setStatus(`export failed: ${String(err)}`);
    }
  }
}

/**
 * The same gradient-plus-shapes target the CLI generates, so the browser demo
 * and the command line show the same problem.
 */
function syntheticTarget(size: number): Uint8Array {
  const data = new Uint8Array(size * size * 4);
  const shapes: Array<{ v: number[][]; c: number[] }> = [
    { v: [[0.15, 0.7], [0.55, 0.68], [0.35, 0.2]], c: [242, 217, 64] },
    { v: [[0.5, 0.3], [0.88, 0.45], [0.62, 0.85]], c: [26, 64, 110] },
  ];

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const u = x / size;
      const v = y / size;
      let r = (0.15 + 0.5 * u) * 255;
      let g = (0.1 + 0.4 * v) * 255;
      let b = (0.55 - 0.3 * u) * 255;

      const px = (x + 0.5) / size;
      const py = (y + 0.5) / size;
      for (const shape of shapes) {
        if (insideTriangle(shape.v, px, py)) {
          [r, g, b] = shape.c as [number, number, number];
        }
      }

      const i = (y * size + x) * 4;
      data[i] = r;
      data[i + 1] = g;
      data[i + 2] = b;
      data[i + 3] = 255;
    }
  }
  return data;
}

function insideTriangle(v: number[][], px: number, py: number): boolean {
  const [a, b, c] = v as [number[], number[], number[]];
  const sign = (p: number[], q: number[]) =>
    (q[0]! - p[0]!) * (py - p[1]!) - (q[1]! - p[1]!) * (px - p[0]!);
  const d0 = sign(a, b);
  const d1 = sign(b, c);
  const d2 = sign(c, a);
  return (d0 >= 0 && d1 >= 0 && d2 >= 0) || (d0 <= 0 && d1 <= 0 && d2 <= 0);
}

async function main(): Promise<void> {
  const status = document.getElementById("status");
  try {
    await init();
  } catch (err) {
    if (status) {
      status.textContent =
        "could not load WebAssembly — run `npm run wasm`, and serve this page over http rather than opening the file directly";
    }
    console.error(err);
    return;
  }

  const viewer = new Viewer({
    stage: element("stage"),
    target: element("target"),
    chart: element("chart"),
    start: element("start"),
    reset: element("reset"),
    download: element("download"),
    file: element("file"),
    triangles: element("triangles"),
    iters: element("iters"),
    trianglesOut: element("triangles-out"),
    itersOut: element("iters-out"),
    status: element("status"),
    statIter: element("stat-iter"),
    statLoss: element("stat-loss"),
    statBest: element("stat-best"),
    statSigma: element("stat-sigma"),
  });
  await viewer.begin();
}

void main();
