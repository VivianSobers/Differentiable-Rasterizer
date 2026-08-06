/* tslint:disable */
/* eslint-disable */

/**
 * A fit running in the browser.
 */
export class WasmFitter {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Every loss so far, for plotting.
     */
    losses(): Float32Array;
    /**
     * Create a fitter for an RGBA image.
     *
     * `rgba` must be `width * height * 4` bytes in sRGB, straight from a
     * canvas `getImageData` call. Returns a JS error rather than panicking on
     * bad input, so the page can show a message instead of dying.
     */
    constructor(width: number, height: number, rgba: Uint8Array, triangles: number, iters: number, seed: number);
    /**
     * The current scene rendered as RGBA, ready for `ImageData`.
     */
    render_rgba(): Uint8Array;
    /**
     * The current scene rendered at an arbitrary size, for sharp export.
     */
    render_rgba_at(width: number, height: number): Uint8Array;
    /**
     * The current scene as JSON, in the same format the CLI writes.
     */
    scene_json(): string;
    /**
     * Run up to `n` iterations, returning how many actually ran.
     *
     * Batched because a single iteration is far shorter than a frame, and
     * crossing the wasm boundary once per iteration would cost more than the
     * work itself.
     */
    step_many(n: number): number;
    readonly best_loss: number;
    readonly done: boolean;
    readonly iter: number;
    readonly loss: number;
    readonly sigma: number;
    readonly triangles: number;
}

/**
 * Install a panic hook that reports Rust panics to the browser console.
 *
 * Without this a panic in WebAssembly surfaces as `unreachable executed` with
 * no location, which is close to undebuggable.
 */
export function start(): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_wasmfitter_free: (a: number, b: number) => void;
    readonly wasmfitter_best_loss: (a: number) => number;
    readonly wasmfitter_done: (a: number) => number;
    readonly wasmfitter_iter: (a: number) => number;
    readonly wasmfitter_loss: (a: number) => number;
    readonly wasmfitter_losses: (a: number) => [number, number];
    readonly wasmfitter_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number, number];
    readonly wasmfitter_render_rgba: (a: number) => [number, number];
    readonly wasmfitter_render_rgba_at: (a: number, b: number, c: number) => [number, number, number, number];
    readonly wasmfitter_scene_json: (a: number) => [number, number];
    readonly wasmfitter_sigma: (a: number) => number;
    readonly wasmfitter_step_many: (a: number, b: number) => number;
    readonly wasmfitter_triangles: (a: number) => number;
    readonly start: () => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
