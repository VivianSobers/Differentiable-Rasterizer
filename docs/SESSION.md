# Session summary — 2026-08-10/11

> **Correction, verified 2026-08-11.** As first written this file cited three
> commit hashes that do not exist in the repository (`5b1751e`, `1e9f2ce`,
> `71bc01e`), gave the commit count as eleven when it is twelve, and named a
> final commit the `v0.3.0` tag does not point at. The hashes below are the
> real ones. Everything else here was re-checked against the repository and
> holds: the code, the tests (86 Rust, 65 Python, clippy and fmt clean),
> `docs/gpu-report.txt`, `python/export_stl10.py` and `--init-from` are all
> present and pass. The experimental results were unaffected — this was
> provenance metadata written from memory rather than from `git log`, which is
> exactly the failure mode the rest of these documents exist to guard against.

Overnight autonomous session on a shared RTX 4090 box. Started from `6139e20`
(README + AMORTIZED.md up to date through the diversity/refinement-rate
sweeps), ended at `bc1380f`, tagged `v0.3.0`. Twelve commits, all real —
nothing padded or split to inflate the count.

## What was run, and what each result was

**Baseline.** Rebuilt `diffrast_rs` bindings (`maturin build --release`,
`pip install`) since the last build predated the pulled commits. 86 Rust
tests, 65 Python tests, `cargo clippy` and `cargo fmt --check` all clean
before touching anything.

**Phase 1 — resolution jitter, re-run.** The prior run was invalid: checkpoint
selection compared training losses across different resolutions. Selection
now runs on a held-out set at one fixed resolution (already fixed, `42a86e6`,
before this session). Re-running `--sizes` with `--workers 4` surfaced a real
deadlock, not present in the prior invalid run: `--sizes` re-forks its
DataLoader every epoch, and by then the main process has already used the GPU
rasterizer — every training step does — so the fork inherits a half-locked
wgpu context and hangs on the very first batch. Reproduced reliably with a
30-second smoke test, fixed (`1de5972`) by forking workers once, before
anything touches the GPU, and passing per-epoch resolution changes through a
`multiprocessing.Value` instead of by re-forking.

Result, three arms graded at 64/96/128px: jitter training helps — closes 1.72
dB of margin at 64px, 0.35 dB at 128px, versus training at 96px alone — but
does **not** close the gap. Every arm, including a 48-160px jitter, still
loses to a flat colour fill away from ~96px. Narrower jitter (64/96/128) beat
wider jitter (48-160) at both off-resolution points, likely a training-density
effect rather than a range effect. `model.py`'s design note corrected.

**Phase 2 — real photographs, run for the first time.** Every prior claim
about the amortized model was measured on synthetic scenes rendered by the
same rasterizer it trains against — inverting a renderer, not approximating
an image. 40k STL-10 images exported via a new `python/export_stl10.py`,
fitted with `precompute.py` (40,000/40,000, ~3.12h), trained from scratch (60
epochs, 96px, 64 triangles, 553.8s wall-clock, PSNR 13.5 -> 19.3 dB).

Evaluated: margin **5.88 dB** and input gain **9.20 dB** — both the highest
recorded anywhere in this project, ahead of every synthetic configuration.
Verdict: `learned a real mapping`. Two predictions did not survive contact
with the data: the mean-colour baseline was expected to be *stronger* on
photos and was measured *weaker* (13.35 dB against 18.68 dB synthetic), and
mirror response — expected to be comparable — fell to **0.59** from 0.87-0.92
on every synthetic configuration. The model generalizes off its own
generator, genuinely, but leans more on colour statistics and less on precise
spatial layout to do it on real photographs than it does on synthetic scenes.

**Phase 3 — the supervised warm start, evaluated with controls for the first
time.** `--pretrain` had never been evaluated end to end. Getting a clean
comparison surfaced two real bugs: the documented `--resume`-based fine-tune
command silently fine-tuned for zero epochs (`--resume` restores the OneCycle
schedule too, and the pretrain run had already completed every step of its
own schedule, so there was nothing left to anneal); and the first fine-tune
attempt crashed outright from a `--triangles` default mismatch against the
checkpoint's head layer. Added `--init-from` (`7092e77`) — loads weights only,
starts a fresh optimizer/schedule/epoch-counter — and fixed the RUNBOOK
example that never actually wired the pretrained weights in.

Three arms, all graded at matched 96px:

| arm | wall-clock | margin | gain | mirror | one-shot | refined |
| --- | --- | --- | --- | --- | --- | --- |
| scratch (60 ep) | 553.8s | 5.88 | 9.20 | 0.59 | 19.23 | 23.50 |
| pretrain only (60 ep) | 145.4s | 2.00 | 4.39 | 0.60 | 15.35 | 23.39 |
| pretrain + finetune (20 ep) | 362.8s | 5.45 | 8.69 | 0.60 | 18.80 | **23.62** |

The pretrain-only row is graded 1.5x out of its native resolution —
`precompute.py` stores images at its default 64px, not the 96px used
everywhere else, an unnoticed mismatch this table exposed. Graded at its
native 64px it scores 5.28 dB margin, competitive with scratch. The warm start
pays for itself: pretrain+finetune reaches 92.7% of scratch's margin for
1.53x less wall-clock, and is not behind scratch at all after refinement —
23.62 dB against 23.50 dB, the best number in the table.

**Phase 4 — browser viewer, verified for the first time since the GPU and
model work landed.** Installed `wasm-pack`, built the wasm package and the
TypeScript bundle, both clean — the `diffrast_wasm_bg.wasm` binary changed by
126 bytes (rebuilt against the current crate) but the exported API is
byte-identical to what was committed, no drift. Verified it actually *runs*,
not just compiles: drove a real headless Chrome instance over the DevTools
protocol, clicked Start, watched iteration 32 -> 76 and loss 2.55e-3 -> 1.90e-3
over six seconds, zero console errors. `cargo test -p diffrast-wasm` (9
tests) and `stepping_matches_the_batch_loop` both pass.

**Phase 5 — consolidation.** `docs/gpu-report.txt` regenerated — GPU idle,
no regression against the README's existing tables (this file existed on
disk from a prior session but had never actually been committed; it is now).
README, AMORTIZED.md and RUNBOOK.md updated with everything above. Python
test count in RUNBOOK corrected (61 -> 65, stale). RUNBOOK's "what to send
back" section rewritten to describe what is genuinely still open rather than
tasks this session already closed.

## An operational mess, worth recording honestly

Managing three long-running background jobs (a two-run sweep, a 3.7-hour CPU
precompute, and later a fine-tune) through repeated background/foreground
tool calls went wrong twice, independently of the actual experiments:

1. **CPU oversubscription.** Neither `precompute.py`'s worker pool nor the
   sweep's DataLoader workers capped PyTorch's per-process thread count.
   Eight dataloader workers defaulting to 24 OpenMP threads each drove the
   load average to 125+ on a 26-core box — a real cost to other tenants on a
   shared machine, not just to this session's own throughput. Fixed by
   setting `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `RAYON_NUM_THREADS=1` on
   every subsequent launch.
2. **Duplicate process launches.** A `pgrep -f` pattern with a misplaced
   trailing `$` anchor (matching a substring that wasn't actually at the end
   of the command line) produced false "process is dead" readings, which led
   to launching a second `sweep.py` on top of a first one — twice — both
   writing into the same output directories. No data was corrupted (`last.pt`
   writes are atomic, RUNBOOK's own design), but real training time was
   wasted and had to be untangled with careful `ps`-based process-group
   inspection rather than pattern matching.

Both are now understood and avoided for the rest of the session; recorded
here because a wasted hour tracking down a shell scripting bug is a cost as
real as an invalid experiment, even if it doesn't belong in AMORTIZED.md.

## What was not finished, and why

- **The mirror-response gap on real photos (0.59-0.60 vs 0.87-0.92 synthetic)
  has no diagnosis, only a hypothesis** — the head reads one flat pooled
  vector and emits every triangle from it, which is exactly what a low mirror
  response would predict. Testing a per-triangle-attention head is a real
  architecture change, not a measurement, and did not fit inside this
  session's scope of "measure what exists," so it was left as the open item
  in RUNBOOK.md rather than attempted without time to validate it properly.
- **The real-photo corpus was run at a modest scale** (40k images, 60 epochs)
  to fit the session, not because that is where photos saturate. Whether
  photos follow the same data/compute scaling curve measured for synthetic
  scenes (knee around 160k, compute worth half of data) is unmeasured and
  might not even hold — photos are a fixed corpus, not an infinite generator.
- **`precompute.py`'s `--size` default (64px) does not match the resolution
  used everywhere else (96px)**, and it went unnoticed until it cost 3.28 dB
  of margin in the pretrain evaluation. Documented in RUNBOOK.md rather than
  silently changed — changing a default at the end of a long session without
  time to check every path that depends on it seemed like the wrong kind of
  confidence to have.

## The single most important open question

**Does the model's real-photo behavior — high margin and gain, but mirror
response nearly half of the synthetic figure — mean it is finding a
genuinely different, colour-led strategy on photos, or is it the pooled
architecture's known layout-blindness (already diagnosed once, on synthetic
data, as `AdaptiveAvgPool2d(1)`'s spatial-permutation invariance) showing up
again in a form that pooling to 4x4 did not fully fix?** The synthetic
capacity test showed `pool=4` reaches the fitter's own quality when it only
has to memorize — so this is not obviously a representational ceiling. If a
per-triangle-attention head closes the mirror-response gap on photos the way
`pool=4` closed the memorization gap over `pool=1`, that is the next real
result. If it does not, the honest conclusion is that colour statistics are
simply a stronger, easier signal on natural images than on procedurally
generated triangle scenes, and no architecture change will fully close it.
