# Runbook: two RTX 4090 boxes

Everything here is meant to be run on your machines — I have no access to them,
so these are instructions rather than something already executed.

Baseline established 2026-08-06 on an RTX 4090 + 26-core i9-13900: forward is up
to **70x** faster than the CPU, and after three rounds of optimization the
batched backward pass wins every cell of the crossover. Raw output is in
`docs/gpu-report.txt`; the README discusses what it means.

## 0. One-time setup, both boxes

```sh
git clone https://github.com/VivianSobers/Differentiable-Rasterizer
cd Differentiable-Rasterizer

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env"

cargo test --release                 # expect 86 passing
cargo run --release --bin gpu_bench  # confirms the GPU path sees the 4090

python -m venv .venv && . .venv/bin/activate
pip install -r python/requirements.txt maturin torch
(cd crates/diffrast-py && maturin develop --release)

python -m unittest discover -s python -p "test_*.py"   # expect 65 passing
```

If `gpu_bench` reports a Vulkan adapter that is not the 4090, force it:

```sh
WGPU_BACKEND=vulkan WGPU_ADAPTER_NAME=4090 cargo run --release --bin gpu_bench
```

## 1. Establish a baseline before changing anything

One command on each box. It builds, tests, benchmarks, and records the hardware
into a single file — read-only apart from this repo's `target/` directory.

```sh
./scripts/collect-gpu-report.sh          # writes docs/gpu-report.txt
```

That file is the number every later change gets compared against, and it takes
about two minutes.

Compare against `docs/gpu-report.txt`. A regression in the forward speedup
usually means the adapter selection picked the wrong device; check the `adapter:`
line names the 4090.

## 2. Splitting work across the two boxes

The two stages have opposite bottlenecks, which is what makes splitting them
worthwhile.

### Box A — dataset generation (CPU-bound)

Fitting is embarrassingly parallel and barely touches the GPU.

```sh
python python/precompute.py \
  --data /path/to/images \
  --out data/fits.pt \
  --tris 128 --iters 800 --fit-size 96 \
  --workers $(nproc)
```

Then copy `data/fits.pt` to box B.

### Box B — supervised training (GPU-bound)

```sh
torchrun --nproc_per_node=2 python/train.py \
  --pretrain data/fits.pt \
  --epochs 60 --batch 128 --lr 3e-4 \
  --out runs/pretrain
```

`--nproc_per_node=2` only makes sense if box B has both cards. If you have one
card per box, use the multi-node form below.

### Both boxes as one job (multi-node DDP)

On the box you designate as rank 0 (call its address `MASTER`):

```sh
# box A
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
  --rdzv_backend=c10d --rdzv_endpoint=MASTER:29500 \
  python/train.py --pretrain data/fits.pt --epochs 60 --batch 64

# box B — identical but --node_rank=1
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=1 \
  --rdzv_backend=c10d --rdzv_endpoint=MASTER:29500 \
  python/train.py --pretrain data/fits.pt --epochs 60 --batch 64
```

Both boxes need the same `fits.pt`, the same commit, and port 29500 reachable
between them. `--batch` is **per rank**, so the effective batch is
`batch x nnodes x nproc_per_node`; scale the learning rate if you change it.

Worth being blunt: for a model this small, two nodes over ethernet will likely
be *slower* than one node with both cards. Gradient all-reduce over a network
costs more than the compute it parallelizes when the model is a few hundred
thousand parameters. The multi-node path is here because you asked how to use
both machines, but if the cards can live in one box, put them there.

### Fine-tuning end to end

After the supervised warm start, fine-tune through the rasterizer:

```sh
torchrun --nproc_per_node=2 python/train.py \
  --data /path/to/images --epochs 20 \
  --init-from runs/pretrain/best.pt \
  --param-weight 0 --raster-device auto \
  --out runs/finetune
```

`--init-from`, not `--resume` — this loads the pretrained weights as a warm
start for a new training phase and nothing else. `--resume` restores the
optimizer and OneCycle schedule too, both defined over the checkpoint's own
step count; the pretrain run already completed every step of *its* schedule,
so resuming into it finds nothing left to anneal and fine-tunes for zero
epochs. This distinction was not academic: the command above used to omit
`--init-from` entirely, silently fine-tuning a freshly initialized model
instead of the pretrained one.

`--param-weight 0` removes the parameter-supervision term, which means the
render loss is the only loss — so leave `--render-fraction` at its default of
1.0 here. Lowering it would discard batch outright rather than trade accuracy
for speed. Raise `--workers` if the cards are idling waiting on data.

## 3. A suggested division of labor

If you want the two boxes doing genuinely different things:

| Box | Job | Why |
| --- | --- | --- |
| A | `precompute.py` on a large corpus, continuously | CPU-bound, wants cores, no GPU contention |
| B | training + fine-tuning | GPU-bound, wants both cards and fast interconnect |

Then A also serves as the evaluation runner: point it at `runs/*/best.pt` and
render comparisons while B keeps training.

## 3b. The twelve-hour sweep

One run answers "does it work". It cannot answer "what makes it better",
because a single point has no slope. `sweep.py` runs several configurations
that differ in one variable at a time, evaluates each with the controls from
`evaluate.py`, and prints a table.

```sh
nohup python python/sweep.py --plan scaling --hours 12 \
    --concurrency 3 --workers 6 --out runs/sweep > runs/sweep.log 2>&1 &

tail -f runs/sweep.log          # progress
nvidia-smi                      # should show three python processes
```

**Concurrency is deliberate.** A 3.7M-parameter model that round-trips the
rasterizer through the host leaves a 4090 substantially idle; three runs share
it far better than one does. `--concurrency` and `--workers` multiply, so
3 x 6 = 18 dataloader workers plus 3 main processes fits 26 cores with room.

Safety properties, because twelve hours is long enough for something to go
wrong:

- Each run is **skipped if already complete** and **resumed from `last.pt` if
  interrupted**, so re-running the same command continues the sweep.
- `--hours` bounds when new runs are *launched*, not when they finish.
- A run that completes every epoch but exits non-zero is kept, with a note.
  That is the shutdown segfault, and it happens after the checkpoints are
  safely written.
- `last.pt` is written to a temporary path and renamed, so a crash mid-save
  cannot leave an unloadable file where the resume path looks for one.

Other plans: `data` (10k/40k/160k scenes), `model` (width 32/64/96),
`triangles` (32/64/128/256), `baseline` (just reproduces the measured run).

## 4. What to send back

Updated after a session that ran most of the outstanding items on this list
directly. What follows is what is genuinely still open, not the original ask.

**Closed this session**, each with a full writeup in
[AMORTIZED.md](AMORTIZED.md):

- The data/compute/triangle/width scaling sweep — data saturates around 160k
  scenes, compute is worth about half of data, width is worth almost nothing.
  Re-running any of these levers wastes the card; they are spent.
- Resolution transfer, both whether it needs training (yes) and whether
  `--sizes` jitter closes the gap once trained (no — narrows it, does not
  close it, at any width tested including 48-160px).
- Real photographs (`ImageFolderDataset` + `precompute.py` + `--pretrain`),
  never exercised before this session. The model generalizes off its own
  generator — margin and gain are the highest recorded anywhere in this
  project on 40k STL-10 photos — but mirror response drops to ~0.59 from
  ~0.90 on synthetic, meaning it leans more on colour and less on layout to
  do it.
- The supervised warm start's actual payoff: pretrain-then-finetune reaches
  92.7% of from-scratch margin for 1.53x less wall-clock, and is not behind
  scratch at all after 100 refinement steps. Getting a clean measurement of
  it surfaced two real bugs (a `--resume` that silently fine-tuned for zero
  epochs; a `--triangles` default that crashed a fine-tune with a checkpoint
  size mismatch), both fixed.
- The browser viewer, unverified since the GPU and model work landed — builds
  clean and now confirmed to actually run: driven headlessly through Chrome's
  DevTools protocol, not just checked for a successful compile.

**Still open:**

1. **The mirror-response gap on real photos** (~0.59-0.60, against ~0.90 on
   synthetic) has no diagnosis yet, only a hypothesis: the head reads one
   flat pooled vector and emits every triangle from it, so a triangle has no
   way to attend to its own region — exactly what a low mirror response
   would predict, and exactly the structural change [AMORTIZED.md] proposed
   before real photos were ever tried. Worth testing directly now that there
   is a real-photo benchmark to test it against, rather than only a synthetic
   one that never exposed the problem this clearly.
2. **`precompute.py`'s `--size` default (64px) does not match the resolution
   used everywhere else in this project (96px)**, and it went unnoticed until
   grading a pretrain checkpoint at 96px cost 3.28 dB of margin that had
   nothing to do with the pretrain approach itself. Either change the
   default or make `train.py --pretrain` warn when the stored image size
   disagrees with `--size`.
3. **The real-photo corpus was run at synthetic-experiment scale** (40k
   images, 60 epochs, 64 triangles, 96px) to fit inside a session, not
   because that is where photos saturate. Whether photos follow the same
   data/compute curve as synthetic scenes — a real knee around 160k, compute
   worth half of data — has not been measured and might not even hold, since
   photos are a fixed corpus rather than an infinite generator.
4. `docs/gpu-report.txt` from `scripts/collect-gpu-report.sh` is a regression
   check now, not an investigation — the crossover favours the 4090 in every
   cell and a backward call at 256px is dispatch-dominated. Worth a fresh run
   whenever the shaders change; not worth investigating further on its own.
