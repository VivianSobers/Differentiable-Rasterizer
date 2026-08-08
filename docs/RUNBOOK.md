# Runbook: two RTX 4090 boxes

Everything here is meant to be run on your machines — I have no access to them,
so these are instructions rather than something already executed.

Baseline established 2026-08-06 on an RTX 4090 + 26-core i9-13900: forward is up
to **71x** faster than the CPU, forward+backward up to **4.7x**. Raw output is in
`docs/gpu-report.txt`; the README discusses what it means.

## 0. One-time setup, both boxes

```sh
git clone https://github.com/VivianSobers/Differentiable-Rasterizer
cd Differentiable-Rasterizer

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env"

cargo test --release                 # expect 72 passing
cargo run --release --bin gpu_bench  # confirms the GPU path sees the 4090

python -m venv .venv && . .venv/bin/activate
pip install -r python/requirements.txt maturin torch
(cd crates/diffrast-py && maturin develop --release)

python -m unittest discover -s python -p "test_*.py"   # expect 43 passing
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
  --render-fraction 0.5 --param-weight 0 \
  --out runs/finetune
```

This stage is CPU-bound today, because the PyO3 layer calls the CPU rasterizer.
Raise `--workers` and lower `--render-fraction` if the GPUs are idling.

## 3. A suggested division of labor

If you want the two boxes doing genuinely different things:

| Box | Job | Why |
| --- | --- | --- |
| A | `precompute.py` on a large corpus, continuously | CPU-bound, wants cores, no GPU contention |
| B | training + fine-tuning | GPU-bound, wants both cards and fast interconnect |

Then A also serves as the evaluation runner: point it at `runs/*/best.pt` and
render comparisons while B keeps training.

## 4. What to send back

Three things, in order of usefulness:

1. `docs/gpu-report.txt` from `scripts/collect-gpu-report.sh`. The GPU work is
   done for now — the crossover favours the 4090 in all nine cells and a
   backward call at 256px is 61% dispatch — so this is a regression check
   rather than an investigation. Worth a fresh run whenever the shaders change,
   and worth re-reading `crossover` if `prefer_gpu`'s thresholds are touched:
   `test_policy_matches_the_discrete_crossover` pins that table and has to move
   with it.
2. **The output of `evaluate.py`** — now the most useful thing on this list,
   and the reason to run training at all. `history.json` shows the loss going
   down, which turns out not to be evidence of anything: a model that ignores
   its input entirely also drives that loss down. See
   [AMORTIZED.md](AMORTIZED.md).

   A real run, and its evaluation:

   ```sh
   python python/train.py --synthetic --epochs 60 --synthetic-size 40000 \
       --batch 64 --triangles 64 --size 96 --width 48 --pool 4 \
       --render-fraction 1.0 --raster-device auto --out runs/pool4

   python python/evaluate.py --checkpoint runs/pool4/best.pt --synthetic \
       --count 256 --refine-steps 100 --out runs/pool4/eval.json
   ```

   `--render-fraction 1.0` is now the default without `--pretrain`, but pass it
   explicitly — an earlier default of 0.25 discarded three quarters of every
   batch here and invalidated a local experiment before it was caught.

   **The two numbers to read** are `input gain` and whether one-shot PSNR
   clears the mean-colour baseline. Locally they were 3.62 dB and *no*, losing
   by 0.17 dB. The capacity test says this is a generalization gap rather than
   a representational limit — the same model memorizes 16 images to within
   0.4 dB of what a direct fit achieves — so more data at higher resolution is
   the treatment, and this run is the test of it.

   Read the result against **27.5 dB**, not against perfection: that is what a
   long direct fit reaches on this data, and no predictor should be expected to
   beat the optimizer it is imitating.

   The `--pool 1` arm is optional and lower value now. It costs the same run
   again to confirm a 2.3 dB capacity gap already measured locally.
3. Peak GPU memory during training (`nvidia-smi` during a run) — tells us how
   much headroom there is for larger batches or more triangles.
