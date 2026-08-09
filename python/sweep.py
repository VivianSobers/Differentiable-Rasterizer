#!/usr/bin/env python3
"""Run a set of training configurations and evaluate every one of them.

    python python/sweep.py --plan scaling --hours 12 --concurrency 3

One 40k-scene run answered "does the amortized model work?" — yes, 6.41 dB of
input gain. It cannot answer "what makes it better", because a single point
does not have a slope. This runs several configurations that differ in one
variable at a time and tabulates what the controls in `evaluate.py` say about
each.

## Why concurrency

The model is a few million parameters and the rasterizer round-trips through
the host, so a single run leaves an RTX 4090 substantially idle. Several runs
share it far better than one does. The limits worth respecting are host RAM and
CPU: each run spawns its own dataloader workers, so `--concurrency` and
`--workers` multiply.

## Safety for a long unattended run

- Each run writes to its own directory and is **skipped if already complete**,
  so re-invoking the sweep continues it rather than restarting it.
- An interrupted run is **resumed** from `last.pt` rather than restarted.
- `--hours` is a deadline for *launching* work, not for finishing it: no new
  run starts past it, and running ones are left to complete.
- Every run's stdout goes to `<out>/<name>/train.log`.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: Configurations, as overrides on top of `BASE`. Each plan changes one thing
#: at a time so the resulting table has interpretable columns.
BASE = dict(
    size=96,
    triangles=64,
    width=32,
    pool=4,
    batch=64,
    epochs=60,
    synthetic_size=40000,
    sigma_start=0.02,
    sigma_end=0.003,
)

PLANS: dict[str, list[dict]] = {
    # The measured run, repeated, so the sweep has a point of comparison
    # against a number that was produced before any of this existed.
    "baseline": [dict(name="baseline")],
    # Does more data keep helping? The 1k-scene local run generalized badly and
    # the 40k run generalized well; the interesting question is where it stops.
    "data": [
        dict(name="data-10k", synthetic_size=10_000),
        dict(name="data-40k", synthetic_size=40_000),
        dict(name="data-160k", synthetic_size=160_000),
    ],
    # Capacity was never the binding constraint at 40k scenes. Whether it
    # becomes one at higher resolution and triangle count is untested.
    "model": [
        dict(name="width-32", width=32),
        dict(name="width-64", width=64),
        dict(name="width-96", width=96),
    ],
    # More triangles is a strictly richer output space and a strictly harder
    # prediction problem. Which effect dominates is an empirical question.
    "triangles": [
        dict(name="tris-32", triangles=32),
        dict(name="tris-64", triangles=64),
        dict(name="tris-128", triangles=128),
        dict(name="tris-256", triangles=256),
    ],
    # Separates dataset *diversity* from training *duration*, which the `data`
    # plan confounds: it holds epochs fixed, so 16x the scenes is also 16x the
    # gradient steps, and "more data helped" cannot be told apart from "more
    # compute helped".
    #
    # Every row here sees exactly 9.6M samples in exactly 150,000 steps, so the
    # optimizer schedule and the compute are identical and the *only* thing
    # that varies is how often the model sees a scene it has seen before —
    # 240 times, 60 times, or 15.
    "diversity": [
        dict(name="div-40k", synthetic_size=40_000, epochs=240),
        dict(name="div-160k", synthetic_size=160_000, epochs=60),
        dict(name="div-640k", synthetic_size=640_000, epochs=15),
    ],
    # Edge softness has never been tuned, and it is the one hyperparameter that
    # plausibly trades one-shot quality against *refinement* quality. A model
    # trained to a sharp final sigma minimizes its own error directly; a
    # blurrier one commits less, which may leave the fitter a better basin to
    # descend. Those are different objectives and the repo has only ever
    # measured the first.
    "sigma": [
        dict(name="sigma-sharp", sigma_end=0.0015),
        dict(name="sigma-mid", sigma_end=0.003),
        dict(name="sigma-soft", sigma_end=0.006),
        dict(name="sigma-flat", sigma_start=0.006, sigma_end=0.006),
    ],
    # Does resolution invariance have to be *trained*, or does adaptive pooling
    # supply it? Measured: it does not supply it. A 128px-trained model lost
    # 5.6 dB of margin when graded at 96px. These arms train across resolutions
    # instead of at one, and are graded at every resolution by `transfer.py`.
    "resolution": [
        dict(name="res-fixed96", size=96),
        dict(name="res-jitter", sizes=[64, 96, 128]),
        dict(name="res-jitter-wide", sizes=[48, 64, 96, 128, 160]),
    ],
    # The headline sweep: more of everything, in the order most likely to pay.
    "scaling": [
        dict(name="data-160k", synthetic_size=160_000),
        dict(name="tris-128", triangles=128),
        dict(name="width-64", width=64),
        dict(name="res-128", size=128),
        dict(name="big", synthetic_size=160_000, triangles=128, width=64, size=128, epochs=80),
    ],
}


def config_for(plan: str, overrides: dict) -> dict:
    config = dict(BASE)
    config.update(overrides)
    return config


def train_command(config: dict, out: Path, workers: int, resume: bool) -> list[str]:
    cmd = [
        sys.executable,
        str(HERE / "train.py"),
        "--synthetic",
        "--synthetic-size", str(config["synthetic_size"]),
        "--epochs", str(config["epochs"]),
        "--batch", str(config["batch"]),
        "--triangles", str(config["triangles"]),
        "--size", str(config["size"]),
        "--width", str(config["width"]),
        "--pool", str(config["pool"]),
        "--sigma-start", str(config["sigma_start"]),
        "--sigma-end", str(config["sigma_end"]),
        "--render-fraction", "1.0",
        "--raster-device", "auto",
        "--workers", str(workers),
        "--out", str(out),
    ]
    if config.get("sizes"):
        cmd += ["--sizes", *[str(v) for v in config["sizes"]]]
    if resume:
        cmd += ["--resume", str(out / "last.pt")]
    return cmd


#: Every run is graded on this one held-out benchmark, whatever it was trained
#: on. Scoring each model against scenes matching its own triangle count and
#: resolution gave each its own exam: absolute PSNR then reflected how hard
#: that model's eval set happened to be, and the mean-colour baseline moved
#: with it, so the columns could not be compared. A model's *own* config is
#: still visible in the table; the yardstick is now fixed.
BENCHMARK = dict(size=96, triangles=64)


def eval_command(
    config: dict, out: Path, count: int, steps: int, size: int, name: str
) -> list[str]:
    return [
        sys.executable,
        str(HERE / "evaluate.py"),
        "--checkpoint", str(out / "best.pt"),
        "--synthetic",
        "--count", str(count),
        "--size", str(size),
        "--eval-triangles", str(BENCHMARK["triangles"]),
        "--refine-steps", str(steps),
        "--out", str(out / name),
    ]


def is_complete(out: Path, config: dict) -> bool:
    """Has this run already finished all its epochs?"""
    history = out / "history.json"
    if not history.exists():
        return False
    try:
        records = json.loads(history.read_text())
    except json.JSONDecodeError:
        return False
    return bool(records) and records[-1]["epoch"] >= config["epochs"] - 1


#: Every training subprocess currently in flight, so signal handlers can reach
#: them. A sweep killed from the outside must not leave orphans behind: they
#: keep training against the same output directory that a later run will resume
#: from, and two writers in one directory corrupts both.
ACTIVE: list[subprocess.Popen] = []


def terminate_children(*_) -> None:
    """Kill every in-flight run's process group, then exit."""
    for process in ACTIVE:
        if process.poll() is not None:
            continue
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

    deadline = time.time() + 20
    for process in ACTIVE:
        try:
            process.wait(timeout=max(0.0, deadline - time.time()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    sys.exit(130)


class Run:
    def __init__(self, config: dict, out: Path, workers: int) -> None:
        self.config = config
        self.out = out
        self.workers = workers
        self.process: subprocess.Popen | None = None
        self.log = None
        self.failed = False

    @property
    def name(self) -> str:
        return self.config["name"]

    def start(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        resume = (self.out / "last.pt").exists()
        self.log = (self.out / "train.log").open("a")
        self.log.write(f"\n=== {'resuming' if resume else 'starting'} {self.name} ===\n")
        self.log.flush()
        self.process = subprocess.Popen(
            train_command(self.config, self.out, self.workers, resume),
            stdout=self.log,
            stderr=subprocess.STDOUT,
            # Its own process group, so the whole tree — trainer plus its
            # dataloader workers — can be signalled as a unit. Killing only
            # the trainer leaves the workers orphaned and still consuming the
            # GPU, which is exactly what happened once.
            start_new_session=True,
        )
        ACTIVE.append(self.process)

    def finish(self) -> None:
        if self.log is not None:
            self.log.close()
            self.log = None
        # The rasterizer's GPU teardown has segfaulted at interpreter exit
        # before, *after* every checkpoint was safely written. Treat a run
        # whose epochs all completed as successful regardless of exit code,
        # and say so rather than hiding it.
        code = self.process.returncode if self.process else -1
        if code != 0 and is_complete(self.out, self.config):
            print(f"  note: {self.name} exited {code} but completed every epoch; keeping it")
        elif code != 0:
            self.failed = True


def run_sweep(args) -> list[Run]:
    plan = PLANS[args.plan]
    root = Path(args.out)
    deadline = time.time() + args.hours * 3600

    pending = [Run(config_for(args.plan, o), root / o["name"], args.workers) for o in plan]

    # Already-trained runs skip *training*, not evaluation. Dropping them here
    # meant a sweep whose arms had all finished in an earlier invocation
    # reported "no evaluations to summarize" and wrote no summary at all — the
    # results were on disk the whole time, and the resume path is precisely
    # when you most want the table.
    active: list[Run] = []
    done: list[Run] = []

    for run in list(pending):
        if is_complete(run.out, run.config):
            print(f"skipping {run.name}: already trained, will still be evaluated")
            pending.remove(run)
            done.append(run)

    while pending or active:
        while pending and len(active) < args.concurrency:
            if time.time() > deadline:
                print(f"deadline reached; not starting {len(pending)} remaining run(s)")
                pending.clear()
                break
            run = pending.pop(0)
            print(f"[{time.strftime('%H:%M:%S')}] starting {run.name}: {run.config}")
            run.start()
            active.append(run)

        if not active:
            break

        time.sleep(args.poll)
        for run in list(active):
            if run.process is not None and run.process.poll() is not None:
                run.finish()
                active.remove(run)
                done.append(run)
                status = "failed" if run.failed else "done"
                print(f"[{time.strftime('%H:%M:%S')}] {run.name} {status}")

    return done


def evaluate_all(runs: list[Run], args) -> None:
    for run in runs:
        if run.failed or not (run.out / "best.pt").exists():
            continue
        print(f"evaluating {run.name}")
        # Twice, because neither grading alone is fair. The shared benchmark
        # makes rows comparable but penalizes any model trained at another
        # resolution — measured: a 128px-trained model scored +4.44 dB of
        # margin on its own resolution and -1.17 dB at 96px, from the same
        # weights. Reporting both separates "worse model" from "worse
        # transfer", which one column cannot.
        subprocess.run(
            eval_command(
                run.config, run.out, args.eval_count, args.refine_steps,
                BENCHMARK["size"], "eval.json",
            ),
            check=False,
        )
        if run.config["size"] != BENCHMARK["size"]:
            subprocess.run(
                eval_command(
                    run.config, run.out, args.eval_count, args.refine_steps,
                    run.config["size"], "eval_native.json",
                ),
                check=False,
            )


def summarize(runs: list[Run], root: Path) -> None:
    rows = []
    for run in runs:
        path = run.out / "eval.json"
        if not path.exists():
            continue
        m = json.loads(path.read_text())
        native = run.out / "eval_native.json"
        m["margin_native_db"] = (
            json.loads(native.read_text())["margin_db"] if native.exists() else m["margin_db"]
        )
        rows.append((run.name, run.config, m))

    if not rows:
        print("\nno evaluations to summarize")
        return

    rows.sort(key=lambda r: r[2].get("margin_db", 0.0), reverse=True)

    print(f"\nall runs scored on one held-out set: "
          f"{BENCHMARK['triangles']} triangles at {BENCHMARK['size']}px\n")
    print(f"{'run':<14}{'tris':>6}{'px':>5}{'width':>7}{'scenes':>9}"
          f"{'margin':>9}{'native':>8}{'gain':>8}{'one-shot':>10}{'mirror':>8}{'refine+':>9}")
    for name, config, m in rows:
        refined = m.get("refined_from_model_psnr")
        scratch = m.get("refined_from_random_psnr")
        advantage = f"{refined - scratch:+.2f}" if refined and scratch else "-"
        print(
            f"{name:<14}{config['triangles']:>6}{config['size']:>5}{config['width']:>7}"
            f"{config['synthetic_size']:>9}"
            f"{m.get('margin_db', float('nan')):>9.2f}"
            f"{m.get('margin_native_db', float('nan')):>8.2f}{m['input_gain_db']:>8.2f}"
            f"{m['one_shot_psnr']:>10.2f}{m['mirror_response']:>8.2f}{advantage:>9}"
        )

    print("\nmargin   = dB over a flat colour fill on the SHARED benchmark. Comparable")
    print("           across rows, but out-of-distribution for other resolutions.")
    print("native   = the same margin at the model's own training resolution.")
    print("           margin << native means the model is fine and does not transfer.")
    print("gain     = dB of one-shot PSNR attributable to reading the input")
    print("mirror   = layout sensitivity; 1.0 fully spatially aware, 0 blind")
    print("refine+  = dB the prediction still leads a random start by, after refinement")

    out = root / "summary.json"
    out.write_text(json.dumps([{"name": n, "config": c, "metrics": m} for n, c, m in rows], indent=2))
    print(f"\nwrote {out}")


def main() -> int:
    args = parse_args()
    if args.plan not in PLANS:
        print(f"unknown plan {args.plan!r}; choose from {', '.join(PLANS)}")
        return 2

    signal.signal(signal.SIGINT, terminate_children)
    signal.signal(signal.SIGTERM, terminate_children)

    started = time.time()
    if args.evaluate_only:
        # Re-grade checkpoints that already exist. Evaluation is minutes where
        # training is hours, so a change to the benchmark should never mean
        # retraining anything.
        runs = [
            Run(config_for(args.plan, o), Path(args.out) / o["name"], args.workers)
            for o in PLANS[args.plan]
        ]
    else:
        runs = run_sweep(args)
    evaluate_all(runs, args)
    summarize(runs, Path(args.out))
    print(f"\ntotal wall time {(time.time() - started) / 3600:.2f} h")
    return 1 if any(r.failed for r in runs) else 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--plan", default="scaling", help=f"one of: {', '.join(PLANS)}")
    p.add_argument("--out", default="runs/sweep")
    p.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="training runs in flight at once; a small model does not fill a 4090 alone",
    )
    p.add_argument("--workers", type=int, default=6, help="dataloader workers *per run*")
    p.add_argument(
        "--hours",
        type=float,
        default=12.0,
        help="stop launching new runs after this long; running ones still finish",
    )
    p.add_argument("--eval-count", type=int, default=256)
    p.add_argument("--refine-steps", type=int, default=100)
    p.add_argument(
        "--evaluate-only",
        action="store_true",
        help="skip training and re-grade the checkpoints already in --out",
    )
    p.add_argument("--poll", type=float, default=10.0, help="seconds between status checks")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
