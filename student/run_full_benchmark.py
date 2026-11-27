from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from train_student import train_and_export

ROOT = Path(__file__).resolve().parent.parent
STUDENT_TOOL = ROOT / "build-release/engine/student_tool"
RESULTS_ROOT = ROOT / "results/student_benchmark"


def step(num: int, title: str) -> None:
    print(f"\n[Step {num}/4] {title}", flush=True)


def run_cpp(*args: str) -> None:
    cmd = [str(STUDENT_TOOL), *args]
    print(f"  $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def prepare_and_train(
    split: str, seed: int, data_dir: Path, model_path: Path, train_seed: int, skip_export: bool
) -> None:
    if not skip_export:
        step(1, f"C++ export features + Teacher ({split}) -> {data_dir.name}")
        if not (data_dir / "teacher_kept_0.02.bin").exists():
            run_cpp("export", "--split", split, "--seed", str(seed), "--out-dir", str(data_dir))
        else:
            print("  (cached, skipping)")

        step(2, f"C++ fit-simplex Stage-2 labels ({split}) -> {data_dir.name}")
        if not (data_dir / "preset_losses_0.02.bin").exists():
            run_cpp("fit-simplex", "--split", split, "--seed", str(seed), "--out-dir", str(data_dir))
        else:
            print("  (cached, skipping)")

    step(3, f"Python train MLPs -> {model_path.name}")
    if not model_path.exists():
        train_and_export(data_dir, model_path, seed=train_seed)
    else:
        print("  (cached, skipping)")


def run_seed(seed: int, csv_path: Path, skip_export: bool) -> None:
    seed_dir = RESULTS_ROOT / f"seed_{seed}"
    if (seed_dir / "DONE").exists():
        print(f"[Seed {seed}] already complete, skipping.")
        return

    print(f"\n=== Seed {seed}: 2-Fold Teacher-Isolated OOF (full 85/15 GeoLife) ===")
    half0_dir, half1_dir, full_dir = seed_dir / "half0", seed_dir / "half1", seed_dir / "full"
    model_for_fold0 = seed_dir / "model_from_half1.bin"
    model_for_fold1 = seed_dir / "model_from_half0.bin"

    prepare_and_train("half0", seed, half0_dir, model_for_fold1, seed, skip_export)
    prepare_and_train("half1", seed, half1_dir, model_for_fold0, seed + 1, skip_export)

    if not skip_export and not (full_dir / "teacher_kept_0.02.bin").exists():
        run_cpp("export", "--split", "full", "--seed", str(seed), "--out-dir", str(full_dir))

    step(4, f"Pure-C++ 2-Fold OOF inference & evaluation (seed {seed})")
    run_cpp(
        "eval",
        "--split",
        "full",
        "--seed",
        str(seed),
        "--model",
        f"{model_for_fold0},{model_for_fold1}",
        "--teacher-dir",
        str(full_dir),
        "--baselines",
        "--csv",
        str(csv_path),
    )
    (seed_dir / "DONE").write_text("ok\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="7,99,123,2024,777")
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip Steps 1-2 (assume features/Teacher already exported)",
    )
    args = parser.parse_args()

    csv_path = RESULTS_ROOT / "benchmark_results.csv"
    for seed in [int(value) for value in args.seeds.split(",")]:
        run_seed(seed, csv_path, args.skip_export)


if __name__ == "__main__":
    main()
