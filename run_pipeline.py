from __future__ import annotations

import argparse
import subprocess
import sys


def _run(module_path: str, extra_args: list[str]) -> int:
    cmd = [sys.executable, module_path] + extra_args
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified pipeline runner")
    parser.add_argument(
        "--phase",
        required=True,
        choices=["validate_data", "visualize_augmentations", "train", "ablate", "resume", "evaluate", "infer", "benchmark", "review_hard_samples"],
    )
    parser.add_argument("--mode", default="rgb", choices=["rgb"])
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--epochs", default="1")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of augmentation samples to visualize")
    parser.add_argument("--num-augs", type=int, default=4, help="Augmentations per sample")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--input", "-i", type=str, help="Input image or directory for inference")
    parser.add_argument("--output", "-o", type=str, default="outputs/predictions", help="Output directory for inference")
    args = parser.parse_args()

    if args.phase == "train":
        cmd_args = ["--epochs", str(args.epochs)]
        if args.dry_run:
            cmd_args.append("--dry-run")
            cmd_args.extend(["--dry-run-device", args.dry_run_device])
        code = _run("training/train.py", cmd_args)
        raise SystemExit(code)

    if args.phase == "resume":
        print("Resume path scaffolded. Use training/train.py with checkpoint loading in Phase 4.")
        raise SystemExit(0)

    if args.phase == "benchmark":
        cmd_args = []
        if args.checkpoint:
            cmd_args.extend(["--checkpoint", args.checkpoint])
        code = _run("inference/benchmark.py", cmd_args)
        raise SystemExit(code)

    if args.phase == "validate_data":
        code = _run("data/validate_dataset.py", ["--write-splits"])
        raise SystemExit(code)

    if args.phase == "visualize_augmentations":
        cmd_args = [
            "--num-samples",
            str(args.num_samples),
            "--num-augs",
            str(args.num_augs),
            "--split",
            args.split,
        ]
        code = _run("data/visualize_augmentations.py", cmd_args)
        raise SystemExit(code)

    if args.phase == "evaluate":
        print("Evaluation phase is scaffolded and will be implemented after training stack matures.")
        raise SystemExit(0)

    if args.phase == "infer":
        if not args.checkpoint:
            print("Error: --checkpoint must be provided for inference.")
            raise SystemExit(1)
        if not hasattr(args, "input") or not args.input:
            print("Error: --input must be provided for inference.")
            raise SystemExit(1)
        
        cmd_args = ["--checkpoint", args.checkpoint, "--input", args.input, "--output", args.output]
        code = _run("inference/predict.py", cmd_args)
        raise SystemExit(code)




    if args.phase == "review_hard_samples":
        if not args.checkpoint:
            print("Error: --checkpoint must be provided for reviewing hard samples.")
            raise SystemExit(1)
        cmd_args = ["--checkpoint", args.checkpoint, "--split", args.split]
        code = _run("inference/review_hard_samples.py", cmd_args)
        raise SystemExit(code)

    if args.phase == "ablate":
        cmd_args = ["--epochs", str(args.epochs)]
        if args.dry_run:
            cmd_args.append("--dry-run")
        code = _run("training/ablate.py", cmd_args)
        raise SystemExit(code)

if __name__ == "__main__":
    main()
