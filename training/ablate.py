import argparse
import os
import subprocess
import itertools

def main():
    parser = argparse.ArgumentParser(description="Run controlled ablation matrix")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    # Ablation matrix variables
    width_multipliers = [0.5, 1.0]
    boundary_losses = [True, False]
    resolutions = [(256, 512), (384, 768)]
    conf_weighting = [True, False]
    
    # We will log the results of these runs
    combinations = list(itertools.product(width_multipliers, boundary_losses, resolutions, conf_weighting))
    
    log_dir = "outputs/ablations"
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"Starting {len(combinations)} ablation runs...")
    
    for idx, (width, use_boundary, res, conf_w) in enumerate(combinations):
        run_name = f"run_{idx}_w{width}_bound{use_boundary}_res{res[0]}x{res[1]}_confw{conf_w}"
        print(f"\n--- Running Ablation {idx+1}/{len(combinations)}: {run_name} ---")
        
        # Build command depending on how we can override config
        env = os.environ.copy()
        env["ABLATION_WIDTH_MULT"] = str(width)
        env["ABLATION_USE_BOUNDARY"] = str(use_boundary)
        env["ABLATION_IMG_HEIGHT"] = str(res[0])
        env["ABLATION_IMG_WIDTH"] = str(res[1])
        env["ABLATION_RUN_NAME"] = run_name
        
        cmd = ["python", "training/train.py", "--epochs", str(args.epochs)]
        if args.dry_run:
            cmd.append("--dry-run")
        if not conf_w:
            cmd.append("--no-conf-weighting")
            
        proc = subprocess.run(cmd, env=env)
        if proc.returncode != 0:
            print(f"Run {run_name} failed. Skipping remaining.")
            break
            
    print("Ablation matrix complete.")

if __name__ == "__main__":
    main()
