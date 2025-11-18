import torch
import os
import argparse

def compare_runs(run1_dir, run2_dir, steps):
    print(f"Comparing '{run1_dir}' and '{run2_dir}'...")

    for i in range(steps):
        print(f"\n--- Step {i} ---")
        
        # --- Confronto di x_in (input del modello) ---
        file1 = os.path.join(run1_dir, f"x_in_step_{i}.pt")
        file2 = os.path.join(run2_dir, f"x_in_step_{i}.pt")

        if not (os.path.exists(file1) and os.path.exists(file2)):
            print(f"Tensor files for step {i} not found. Stopping.")
            break

        t1_in = torch.load(file1)
        t2_in = torch.load(file2)

        if not torch.equal(t1_in, t2_in):
            diff = (t1_in - t2_in).abs().max()
            print(f"🔴 DIVERGENCE DETECTED in 'x_in' at step {i}!")
            print(f"   Max absolute difference: {diff.item()}")
            return
        else:
            print(f"✅ 'x_in' at step {i} are identical.")

        # --- Confronto di denoised (output del modello) ---
        file1_denoised = os.path.join(run1_dir, f"denoised_step_{i}.pt")
        file2_denoised = os.path.join(run2_dir, f"denoised_step_{i}.pt")
        
        t1_denoised = torch.load(file1_denoised)
        t2_denoised = torch.load(file2_denoised)

        if not torch.equal(t1_denoised, t2_denoised):
            diff = (t1_denoised - t2_denoised).abs().max()
            print(f"🔴 DIVERGENCE DETECTED in 'denoised' at step {i}!")
            print(f"   This means the model function itself is non-deterministic.")
            print(f"   Max absolute difference: {diff.item()}")
            return
        else:
            print(f"✅ 'denoised' at step {i} are identical.")

    print("\n--- Comparison Complete ---")
    print("✅ All tensors in all steps are identical. No divergence found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare debug tensors from two runs.")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps to check.")
    args = parser.parse_args()

    run1_dir = "./debug_run_1"
    run2_dir = "./debug_run_2"

    if not (os.path.exists(run1_dir) and os.path.exists(run2_dir)):
        print("Error: Make sure both './debug_run_1' and './debug_run_2' exist.")
        print("Please run 'amg_infer.py' with DEBUG_RUN_ID=1 and then with DEBUG_RUN_ID=2.")
    else:
        compare_runs(run1_dir, run2_dir, args.steps)