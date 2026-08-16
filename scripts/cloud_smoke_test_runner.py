import os
import sys
import subprocess
import time
import shutil
from pathlib import Path

def run_test():
    print("="*60)
    print(" DHRUVA V0 -- CLOUD SMOKE TEST SEQUENCE")
    print("="*60)
    
    out_dir = Path("output/cloud_smoke_test")
    if out_dir.exists():
        shutil.rmtree(out_dir)
        print("Cleaned previous output.")
        
    print("\n[1/5] Starting Phase 1: Train until checkpoint (Step 5)")
    cmd = [sys.executable, "scripts/train.py", "--config", "configs/cloud_smoke_test.yaml"]
    
    # Run process
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    checkpoint_reached = False
    for line in iter(proc.stdout.readline, ''):
        print("  |", line.strip())
        if "Step    5 | Loss" in line or "Step    6 | Loss" in line:
            # Step 5 completed (and possibly step 6 started), checkpoint 5 should be saved
            time.sleep(1)
            checkpoint_reached = True
            break
            
    if not checkpoint_reached:
        print("FAILED: Did not reach step 5 checkpoint.")
        proc.kill()
        sys.exit(1)
        
    print("\n[2/5] Intentional Interruption: KILLING process.")
    proc.terminate()
    proc.wait()
    print("Process killed. State: Interrupted at step 5.")
    
    print("\n[3/5] Starting Phase 2: Resume training (Step 5 -> 10)")
    # Start again, should automatically pick up from checkpoint-5
    proc2 = subprocess.run(cmd, capture_output=True, text=True)
    
    print("Resume output tail:")
    lines = proc2.stdout.split('\n')
    for line in lines[-15:]:
        if line.strip():
            print("  |", line)
            
    if proc2.returncode != 0:
        print(f"FAILED: Resume run exited with {proc2.returncode}")
        sys.exit(1)
        
    if "Resuming from checkpoint" not in proc2.stdout:
        print("WARN: 'Resuming from checkpoint' string not found in output, check if it actually resumed.")
    
    print("\n[4/5] Evaluation")
    eval_cmd = [sys.executable, "myllm/cli/main.py", "evaluate", "--model", "output/cloud_smoke_test/final_model"]
    proc_eval = subprocess.run(eval_cmd, capture_output=True, text=True)
    print(proc_eval.stdout)
    if proc_eval.returncode != 0:
        print("FAILED: Evaluation step failed")
        print(proc_eval.stderr)
        sys.exit(1)
        
    print("\n[5/5] Local Offline Inference")
    gen_cmd = [sys.executable, "myllm/cli/main.py", "generate", "--model", "output/cloud_smoke_test/final_model", "--prompt", "The future of AI is", "--max-tokens", "10"]
    proc_gen = subprocess.run(gen_cmd, capture_output=True, text=True)
    print("Generation Output:")
    print(proc_gen.stdout)
    
    if proc_gen.returncode != 0:
        print("FAILED: Inference step failed")
        print(proc_gen.stderr)
        sys.exit(1)

    print("="*60)
    print(" CLOUD SMOKE TEST: FULLY PASSED ")
    print("="*60)

if __name__ == "__main__":
    run_test()
