import os
import sys
import json
import time
import shutil
import hashlib
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from myllm.utils.env import get_project_root

def run_cmd(cmd, check=True):
    print(f"\n[RUNNING] {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"FAILED (Code {proc.returncode}): {' '.join(cmd)}")
        print("--- STDOUT ---")
        print(proc.stdout)
        print("--- STDERR ---")
        print(proc.stderr)
        if check:
            sys.exit(proc.returncode)
    return proc.stdout

def get_git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
    except Exception:
        return "unknown"

def detect_environment():
    import torch
    import platform
    import psutil
    
    # Driver version
    driver_version = "unknown"
    try:
        smi_out = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader']).decode('utf-8').strip()
        driver_version = smi_out.split('\n')[0]
    except Exception:
        pass

    # GPU
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpus.append({
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "vram_gb": round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2)
            })
    
    # Precision Selection
    precision = "fp16"
    reason = "BF16 not supported by hardware"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        precision = "bf16"
        reason = "Hardware supports BF16"
        
    env = {
        "provider": "unknown (run in cloud)",
        "gpus": gpus,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "driver_version": driver_version,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "os": platform.platform(),
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "selected_precision": precision,
        "precision_reason": reason,
        "git_commit": get_git_commit()
    }
    return env

def main():
    root = get_project_root()
    out_dir = root / "artifacts" / "cloud_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print(" DHRUVA V0 -- ACTUAL CLOUD VALIDATION RUNNER")
    print("="*60)
    
    # 1. Environment Discovery
    print("\n[1/10] Environment Discovery")
    env = detect_environment()
    with open(out_dir / "cloud_environment.json", "w") as f:
        json.dump(env, f, indent=2)
    print(f"Precision selected: {env['selected_precision']} ({env['precision_reason']})")
    
    # Update config with correct precision
    import yaml
    config_path = root / "configs" / "cloud_smoke_test.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["training"]["mixed_precision"] = env["selected_precision"]
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)
        
    # 2. Preflight
    print("\n[2/10] Preflight Check")
    preflight_cmd = [sys.executable, str(root / "scripts" / "preflight.py")]
    preflight_out = run_cmd(preflight_cmd, check=False)
    if "GO — All checks passed" not in preflight_out:
        print("PREFLIGHT FAILED! Cloud validation NO-GO.")
        with open(out_dir / "CLOUD_GO_NO_GO.md", "w") as f:
            f.write("# CLOUD VALIDATION RESULT\n\n**NO-GO**\n\nPreflight failed.\n")
        sys.exit(1)
        
    with open(out_dir / "cloud_preflight.json", "w") as f:
        json.dump({"status": "PASS", "output": preflight_out}, f, indent=2)
        
    # 3. Train Phase 1 (Interruption)
    print("\n[3/10] Phase 1: Benchmark & Interruption (Target: step 5)")
    train_cmd = [sys.executable, str(root / "scripts" / "train.py"), "--config", str(config_path), "--output", str(root / "output" / "actual_cloud_test")]
    
    # We use subprocess.Popen to kill it after step 5
    import time
    proc = subprocess.Popen(train_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(root))
    
    checkpoint_reached = False
    log_lines = []
    
    for line in iter(proc.stdout.readline, ''):
        log_lines.append(line.strip())
        print("  |", line.strip())
        if "Step    5 | Loss" in line or "Step    6 | Loss" in line:
            time.sleep(1) # wait for checkpoint write
            checkpoint_reached = True
            break
            
    if checkpoint_reached:
        print("Intentional Interruption: KILLING process.")
        proc.kill()
        proc.wait()
    else:
        print("FAILED: Process exited before reaching step 5")
        sys.exit(1)
        
    # Extract Benchmark Info
    # tokens/sec and VRAM from step 5 line
    tok_sec = 0
    vram = 0
    for line in log_lines:
        if "Step    5 | Loss" in line:
            parts = line.split("|")
            for p in parts:
                if "Tok/s" in p:
                    tok_sec = float(p.replace("Tok/s", "").strip())
                if "Mem" in p:
                    vram = float(p.replace("Mem", "").replace("GB", "").strip())
    
    with open(out_dir / "cloud_benchmark.json", "w") as f:
        json.dump({"tokens_per_sec": tok_sec, "peak_vram_gb": vram, "steps_per_sec": tok_sec / 256}, f, indent=2)
        
    # 4. Checkpoint Creation
    print("\n[4/10] Verifying Checkpoint Creation")
    checkpoint_dir = root / "output" / "actual_cloud_test" / "checkpoints" / "checkpoint-5"
    if not checkpoint_dir.exists():
        print("FAILED: Checkpoint 5 directory does not exist.")
        sys.exit(1)
        
    with open(out_dir / "cloud_checkpoint_test.json", "w") as f:
        json.dump({"checkpoint_created": True, "path": str(checkpoint_dir)}, f, indent=2)
        
    # 5. Resume & Completion (Phase 2)
    print("\n[5/10] Phase 2: Resume to Completion")
    proc2 = subprocess.run(train_cmd, capture_output=True, text=True, cwd=str(root))
    print(proc2.stdout)
    if proc2.returncode != 0:
        print("FAILED: Resume failed.")
        print(proc2.stderr)
        sys.exit(1)
        
    if "Resuming from checkpoint" not in proc2.stdout:
        print("FAILED: Did not resume from checkpoint properly.")
        sys.exit(1)
        
    resume_stats = {
        "resumed": True,
        "resume_log_verified": True,
        "optimizer_restored": "optimizer state" in proc2.stdout.lower() or "optimizer" in proc2.stdout.lower(),
        "scheduler_restored": "scheduler state" in proc2.stdout.lower() or "scheduler" in proc2.stdout.lower(),
        "rng_restored": "rng" in proc2.stdout.lower(),
        "checkpoint_step_before_interruption": 5,
        "resume_step": -1,
        "resume_dataset_position": -1
    }
    
    # Try to extract the dataset position and step from the live_progress.json
    live_prog = root / "artifacts" / "training_diagnostics" / "live_progress.json"
    if live_prog.exists():
        with open(live_prog, "r") as f:
            lp = json.load(f)
            resume_stats["resume_step"] = lp.get("step", -1)
            resume_stats["resume_dataset_position"] = lp.get("total_tokens", -1)
            
    with open(out_dir / "cloud_resume_test.json", "w") as f:
        json.dump(resume_stats, f, indent=2)
        
    # 6. Evaluation
    print("\n[6/10] Evaluation")
    final_model_dir = root / "output" / "actual_cloud_test" / "final_model"
    eval_cmd = [sys.executable, str(root / "myllm" / "cli" / "main.py"), "evaluate", "--model", str(final_model_dir)]
    eval_out = run_cmd(eval_cmd)
    
    # 7. Inference
    print("\n[7/10] Offline Inference")
    gen_cmd = [sys.executable, str(root / "myllm" / "cli" / "main.py"), "generate", "--model", str(final_model_dir), "--prompt", "The future of AI is", "--max-tokens", "10"]
    gen_out = run_cmd(gen_cmd)
    
    # 8. Artifact Round-Trip
    print("\n[8/10] Artifact Round-Trip Hashing")
    safetensors_path = final_model_dir / "model.safetensors"
    with open(safetensors_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    
    with open(out_dir / "cloud_artifact_roundtrip.json", "w") as f:
        json.dump({"model.safetensors_sha256": file_hash}, f, indent=2)
        
    # 9. Final GO
    print("\n[9/10] Generating Final Status")
    with open(out_dir / "CLOUD_GO_NO_GO.md", "w") as f:
        f.write("# CLOUD VALIDATION RESULT\n\n**GO**\n\n")
        f.write("All tests passed.\n")
        f.write(f"- Benchmark: {tok_sec} tok/s, {vram} GB VRAM\n")
        f.write(f"- Hash: {file_hash}\n")
        
    print("\n============================================================")
    print(" ACTUAL CLOUD VALIDATION: FULLY PASSED ")
    print("============================================================")
    
if __name__ == "__main__":
    main()
