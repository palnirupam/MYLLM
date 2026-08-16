#!/usr/bin/env python3
"""
scripts/preflight.py — Dhruva V0 Cloud Preflight Check
=======================================================
Usage:
    python scripts/preflight.py
    python scripts/preflight.py --config configs/v0_100m.yaml
"""
import sys, io
# Force UTF-8 on Windows to handle Unicode characters in output
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import sys
import os
import json
import hashlib
import shutil
import tempfile
import argparse
from pathlib import Path

# ── ANSI colors ───────────────────────────────────────────────────────────────
def _green(s): return f"\033[92m{s}\033[0m"
def _red(s):   return f"\033[91m{s}\033[0m"
def _yellow(s):return f"\033[93m{s}\033[0m"

PASS  = _green("  PASS")
FAIL  = _red("  FAIL")
WARN  = _yellow("  WARN")
SKIP  = "  SKIP"


class PreflightChecker:
    def __init__(self, config_path: str = "configs/v0_100m.yaml"):
        self.config_path = config_path
        self.config = None
        self.failures = []
        self.warnings = []

    def _log(self, status: str, check: str, detail: str = ""):
        detail_str = f" — {detail}" if detail else ""
        print(f"{status}  {check}{detail_str}")

    def fail(self, check: str, detail: str = ""):
        self._log(FAIL, check, detail)
        self.failures.append(f"{check}: {detail}")

    def warn(self, check: str, detail: str = ""):
        self._log(WARN, check, detail)
        self.warnings.append(f"{check}: {detail}")

    def ok(self, check: str, detail: str = ""):
        self._log(PASS, check, detail)

    # ── Checks ────────────────────────────────────────────────────────────────

    def check_python(self):
        ver = sys.version_info
        if ver.major < 3 or (ver.major == 3 and ver.minor < 9):
            self.fail("Python version", f"Python >= 3.9 required. Found {sys.version}")
        else:
            self.ok("Python version", f"{sys.version.split()[0]}")

    def check_pytorch(self):
        try:
            import torch
            self.ok("PyTorch", f"v{torch.__version__}")
        except ImportError as e:
            self.fail("PyTorch", f"Not installed: {e}")

    def check_cuda(self):
        try:
            import torch
            if not torch.cuda.is_available():
                self.fail("CUDA", "CUDA not available — GPU training impossible")
                return
            cuda_ver = torch.version.cuda
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.ok("CUDA", f"v{cuda_ver} | {device_name} | {vram_gb:.1f} GB VRAM")

            if vram_gb < 5.9:
                self.warn("VRAM", f"{vram_gb:.1f} GB < 6 GB — batch size may need reduction")
            else:
                self.ok("VRAM", f"{vram_gb:.1f} GB sufficient")
        except Exception as e:
            self.fail("CUDA check", str(e))

    def check_dependencies(self):
        deps = [
            ("safetensors", "safetensors"),
            ("tokenizers", "tokenizers"),
            ("yaml", "PyYAML"),
            ("numpy", "numpy"),
        ]
        for module, package in deps:
            try:
                __import__(module)
                import importlib
                mod = importlib.import_module(module)
                ver = getattr(mod, '__version__', 'unknown')
                self.ok(f"Dependency: {package}", f"v{ver}")
            except ImportError:
                self.fail(f"Dependency: {package}", f"Not installed: pip install {package}")

    def check_config(self):
        if not Path(self.config_path).exists():
            self.fail("Config file", f"Not found: {self.config_path}")
            return
        try:
            import yaml
            with open(self.config_path) as f:
                self.config = yaml.safe_load(f)
            self.ok("Config file", f"Loaded: {self.config_path}")

            # Validate required keys
            required = [("model", "vocab_size"), ("model", "d_model"), ("model", "n_layers"),
                       ("training", "learning_rate"), ("training", "batch_size")]
            for section, key in required:
                val = self.config.get(section, {}).get(key)
                if val is None:
                    self.fail(f"Config field {section}.{key}", "Missing from config")
                else:
                    self.ok(f"Config field {section}.{key}", str(val))

            # Check vocab_size consistency
            model_vs = self.config.get("model", {}).get("vocab_size")
            tok_vs   = self.config.get("tokenizer", {}).get("vocab_size")
            if model_vs and tok_vs and model_vs != tok_vs:
                self.fail("vocab_size consistency",
                          f"model.vocab_size={model_vs} != tokenizer.vocab_size={tok_vs}")
            elif model_vs and tok_vs:
                self.ok("vocab_size consistency", f"{model_vs} == {tok_vs}")

        except Exception as e:
            self.fail("Config parsing", str(e))

    def check_tokenizer(self):
        if self.config is None:
            self.warn("Tokenizer check", "Skipped — config not loaded")
            return

        output_dir = self.config.get("output", {}).get("dir", "./output/v0_100m")
        tokenizer_path = Path(output_dir) / "tokenizer"

        if not tokenizer_path.exists():
            self.warn("Tokenizer", f"Not found at {tokenizer_path} — will be trained at start")
            return

        tok_json = tokenizer_path / "tokenizer.json"
        if not tok_json.exists():
            self.fail("Tokenizer file", f"tokenizer.json missing in {tokenizer_path}")
            return

        # Hash the tokenizer file
        sha256 = hashlib.sha256(tok_json.read_bytes()).hexdigest()[:16]
        self.ok("Tokenizer file", f"Found at {tokenizer_path} | sha256[:16]={sha256}")

        # Verify vocab_size
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from myllm.core.tokenizer.bpe import BPETokenizer
            tok = BPETokenizer.load(str(tokenizer_path))
            expected_vs = self.config.get("model", {}).get("vocab_size")
            if expected_vs and tok.vocab_size != expected_vs:
                self.fail("Tokenizer vocab_size",
                          f"Tokenizer has {tok.vocab_size} but config expects {expected_vs}")
            else:
                self.ok("Tokenizer vocab_size", f"{tok.vocab_size}")
        except Exception as e:
            self.warn("Tokenizer vocab_size check", f"Could not verify: {e}")

    def check_model_build(self):
        if self.config is None:
            self.warn("Model build check", "Skipped — config not loaded")
            return
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from myllm.core.model.config import ModelConfig
            from myllm.core.model.transformer import MyLLMModel
            import torch

            model_cfg_dict = self.config.get("model", {})
            cfg = ModelConfig(**{k: v for k, v in model_cfg_dict.items()
                                if k in ModelConfig.__dataclass_fields__})
            model = MyLLMModel(cfg)

            # Count unique parameters (tied weights counted once)
            counted = set()
            total = 0
            for p in model.parameters():
                if p.data_ptr() not in counted:
                    counted.add(p.data_ptr())
                    total += p.numel()

            self.ok("Model build", f"{total/1e6:.2f}M parameters")

            # Verify weight tying
            if cfg.tie_word_embeddings:
                if model.output_proj.weight.data_ptr() == model.token_embedding.weight.data_ptr():
                    self.ok("Weight tying", "Confirmed shared storage")
                else:
                    self.fail("Weight tying", "output_proj and token_embedding are NOT sharing storage!")

        except Exception as e:
            self.fail("Model build", str(e))

    def check_storage_write(self):
        output_dir = Path(self.config.get("output", {}).get("dir", "./output/v0_100m") if self.config else "./output")
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            test_file = output_dir / ".preflight_write_test"
            test_data = b"preflight_test_1234"
            test_file.write_bytes(test_data)
            read_back = test_file.read_bytes()
            test_file.unlink()

            if read_back != test_data:
                self.fail("Storage write/read", "Data corruption detected in write-read cycle!")
            else:
                self.ok("Storage write/read", f"Write/read verified at {output_dir}")
        except Exception as e:
            self.fail("Storage write/read", str(e))

    def check_disk_space(self):
        output_dir = Path(self.config.get("output", {}).get("dir", "./output") if self.config else "./output")
        try:
            total, used, free = shutil.disk_usage(output_dir.resolve().anchor)
            free_gb = free / 1024**3
            if free_gb < 10.0:
                self.fail("Disk space", f"{free_gb:.1f} GB free < 10 GB required")
            elif free_gb < 20.0:
                self.warn("Disk space", f"{free_gb:.1f} GB free — recommend > 20 GB")
            else:
                self.ok("Disk space", f"{free_gb:.1f} GB free")
        except Exception as e:
            self.warn("Disk space check", str(e))

    def check_checkpoint_dir(self):
        if self.config is None:
            return
        output_dir = Path(self.config.get("output", {}).get("dir", "./output/v0_100m"))
        ckpt_dir = output_dir / "checkpoints"
        try:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            self.ok("Checkpoint dir", f"Writable: {ckpt_dir}")
        except Exception as e:
            self.fail("Checkpoint dir", f"Cannot create: {e}")

    def run_all(self):
        print("\n" + "="*60)
        print("  DHRUVA V0 — CLOUD PREFLIGHT CHECK")
        print("="*60 + "\n")

        print("[ System ]")
        self.check_python()
        self.check_pytorch()
        self.check_cuda()
        self.check_dependencies()

        print("\n[ Configuration ]")
        self.check_config()

        print("\n[ Model ]")
        self.check_model_build()

        print("\n[ Tokenizer ]")
        self.check_tokenizer()

        print("\n[ Storage ]")
        self.check_storage_write()
        self.check_disk_space()
        self.check_checkpoint_dir()

        print("\n" + "="*60)
        if self.failures:
            print(_red(f"RESULT: NO-GO — {len(self.failures)} critical failure(s)"))
            for f in self.failures:
                print(f"  ✗ {f}")
            if self.warnings:
                print(_yellow(f"\n  {len(self.warnings)} warning(s):"))
                for w in self.warnings:
                    print(f"  ⚠ {w}")
            print("="*60 + "\n")
            return 1
        elif self.warnings:
            print(_yellow(f"RESULT: GO (with {len(self.warnings)} warning(s))"))
            for w in self.warnings:
                print(f"  ⚠ {w}")
            print("="*60 + "\n")
            return 0
        else:
            print(_green("RESULT: GO — All checks passed"))
            print("="*60 + "\n")
            return 0


def main():
    parser = argparse.ArgumentParser(description="Dhruva V0 Cloud Preflight Check")
    parser.add_argument("--config", type=str, default="configs/v0_100m.yaml")
    args = parser.parse_args()

    checker = PreflightChecker(config_path=args.config)
    exit_code = checker.run_all()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
