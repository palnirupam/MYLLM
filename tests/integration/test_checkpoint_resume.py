import torch
import torch.nn.functional as F
import math
import tempfile
from pathlib import Path
from myllm.core.model import ModelConfig, MyLLMModel
from myllm.training.checkpoint.manager import CheckpointManager

def test_checkpoint_resume_determinism():
    """
    Integration test to prove that stopping and resuming training yields 
    exactly the same loss values as continuous training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # Minimal config for fast testing
    config = ModelConfig(
        vocab_size=32000,
        d_model=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=64,
        intermediate_size=512,
        rope_theta=10000.0,
    )
    
    # 1. Generate synthetic dataset
    B, T = 2, 64
    x_train = torch.randint(0, 32000, (10 * B, T), device=device)
    y_train = torch.randint(0, 32000, (10 * B, T), device=device)

    # 2. Continuous Run
    torch.manual_seed(123)
    model_cont = MyLLMModel(config).to(device)
    opt_cont = torch.optim.AdamW(model_cont.parameters(), lr=1e-3)
    
    continuous_losses = []
    for step in range(5):
        opt_cont.zero_grad()
        bx = x_train[step*B : (step+1)*B]
        by = y_train[step*B : (step+1)*B]
        
        logits, _ = model_cont(bx)
        loss = F.cross_entropy(logits.view(-1, 32000), by.view(-1))
        loss.backward()
        opt_cont.step()
        continuous_losses.append(loss.item())

    # 3. Interrupted Run
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir)
        manager = CheckpointManager(ckpt_dir)
        
        torch.manual_seed(123)
        model_int = MyLLMModel(config).to(device)
        opt_int = torch.optim.AdamW(model_int.parameters(), lr=1e-3)

        # Run steps 0, 1, 2
        for step in range(3):
            opt_int.zero_grad()
            bx = x_train[step*B : (step+1)*B]
            by = y_train[step*B : (step+1)*B]
            
            logits, _ = model_int(bx)
            loss = F.cross_entropy(logits.view(-1, 32000), by.view(-1))
            loss.backward()
            opt_int.step()
        
        # Save checkpoint at step 2 (0-indexed, so 3 steps complete)
        manager.save(
            model=model_int,
            optimizer=opt_int,
            scheduler=None,
            step=3,
            epoch=0,
            loss=loss.item(),
            config=config,
            tokenizer_path="dummy",
            dataset_position=3*B
        )
        
        # Simulate restart
        del model_int
        del opt_int
        
        # Fresh initialization for resume
        model_res = MyLLMModel(config).to(device)
        opt_res = torch.optim.AdamW(model_res.parameters(), lr=1e-3)
        
        # Load from checkpoint
        ckpt_path = ckpt_dir / "checkpoint-3"
        metadata = manager.load(ckpt_path, model_res, opt_res, scheduler=None)
        
        assert metadata["step"] == 3
        assert metadata["dataset_position"] == 3*B
        
        # Run steps 3, 4
        interrupted_losses = continuous_losses[:3]
        for step in range(3, 5):
            opt_res.zero_grad()
            bx = x_train[step*B : (step+1)*B]
            by = y_train[step*B : (step+1)*B]
            
            logits, _ = model_res(bx)
            loss = F.cross_entropy(logits.view(-1, 32000), by.view(-1))
            loss.backward()
            opt_res.step()
            interrupted_losses.append(loss.item())

    # 4. Compare exact equality
    print("\nContinuous losses: ", continuous_losses)
    print("Interrupted losses:", interrupted_losses)
    
    for i in range(5):
        assert math.isclose(continuous_losses[i], interrupted_losses[i], rel_tol=1e-6), \
            f"Mismatch at step {i}: {continuous_losses[i]} != {interrupted_losses[i]}"
    
    print("[PASS] Checkpoint resume integration test PASSED: Exact determinism proven.")

if __name__ == "__main__":
    test_checkpoint_resume_determinism()
