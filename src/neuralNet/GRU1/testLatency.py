import torch
import time
import numpy as np
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import TensorDataset, DataLoader
from src.neuralNet.GRU1.bartDistil import WindowGRU
from src.neuralNet.GRU1.helpers import load_first

def benchmark_fidelity(model_path, batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running Fidelity Benchmark on: {device} ---")
    
    # 1. Load the "Baked" Data
    try:
        # Expected: encodings [N, 100, 768], labels [N, 100, 6], deltas [N, 100]
        encodings, labels, deltas = load_first()
        dataset = TensorDataset(encodings, labels, deltas)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        print(f"Loaded {len(dataset)} sequence samples.")
    except Exception as e:
        print(f"Error loading: {e}")
        return

    # 2. Load the Frozen Stage 1 Model
    model = WindowGRU(input_dim=768, hidden_dim=256, output_dim=6).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 3. Metrics Setup
    criterion = torch.nn.MSELoss()
    all_preds, all_targets, latencies = [], [], []
    total_loss = 0.0

    print("Processing sequences and measuring latency...")
    with torch.no_grad():
        for batch_x, batch_y, delta in dataloader:
            batch_x, batch_y, delta = batch_x.to(device), batch_y.to(device), delta.to(device)

            # Benchmark p95 Latency (Inference Only)
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            preds = model(batch_x, delta)
            
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)

            # Collect results for Fidelity tests
            total_loss += criterion(preds, batch_y).item()
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    # 4. Statistical Post-processing
    # Reshape to [Total Samples, 6] for per-dimension correlation
    all_preds = np.concatenate(all_preds, axis=0).reshape(-1, 6)
    all_targets = np.concatenate(all_targets, axis=0).reshape(-1, 6)
    
    avg_mse = total_loss / len(dataloader)
    p95_latency = np.percentile(latencies, 95)
    
    # 5. Output Final Report
    print("\n" + "="*50)
    print(f"GLOBAL PERFORMANCE RESULTS")
    print(f"Average MSE Loss: {avg_mse:.6f}")
    print(f"p95 Inference Latency: {p95_latency:.2f} ms")
    print("="*50)
    
    dimensions = ['Logic', 'Perception', 'Knowledge', 'Fear', 'Desire', 'Stress']
    print(f"{'Dimension':<12} | {'Pearson (r)':<12} | {'Spearman (rho)':<12} | {'Status'}")
    print("-" * 65)

    for i, name in enumerate(dimensions):
        r_score, _ = pearsonr(all_targets[:, i], all_preds[:, i])
        rho_score, _ = spearmanr(all_targets[:, i], all_preds[:, i])
        
        # Verify against your 80% (0.80) fidelity gate
        status = "PASS" if (r_score >= 0.80 and rho_score >= 0.80) else "FAIL"
        print(f"{name:<12} | {r_score:<12.4f} | {rho_score:<12.4f} | {status}")
    print("="*65)

if __name__ == "__main__":
    benchmark_fidelity("gru1_learned_labeler.pth")