#actually run model here
from scipy.stats import pearsonr, spearmanr
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Import your custom modules
# Adjust these import paths if your folder structure is slightly different
from src.neuralNet.GRU1.helpers import load_first, load_all
from src.neuralNet.GRU1.bartDistil import WindowGRU

def train_model(training: bool):
    # 1. Setup Device (Utilize that RTX 3060 we just fixed!)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training on device: {device} ---")

    # 2. Hyperparameters
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-3
    EPOCHS = 20
    BASE_DIR = Path(__file__).resolve().parent

    # 3. Load Data
    print("Loading distilled data...")
    if not training:
        try:
            encodings, labels, deltas = load_first() #default only first book
            print(f"Encodings shape: {encodings.shape}") # Expected: [N, 100, 768]
            print(f"Labels shape: {labels.shape}")       # Expected: [N, 100, 6]
        except FileNotFoundError:
            print("Error: 'distill_data.pt' not found. Run your helper.py extraction first!")
            return
        dataset = TensorDataset(encodings, labels, deltas)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        sample=len(dataloader)

    model = WindowGRU(input_dim=768, hidden_dim=256, output_dim=6).to(device)

    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n--- Starting Training ---")
    model.train() # Set model to training mode

    least_loss = 0
    for epoch in range(EPOCHS):
        total_loss = 0.0 
        #epoch will connsist of all batches per each book
        if training:
            data_gen = load_all()
            for encodings, labels, deltas in data_gen:
                sample+=len(dataloader)
                dataset = TensorDataset(encodings, labels, deltas)
                dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
            
                for batch_idx, (batch_x, batch_y, delta) in enumerate(dataloader):
                    # Move data to the GPU
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    delta = delta.to(device)

                    # Zero the gradients
                    optimizer.zero_grad()

                    # Forward pass
                    predictions = model(batch_x, delta)

                    # Compute mse
                    loss = criterion(predictions, batch_y)

                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                
        else: 
            for batch_idx, (batch_x, batch_y, delta) in enumerate(dataloader):
                    # Move data to the GPU
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    delta = delta.to(device)

                    # Zero the gradients
                    optimizer.zero_grad()

                    # Forward pass
                    predictions = model(batch_x, delta)

                    # Compute mse
                    loss = criterion(predictions, batch_y)

                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

        # Print average loss for the epoch
        avg_loss = total_loss / sample
        current_alpha = model.alpha.item()            
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Average Loss: {avg_loss:.4f} | Alpha: {current_alpha:.4f}")
        model.eval() #not training
        all_preds = []
        all_targets = []

        with torch.no_grad(): #no backprop
            for batch_x, batch_y, delta in dataloader:
                preds = model(batch_x.to(device), delta.to(device)) #prefiction - forward pass
                all_preds.append(preds.cpu().numpy())
                all_targets.append(batch_y.numpy())
        #in cur implementations, we dont care of char-pivot at this point (not train to learn character)
        all_preds = np.concatenate(all_preds, axis=0).reshape(-1, 6)
        all_targets = np.concatenate(all_targets, axis=0).reshape(-1, 6)

        print("\n--- Distillation Fidelity (Correlation) ---")
        dimensions = ['Logic', 'Perception', 'Knowledge', 'Fear', 'Desire', 'Stress']
        for i, name in enumerate(dimensions):
            p_corr, _ = pearsonr(all_targets[:, i], all_preds[:, i])
            s_corr, _ = spearmanr(all_targets[:, i], all_preds[:, i])
            print(f"{name} - Pearson: {p_corr:.4f}, Spearman: {s_corr:.4f}")

        model.train()
        
    save_path = "gru1_learned_labeler.pth"
    torch.save(model.state_dict(), save_path)
    print("\nTraining complete! Model saved")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser() 
    parser.add_argument("-t", "--training", action="store_true")
    parser.add_argument("-v", "--validating", action="store_true")
    args = parser.parse_args()

    if args.training:
        train_model(True)
    if args.validating:
        train_model(False)