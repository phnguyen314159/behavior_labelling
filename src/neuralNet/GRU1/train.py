#actually run model here
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


from helpers import load_distill_data
from bartDistil import WindowGRU

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training on device: {device} ---")

    BATCH_SIZE = 8
    LEARNING_RATE = 1e-3
    EPOCHS = 20 #read through entire dataset exactly 20 times.

    print("Loading distilled data...")
    try:
        encodings, labels = load_distill_data("distill_data_val.pt")
        print(f"Encodings shape: {encodings.shape}") # Expected: [N, 100, 768]
        print(f"Labels shape: {labels.shape}")       # Expected: [N, 100, 6]
    except FileNotFoundError:
        print("Error: 'distill_data_val.pt' not found. Run your helper.py extraction first!")
        return

    # TensorDataset pairs the input and target tensors together here
    dataset = TensorDataset(encodings, labels)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = WindowGRU(input_dim=768, hidden_dim=256, output_dim=6).to(device)
    
    # Mean Squared Error is standard for regression on continuous label dimensions
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n--- Starting Training ---")
    model.train() # Set model to training mode

    for epoch in range(EPOCHS):
        total_loss = 0.0
        
        for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
            # Move data to the GPU
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            predictions = model(batch_x)

            # Compute loss
            loss = criterion(predictions, batch_y)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print average loss for the epoch
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Average Loss: {avg_loss:.4f}")

    # 7. Save the trained model
    save_path = "gru1_learned_labeler.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining complete! Model saved to {save_path}")

if __name__ == "__main__":
    train_model()