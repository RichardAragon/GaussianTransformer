import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Hyperparameters
vocab_size = 10000
d_model = 512
nhead = 8
num_layers = 6
dim_feedforward = 2048
learning_rate = 1e-4
batch_size = 32
num_epochs = 10

# Model, optimizer, and loss function
model = GaussianTransformerLLM(vocab_size, d_model, nhead, num_layers, dim_feedforward)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    model.train()
    for src, tgt in train_loader:
        optimizer.zero_grad()
        output, uncertainty = model(src)
        loss = criterion(output.view(-1, vocab_size), tgt.view(-1))
        loss.backward()
        optimizer.step()
    
    # Evaluation and scheduler step
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for src, tgt in val_loader:
            output, uncertainty = model(src)
            loss = criterion(output.view(-1, vocab_size), tgt.view(-1))
            val_loss += loss.item()
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}, Training Loss: {loss.item()}, Validation Loss: {val_loss / len(val_loader)}")

print("Training complete.")
