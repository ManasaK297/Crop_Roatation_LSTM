import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import json
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# Load data
with open("crop_lstm_dataset.pkl", "rb") as f:
    data = pickle.load(f)
    if len(data) == 3:
        X, y, X_features = data
        print("✅ Loaded dataset with features")
    else:
        X, y = data
        X_features = None
        print("⚠️ Features ignored for basic model")

with open("vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

crop2idx = vocab["crop2idx"]
idx2crop = {int(k): v for k, v in vocab["idx2crop"].items()}
vocab_size = len(crop2idx)

# Hyperparameters
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
BATCH_SIZE = 32
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🔧 Using device: {DEVICE}")

class CropDataset(Dataset):
    def __init__(self, X, y, X_features=None):
        self.X = [torch.tensor(seq, dtype=torch.long) for seq in X]
        self.y = torch.tensor(y, dtype=torch.long)
        self.X_features = None
        if X_features is not None:
            max_len = max(len(seq) for seq in X_features)
            feature_dim = len(X_features[0][0])
            self.X_features = [torch.tensor(seq + [[0]*feature_dim]*(max_len-len(seq)), dtype=torch.float32) for seq in X_features]
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        if self.X_features is not None:
            return self.X[idx], self.y[idx], self.X_features[idx]
        else:
            return self.X[idx], self.y[idx]

def collate_fn(batch):
    if len(batch[0]) == 3:
        inputs, targets, features = zip(*batch)
        padded_inputs = pad_sequence(inputs, batch_first=True)
        padded_features = pad_sequence(features, batch_first=True)
        return padded_inputs.to(DEVICE), torch.tensor(targets).to(DEVICE), padded_features.to(DEVICE)
    else:
        inputs, targets = zip(*batch)
        padded_inputs = pad_sequence(inputs, batch_first=True)
        return padded_inputs.to(DEVICE), torch.tensor(targets).to(DEVICE)

class CropLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, feature_dim=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.use_features = feature_dim is not None
        if self.use_features:
            self.feature_fc = nn.Linear(feature_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim if not self.use_features else embedding_dim*2, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, vocab_size)
    def forward(self, x, features=None):
        embedded = self.embedding(x)
        if self.use_features and features is not None:
            features_proj = self.feature_fc(features)
            combined = torch.cat([embedded, features_proj], dim=-1)
            out, _ = self.lstm(combined)
        else:
            out, _ = self.lstm(embedded)
        out = out[:, -1, :]
        out = self.dropout(self.relu(self.fc1(out)))
        out = self.dropout(out)
        return self.fc2(out)

if X_features is not None:
    feature_dim = len(X_features[0][0])
    dataset = CropDataset(X, y, X_features)
    model = CropLSTM(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, feature_dim=feature_dim).to(DEVICE)
else:
    dataset = CropDataset(X, y)
    model = CropLSTM(vocab_size, EMBEDDING_DIM, HIDDEN_DIM).to(DEVICE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print(f"🚀 Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

# Training Loop
best_loss = float("inf")
patience = 10
wait = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        optimizer.zero_grad()
        if X_features is not None:
            batch_x, batch_y, batch_features = batch
            output = model(batch_x, batch_features)
        else:
            batch_x, batch_y = batch
            output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"✅ Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    if avg_loss < best_loss:
        best_loss = avg_loss
        wait = 0
        torch.save(model.state_dict(), "crop_lstm_model.pth")
        print(f"💾 New best model saved! Loss: {avg_loss:.4f}")
    else:
        wait += 1
        if wait >= patience:
            print("⏹️ Early stopping triggered.")
            break
print("🎉 Training complete. Best loss =", best_loss)
