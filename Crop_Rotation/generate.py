import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math

# --- Load vocab ---
with open("vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

crop2idx = vocab["crop2idx"]
idx2crop = {int(k): v for k, v in vocab["idx2crop"].items()}
vocab_size = len(crop2idx)

# --- Load crop features ---
with open("crop_data.json", "r", encoding="utf-8") as f:
    crop_data = json.load(f)

# Feature extraction functions (same as prepare_data.py)
def encode_soil_type(soil_types):
    soil_encoding = {
        "Clayey": [1, 0, 0, 0, 0, 0, 0, 0],
        "Loamy": [0, 1, 0, 0, 0, 0, 0, 0],
        "Alluvial": [0, 0, 1, 0, 0, 0, 0, 0],
        "Red": [0, 0, 0, 1, 0, 0, 0, 0],
        "Black": [0, 0, 0, 0, 1, 0, 0, 0],
        "Sandy loam": [0, 0, 0, 0, 0, 1, 0, 0],
        "Laterite": [0, 0, 0, 0, 0, 0, 1, 0],
        "Light soil": [0, 0, 0, 0, 0, 0, 0, 1]
    }
    encoded = [0] * 8
    for soil_type in soil_types:
        if soil_type in soil_encoding:
            for i, val in enumerate(soil_encoding[soil_type]):
                encoded[i] = max(encoded[i], val)
    return encoded

def get_crop_features(crop_name):
    if crop_name not in crop_data:
        return {
            'soil_encoding': [0] * 8,
            'water_requirement': 1,
            'nutrient_group': 1,
            'family_encoding': [0] * 10,
            'grow_duration': 4
        }
    crop_info = crop_data[crop_name]
    soil_encoding = encode_soil_type(crop_info.get('soil_type', []))
    water_requirement = crop_info.get('water_requirement', 1)
    nutrient_group = crop_info.get('nutrient_group', 1)
    families = ["Poaceae", "Fabaceae", "Brassicaceae", "Solanaceae", "Zingiberaceae", 
                "Malvaceae", "Amaranthaceae", "Apiaceae", "Araceae", "Arecaceae"]
    family_encoding = [1 if crop_info.get('family') == family else 0 for family in families]
    grow_duration = crop_info.get('grow_duration', 4)
    return {
        'soil_encoding': soil_encoding,
        'water_requirement': water_requirement,
        'nutrient_group': nutrient_group,
        'family_encoding': family_encoding,
        'grow_duration': grow_duration
    }

def get_feature_vector(seq):
    feature_vec = []
    for crop in seq:
        crop_features = get_crop_features(crop)
        combined = (
            crop_features['soil_encoding'] +
            [crop_features['water_requirement']] +
            [crop_features['nutrient_group']] +
            crop_features['family_encoding'] +
            [crop_features['grow_duration']]
        )
        feature_vec.append(combined)
    return feature_vec

# --- Model (with features) ---
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

EMBEDDING_DIM = 128
HIDDEN_DIM = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEMPERATURE = 0.7
FEATURE_DIM = 8 + 1 + 1 + 10 + 1  # soil(8), water(1), nutrient(1), family(10), duration(1)

model = CropLSTM(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, feature_dim=FEATURE_DIM).to(DEVICE)
model.load_state_dict(torch.load("crop_lstm_model.pth", map_location=DEVICE))
model.eval()
print("🚀 Model loaded successfully")

# --- Sequence generator ---
def top_k_confident_sequences(start_crop, k=5, max_len=5, num_sequences=5, threshold=0.3):
    if start_crop not in crop2idx:
        return [["Unknown crop."]]
    unique_sequences = set()
    final_sequences = []
    attempts = 0
    max_attempts = num_sequences * 10
    while len(final_sequences) < num_sequences and attempts < max_attempts:
        input_seq = [crop2idx[start_crop]]
        result = [start_crop]
        confidences = []
        for _ in range(max_len):
            x = torch.tensor([input_seq], dtype=torch.long).to(DEVICE)
            features = get_feature_vector([idx2crop[idx] for idx in input_seq])
            features_tensor = torch.tensor([features], dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                output = model(x, features_tensor)
                probs = F.softmax(output / TEMPERATURE, dim=1).squeeze()
            top_k = min(k, probs.shape[0])
            topk_probs, topk_indices = torch.topk(probs, k=top_k)
            sampled_idx = topk_indices[torch.multinomial(topk_probs, 1)].item()
            pred_crop = idx2crop[sampled_idx]
            confidence = probs[sampled_idx].item()
            if confidence < threshold:
                break
            if pred_crop in [r.split(" ")[0] for r in result]:
                break
            result.append(f"{pred_crop} ({confidence:.2f})")
            input_seq.append(sampled_idx)
            confidences.append(confidence)
        if len(confidences) == 0:
            continue
        geo_conf = math.exp(sum(math.log(p) for p in confidences) / len(confidences))
        tuple_seq = tuple(result)
        if tuple_seq not in unique_sequences:
            unique_sequences.add(tuple_seq)
            final_sequences.append((result, geo_conf))
        attempts += 1
    return final_sequences

# --- CLI ---
if __name__ == "__main__":
    print("🌿 Crop Rotation Generator (with features)")
    while True:
        crop = input("Enter starting crop (or 'exit'): ").strip()
        if crop.lower() == "exit":
            break
        sequences = top_k_confident_sequences(crop)
        if not sequences:
            print("⚠️ No sequences generated.")
        else:
            for i, (seq, conf) in enumerate(sequences, 1):
                print(f"{i}. {' → '.join(seq)} | Confidence: {conf:.4f}")
