import os
from collections import Counter
import json
import numpy as np

# Load crop data with new features
with open("crop_data.json", "r", encoding="utf-8") as f:
    crop_data = json.load(f)

# Load sequences
with open("crop_rotation_sequences.txt", "r", encoding="utf-8") as f:
    sequences = [line.strip().split(" → ") for line in f if line.strip()]

# Build vocabulary
all_crops = [crop for seq in sequences for crop in seq]
crop_counter = Counter(all_crops)
crops = sorted(list(crop_counter))
crop2idx = {crop: idx for idx, crop in enumerate(crops)}
idx2crop = {idx: crop for crop, idx in crop2idx.items()}
vocab_size = len(crop2idx)

# Feature extraction functions
def encode_soil_type(soil_types):
    """Encode soil types to numerical features"""
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
    
    # Initialize with zeros
    encoded = [0] * 8
    for soil_type in soil_types:
        if soil_type in soil_encoding:
            for i, val in enumerate(soil_encoding[soil_type]):
                encoded[i] = max(encoded[i], val)
    return encoded

def get_crop_features(crop_name):
    """Extract features for a given crop"""
    if crop_name not in crop_data:
        # Return default features for unknown crops
        return {
            'soil_encoding': [0] * 8,
            'water_requirement': 1,
            'nutrient_group': 1,
            'family_encoding': [0] * 10,
            'grow_duration': 4
        }
    
    crop_info = crop_data[crop_name]
    
    # Soil type encoding
    soil_encoding = encode_soil_type(crop_info.get('soil_type', []))
    
    # Water requirement (0=low, 1=medium, 2=high)
    water_requirement = crop_info.get('water_requirement', 1)
    
    # Nutrient group (0=low, 1=medium, 2=high)
    nutrient_group = crop_info.get('nutrient_group', 1)
    
    # Family encoding (one-hot encoding for top families)
    families = ["Poaceae", "Fabaceae", "Brassicaceae", "Solanaceae", "Zingiberaceae", 
                "Malvaceae", "Amaranthaceae", "Apiaceae", "Araceae", "Arecaceae"]
    family_encoding = [1 if crop_info.get('family') == family else 0 for family in families]
    
    # Growth duration
    grow_duration = crop_info.get('grow_duration', 4)
    
    return {
        'soil_encoding': soil_encoding,
        'water_requirement': water_requirement,
        'nutrient_group': nutrient_group,
        'family_encoding': family_encoding,
        'grow_duration': grow_duration
    }

# Save vocab
with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump({"crop2idx": crop2idx, "idx2crop": idx2crop}, f, indent=2)

# Create input-output pairs with features
X, y = [], []
X_features = []  # Additional features for each sequence

for seq in sequences:
    for i in range(1, len(seq)):
        input_seq = seq[:i]
        target = seq[i]
        
        # Crop indices
        X.append([crop2idx[c] for c in input_seq])
        y.append(crop2idx[target])
        
        # Extract features for the sequence
        seq_features = []
        for crop in input_seq:
            crop_features = get_crop_features(crop)
            # Combine all features into a single vector
            combined_features = (
                crop_features['soil_encoding'] + 
                [crop_features['water_requirement']] + 
                [crop_features['nutrient_group']] + 
                crop_features['family_encoding'] + 
                [crop_features['grow_duration']]
            )
            seq_features.append(combined_features)
        
        X_features.append(seq_features)

# Save dataset with features
import pickle
with open("crop_lstm_dataset.pkl", "wb") as f:
    pickle.dump((X, y, X_features), f)

print(f"✅ Prepared {len(X)} input-output crop pairs with enhanced features.")
print(f"📊 Feature vector size: {len(X_features[0][0]) if X_features else 0} dimensions per crop")
print(f"🌱 Total crops: {vocab_size}")
print(f"🔧 Features include: soil type (8), water requirement (1), nutrient group (1), family (10), growth duration (1)")
