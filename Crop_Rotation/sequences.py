import json
import random

# Load structured crop rotation data
with open('crop_data.json', 'r', encoding='utf-8') as f:
    crop_data = json.load(f)

# Parameters
NUM_SEQUENCES_PER_CROP = 5
MAX_ROTATION_LENGTH = 4  # how many crops in each sequence (including the start)

def generate_sequences(data, num_sequences=5, max_length=4):
    sequences = []

    for crop_name in data:
        for _ in range(num_sequences):
            sequence = [crop_name]
            current_crop = crop_name

            for _ in range(max_length - 1):
                successors = data.get(current_crop, {}).get("successor_crops", [])
                if not successors:
                    break

                next_crop = random.choice(successors)
                sequence.append(next_crop)

                # Prevent loops and ensure next exists in dataset
                if next_crop not in data:
                    break
                current_crop = next_crop

            sequences.append(" → ".join(sequence))

    return sequences

# Generate sequences
rotation_sequences = generate_sequences(crop_data, NUM_SEQUENCES_PER_CROP, MAX_ROTATION_LENGTH)

# Save to file (UTF-8 encoding!)
with open('crop_rotation_sequences.txt', 'w', encoding='utf-8') as f:
    for seq in rotation_sequences:
        f.write(seq + '\n')

print(f"✅ Generated {len(rotation_sequences)} crop rotation sequences.")
