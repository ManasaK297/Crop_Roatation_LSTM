# Crop Rotation LSTM

This project predicts crop rotation sequences using an LSTM neural network with crop features.  
It supports training and generation of crop sequences based on soil type, water requirement, nutrient group, family, and grow duration.

## Features
- Uses crop features from `crop_data.json`
- Trains an LSTM model for crop sequence prediction
- Generates crop rotation sequences with confidence scores

## Requirements
- Python 3.8+
- PyTorch
- (Optional) Jupyter Notebook for testing

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/crop-rotation-lstm.git
   cd crop-rotation-lstm
   ```

2. Install dependencies:
   ```
   pip install torch
   ```

3. Prepare your data:
   - Ensure `crop_data.json` and `vocab.json` are present in the project folder.

## Usage

### Train the Model
Run the training script:
```
python train_lstm.py
```

### Generate Crop Rotations
Run the generator:
```
python generate.py
```
Follow the prompts to enter a starting crop and view generated sequences.

## Files

- `train_lstm.py` — Train the LSTM model
- `generate.py` — Generate crop rotation sequences
- `prepare_data.py` — Prepare dataset and features
- `crop_data.json` — Crop features data
- `vocab.json` — Crop vocabulary

