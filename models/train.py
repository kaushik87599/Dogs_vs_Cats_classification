# Load dataset, choose model (base/custom)

# Loss: CrossEntropyLoss

# Optimizer: Adam

# Scheduler (optional): StepLR or ReduceLROnPlateau

# Track training/validation loss, accuracy

# Save best model

import torch
from data.dataset import get_dataloaders
from models.basic_cnn import basic_cnn
from models.attention_block import AttentionBlock
from custom_cnn import custom_attention_cnn

train_loader, val_loader = get_dataloaders(train_dir='data/processed/train', val_dir='data/processed/val')

base_model = basic_cnn()
custom_model = custom_attention_cnn(base_model, AttentionBlock(256, 256, 4))
