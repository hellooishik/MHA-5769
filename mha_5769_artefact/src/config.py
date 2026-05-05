import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Training Hyperparameters
BATCH_SIZE = 64
EPOCHS = 2
LEARNING_RATE = 0.001
RANDOM_SEED = 42

# Model paths
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'best_model.pth')
