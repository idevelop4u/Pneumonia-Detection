from pathlib import Path

# Project Paths

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / 'data' / 'chest_xray'
MODELS_PATH = BASE_DIR / 'models'


MODELS_PATH.mkdir(exist_ok=True)


BATCH_SIZE = 64
IMG_SIZE = 224
LEARNING_RATE = 2e-3
EPOCHS = 4
SEED = 42


MODEL_NAME = 'pneumonia_classifier.pkl'
EXPORT_PATH = MODELS_PATH / MODEL_NAME