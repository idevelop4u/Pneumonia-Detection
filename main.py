from src.data_loader import get_data_loaders
from src.model import build_learner
from src.train_utils import train_and_save

def main():
    print("========================================")
    print("   PNEUMONIA DETECTION - TRAINING CLI   ")
    print("========================================")
    
    try:
        # 1. Load Data
        dls = get_data_loaders()
        
        # 2. Build Model
        learn = build_learner(dls)
        
        # 3. Train & Save
        train_and_save(learn)
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Check if your 'data/chest_xray' folder structure is correct.")

if __name__ == "__main__":
    main()