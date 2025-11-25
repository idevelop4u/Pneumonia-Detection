import os
from fastai.vision.all import *
from .config import DATA_PATH, BATCH_SIZE, IMG_SIZE, SEED

def get_data_loaders():
    """
    Creates and returns the FastAI DataLoaders object.
    
    Expected Data Structure:
    data/
      chest_xray/
        train/
          NORMAL/
          PNEUMONIA/
        test/
          NORMAL/
          PNEUMONIA/
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please download the chest_xray dataset from Kaggle.")


    pneumonia_block = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=GrandparentSplitter(train_name='train', valid_name='test'),
        get_y=parent_label,
        item_tfms=Resize(IMG_SIZE),
        batch_tfms=aug_transforms(size=IMG_SIZE, min_scale=0.75) 
    )


    dls = pneumonia_block.dataloaders(DATA_PATH, bs=BATCH_SIZE, seed=SEED)
    
    print(f"Data Loaded Successfully.")
    print(f"   - Classes Detected: {dls.vocab}")
    print(f"   - Training Set: {len(dls.train_ds)} images")
    print(f"   - Validation Set: {len(dls.valid_ds)} images")
    
    return dls