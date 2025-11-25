import matplotlib.pyplot as plt
from fastai.vision.all import *
from .config import EXPORT_PATH, EPOCHS, LEARNING_RATE

def train_and_save(learn):
    """
    Trains the model and exports the inference file.
    """
    print(f"Starting training for {EPOCHS} epochs...")
    

    learn.fine_tune(EPOCHS, base_lr=LEARNING_RATE)
    
    print("Training complete.")
    
    
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix()
    plt.title("Confusion Matrix")
 
    learn.export(EXPORT_PATH)
    print(f"Model exported to: {EXPORT_PATH}")