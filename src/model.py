from fastai.vision.all import *

def build_learner(dls):
    """
    Constructs the CNN learner using ResNet34 architecture.
    """
    # We use transfer learning with a pre-trained ResNet34
    learn = vision_learner(
        dls, 
        resnet34, 
        metrics=[accuracy, error_rate, Precision(), Recall()]
    )
    return learn