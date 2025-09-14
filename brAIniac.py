# brAIniac code base
# for file management and interaction with my OS
import os

# for data analyisis
import pandas as pd

BASE_DIRECTORY = "C:/Users/chidi/Downloads/brisc2025/classification_task"

TRAIN_DIRECTORY = os.path.join(BASE_DIRECTORY, "train")
TEST_DIRECTORY = os.path.join(BASE_DIRECTORY, "test")


IMAGE_SIZE = (224,224)

BATCH_SIZE = 32

# better handling of path files
from pathlib import Path

# CLASS_NAMES = sorted([d.name for d in Path(TRAIN_DIRECTORY).iterdir() if d.is_dir()])
# print (CLASS_NAMES)
# remeber the index shows the class 0, 1, 2, 3
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
