from pathlib import Path
import cv2 as cv 
from PIL import Image
import numpy as np
import argparse 
import shutil
import csv
import random
import glob
import os
from sklearn.model_selection import train_test_split

val = Path("val")
training = Path("training")
val_images = Path("val/images")
val_annotations = Path("val/annotations")
training_images = Path("training/images")
training_annotations = Path("training/annotations")
val.mkdir(exist_ok=True)
training.mkdir(exist_ok=True)
val_images.mkdir(exist_ok=True)
val_annotations.mkdir(exist_ok=True)
training_annotations.mkdir(exist_ok=True)
training_images.mkdir(exist_ok=True)

imgs = sorted(glob.glob("Images/*.jpg")) #Returns array of strings where each element corresponds to the name of an image
masks = sorted(glob.glob("Masks/*.png"))

#Sanity Check that images correspond to masks
if (len(imgs) == len(masks)):
    for _ in range(len(imgs)):
        assert os.path.basename(imgs[_][:-4]) == os.path.basename(masks[_][:-4]) #This prints out the file name with no extension and without the parent folder (as that is also included in teh string)

img_train, img_val, mask_train, mask_val = train_test_split(imgs, masks, train_size = 0.8, random_state = 48, shuffle =True)

for imgs, masks in zip(img_train, mask_train):
    shutil.copy(imgs, "training/images")
    shutil.copy(masks, "training/annotations")

for imgs, masks in zip(img_val, mask_val):
    shutil.copy(imgs, "val/images")
    shutil.copy(masks, "val/annotations")


