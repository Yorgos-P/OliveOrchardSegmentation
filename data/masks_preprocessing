from PIL import Image
import numpy as np
from pathlib import Path

training_masks = Path('training/annotations')
val_masks = Path('val/annotations')


color_to_id = {(0,0,0):0, (66,193,150):255, (217,3,216): 128}
def rgb_2_gray(path):
    mask_rgb = np.array(Image.open(path).convert('RGB'))
    background = np.zeros(mask_rgb.shape[:2], dtype = np.uint8)
    counter = 0
    for color, gray_val in color_to_id.items():
        matches = np.all(color == mask_rgb, axis = -1) #Array of True False entries, True are entries where a pixel corresponds to color
        background[matches] = gray_val #We change the value of those pixels to the grayscale values that we picked in the above dictionary
        counter += 1
    gray_img = Image.fromarray(background, mode = 'L') #Create image from the array
    gray_img.save(path)
    return counter

for masks in training_masks.iterdir():
    if masks.suffix.lower() == ".png":
        rgb_2_gray(masks)

for masks in val_masks.iterdir():
    if masks.suffix.lower() == ".png":
        rgb_2_gray(masks)

    