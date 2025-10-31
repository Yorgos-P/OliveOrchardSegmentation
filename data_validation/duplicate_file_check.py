#The purpose is to see whether there are duplicate or near identical files between the train and val sets
import os 
from PIL import Image
import imagehash 
from pathlib import Path
from tqdm import tqdm
import cv2 as cv
import numpy as np

train_dir = Path("training/images")
val_dir = Path("val/images")
Path("comparison").mkdir(parents = True, exist_ok = True)


 
def compute_whash(img_dir, hash_size):
    hash_dic = {}
    for file in img_dir.iterdir():
        with Image.open(file) as img:
            img_hash = imagehash.whash(img, hash_size = hash_size)
        hash_dic[file] = img_hash
    return hash_dic


val_whashes = compute_whash(val_dir, hash_size = 8)
train_whashes = compute_whash(train_dir, hash_size = 8)


def duplicate_finder(train_whashes, val_whashes, tolerance = None):
    pairs = {}
    for train_name, train_hash in tqdm(train_whashes.items(), desc = 'Hash Comparison'):
        for val_name, val_hash in val_whashes.items():
            distance = train_hash - val_hash #Hamming Distance
            pairs[(train_name, val_name)] = distance
    sorted_pairs = dict(sorted(pairs.items(), key=lambda item: item[1]))
    return sorted_pairs

pairs_dict = duplicate_finder(train_whashes, val_whashes)

def side_by_side(dict, top_n_similar):
    counter = 0
    for key, value in dict.items():
        counter += 1
        if counter == top_n_similar:
            break
        else:
            train_img, val_img = cv.imread(key[0]), cv.imread(key[1])
            name_1, name_2 = key[0].name, key[1].name
            together = np.hstack([train_img, val_img])
            cv.imwrite(f"comparison/{name_1} | {name_2}", together)

side_by_side(pairs_dict, 30)







