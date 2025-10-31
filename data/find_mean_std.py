from pathlib import Path
import torch
import torchvision.transforms.v2 as v2
from torchvision.io import decode_image

def find_mean_std(training_path):
    training_dir = Path(training_path)
    counter = 0
    x_mean = torch.zeros(3, dtype = torch.float32)
    x_std = torch.zeros(3, dtype = torch.float32)
    for img in training_dir.iterdir():
        counter += 1
        img = decode_image(img)
        img = img.to(torch.float32)
        x_mean += img.mean(dim = (1,2))
        x_std += img.std(dim = (1,2), unbiased = False)
    x_mean = (x_mean/counter).tolist()
    x_std = (x_std/counter).tolist()
    return (x_mean, x_std)

#print(find_mean_std('training/images')) = ([0.43240490555763245, 0.5465161204338074, 0.3243345618247986], [0.19969302415847778, 0.2004949152469635, 0.1742226630449295])