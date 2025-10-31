import torchvision.transforms.v2 as v2
from torchvision.transforms import InterpolationMode
import torch

def train_transforms():
    return v2.Compose([v2.RandomHorizontalFlip(p = 0.5), 
                        v2.RandomVerticalFlip(p = 0.5),
                        v2.RandomRotation(degrees=[0,270]),
                        v2.RandomResizedCrop(size=(768, 768), scale=(0.8, 1.0), interpolation = InterpolationMode.NEAREST), #--> This chooses a random portion of the image, from 80-100% and resizes it to 512x512
                        v2.Normalize(mean = [0.43240490555763245, 0.5465161204338074, 0.3243345618247986], std = [0.19969302415847778, 0.2004949152469635, 0.1742226630449295]),
                        v2.ColorJitter(brightness = 0.2, contrast=0.2, saturation=0.3),

]) #These transformations are "chained" meaning that the input of one transformation is the output of the previous. It is clear that we are 
#feeding the model 768x768 images as is evident by the penultimate step of the transforms

def val_transforms():
    return v2.Compose([v2.Resize(size = (768,768), interpolation = InterpolationMode.NEAREST),
                       v2.Normalize(mean = [0.43240490555763245, 0.5465161204338074, 0.3243345618247986], std = [0.19969302415847778, 0.2004949152469635, 0.1742226630449295])
                         ]) 

#The values of the v2.Normalize are found from the function inside the /data/find_mean_std.py