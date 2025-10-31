from torch.utils.data import Dataset
from pathlib import Path
from torchvision.io import decode_image
import os
import torch 
from torchvision.tv_tensors import Image as TVImage, Mask as TVMask
#from augmentations import train_transforms
from torchvision.transforms.functional import to_pil_image
import cv2


#The class is a type of Dataset class imported from torch.utils.data, hence why it is used in the argument
#We are overriding functions like __len__ and so on
#The reason why we are constructing a subclass of a Pytorch class will be clear in the future


class OliveOrchard(Dataset):
    def __init__(self, root_path, transform):
        self.imgs = list(sorted(os.listdir(os.path.join(root_path, "images")))) #root_path either defines training or val folder, then this makes an array of all image names (strings)
        self.masks = list(sorted(os.listdir(os.path.join(root_path, "annotations"))))
        self.root_path = root_path
        self.transform = transform
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        #Here we need to return numpy arrays for the images and masks
        img_path = os.path.join(self.root_path, "images", self.imgs[idx]) #This creates a string for the full path of an image (the idx image). We need the full path so we can pass it to decode_image
        mask_path = os.path.join(self.root_path, "annotations", self.masks[idx])
        image = TVImage(decode_image(img_path)) #why use tv_tensors --> it's more compatible when you are using torchvision augmentation techniques
        mask = TVMask(decode_image(mask_path))
        image = image.to(torch.float32)/255 #We divide by 255 to obtain pixel ranges in [0,1] because in augmentations.py, we apply brightness/saturation/contrast transformations which require the pixels to be in this form
        mask = mask.to(torch.long)
        mask_clone = mask.clone()
        mask_clone[mask_clone == 128] = 1
        mask_clone[mask_clone == 255] = 2
        if self.transform is not None:
            image, mask_clone = self.transform(image, mask_clone)
        return image, mask_clone #returns pytorch tensors of a single mask and single image
    
""" 
#Sanity check to see how the image looks like after the augmentations
train = OliveOrchard('training', transform = train_transforms())
img, mask = train[4]
img = (img.clamp(0, 1) * 255).to(torch.uint8)
img = img.byte()  # Convert to uint8
img = img.permute(1, 2, 0).numpy()  # C x H x W → H x W x C
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

cv2.imshow('window_name', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""



        

