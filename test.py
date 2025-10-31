from pathlib import Path
import torch
import torchvision.transforms.v2 as v2
from torchvision.io import decode_image

img = decode_image("/Users/yorgos/Desktop/OliveCrownCanopies/training/images/DJI_0408_3x3_2_2.jpg").to(torch.float32)
img1 = v2.Normalize(mean = [110.26325988769531, 139.36158752441406, 82.705322265625], std = [50.92170333862305, 51.12620162963867, 44.42678451538086])(img)




def find_mean_std(training_path):
    training_dir = Path(training_path)
    counter = 0
    x_mean = torch.zeros(3, dtype = torch.float32)
    x_std = torch.zeros(3, dtype = torch.float32)
    for img in training_dir.iterdir():
        counter += 1
        img = decode_image(img)
        img = img.to(torch.float32)/255
        x_mean += img.mean(dim = (1,2))
        x_std += img.std(dim = (1,2), unbiased = False)
    x_mean = (x_mean/counter).tolist()
    x_std = (x_std/counter).tolist()
    return (x_mean, x_std)

print(find_mean_std('training/images'))
    











