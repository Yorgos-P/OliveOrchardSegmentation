from PIL import Image
from pathlib import Path
import torchvision.transforms.v2 as v2
from torchvision.transforms import InterpolationMode
from torchvision.io import decode_image
import torch
from torchvision.tv_tensors import Image as TVImage, Mask as TVMask
import segmentation_models_pytorch as smp
from torchvision.io import read_image
import numpy as np

model = smp.Unet(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=3)
state_dict = torch.load("checkpoints/unet_best.pth", map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.to('cpu').eval

def forward_pass(img, overlay = False, eval_mode = False):
    if eval_mode == False:
        raw = Image.open(img)
        raw_for_overlay = np.array(raw.resize((768, 768), Image.NEAREST)) #-->Is it resizing in the same way as pytorch??

        img = TVImage(decode_image(img))
        img = img.to(torch.float32)/255
    else:
        img = v2.Compose([v2.Resize(size = (768, 768), interpolation = InterpolationMode.NEAREST),
                      v2.Normalize(mean = [0.43240490555763245, 0.5465161204338074, 0.3243345618247986], std = [0.19969302415847778, 0.2004949152469635, 0.1742226630449295])])(img)
        
    logits = model(img.unsqueeze(0).to('cpu')) #.unsqueeze 0 does the following: [C,H,W] --> [B,C,H,W] (because pytorch models usually expect batches)
    logits[:,2,:,:] = float('-inf') #at inference, we don't want the model to predict ignore_region so we set all the values of that class to -inf before passing to argmax
    mask = logits.argmax(dim=1).squeeze(0).to(torch.long) #.argmax(dim=1) return the index of the maximum value of an entry compared channel-wise 
    if overlay == False:
        return mask
    else:
         with torch.no_grad():
             colour_mask = torch.tensor([[0,0,0], [200,0,0], [0,200,0]], dtype = torch.uint8) 
             colour_mask = colour_mask[mask] #attention, this is in [H,W,C] format! This line does the following mapping 0 --> [0,0,0], 1 --> [200,0,0], 2 --> [0,200,0]
             overlay = (0.6 * raw_for_overlay + 0.4*colour_mask.numpy()).astype(np.uint8)
             Image.fromarray(overlay).save("prediction_overlay.png")
             print("Saved prediction_overlay.png")

        


#print(forward_pass('val/images/DJI_0407_3x3_0_2.jpg', True))
