import torch
from tqdm import tqdm 
import torch.nn as nn

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0

    for images, masks in tqdm(loader):
        images, masks = images.to(device), masks.to(device) #what does this do?
        optimizer.zero_grad() #We need to reset the gradient because we have stored the previous gradient (for the previous iteration)
        outputs = model(images) #I assume this is where we get the outputs, in this case segmentation mask of our model
        criterion = nn.CrossEntropyLoss(ignore_index = 2) #Set the loss
        loss = criterion(outputs, masks.squeeze(1)) #masks.squeeze transforms [B,1,H,W] to [B,H,W] which is the required format for that loss function
        loss.backward() #Compute the gradient for every parameter (sometimes not all parameters, I think it also depends on things like Dropout)
        optimizer.step() #Updates the parameters
        running_loss += loss.item() 
    return running_loss/len(loader) #I think this calculates the average epoch loss


