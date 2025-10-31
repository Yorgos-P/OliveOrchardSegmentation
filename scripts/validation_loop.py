import torch.nn as nn
import torch

def validation(model, loader, device):
    model.eval() #Set this to eval mode?
    val_loss = 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            criterion = nn.CrossEntropyLoss(ignore_index = 2)
            loss = criterion(outputs, masks.squeeze(1))
            val_loss += loss
    return val_loss/len(loader)
