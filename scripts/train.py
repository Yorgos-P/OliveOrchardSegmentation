from data.dataset_loader import OliveOrchard
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from scripts.training_loop import train_one_epoch
from scripts.validation_loop import validation
import torch
from data.augmentations import train_transforms
from data.augmentations import val_transforms
import torch.optim as optim






def train(train_ds = OliveOrchard('training', transform = train_transforms()), val_ds = OliveOrchard('val', transform = val_transforms()), batch_size = 8, lr = 1e-4, num_epochs = 40, device="cuda",
    checkpoint_path="checkpoints/unet_best.pth"):

    training_set = DataLoader(train_ds, batch_size = batch_size, shuffle = True)
    validation_set = DataLoader(val_ds, batch_size = batch_size, shuffle = False)
    model = smp.Unet(encoder_name= "resnet34", encoder_weights = "imagenet", in_channels = 3, classes = 3)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf') #what is this?
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, training_set, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs} -- Train Loss: {train_loss: .4f}")
        val_loss = validation(model, validation_set, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved new best model with val_loss = {val_loss:.4f}") #.4f means show the val_loss to the 4th decimal point, also format at it as a float
    print("Training Completed")
    return model 

train(train_ds = OliveOrchard('training', transform = train_transforms()), val_ds = OliveOrchard('val', transform = val_transforms()), batch_size = 8, lr = 1e-4, num_epochs = 40, device="cuda",
    checkpoint_path="checkpoints/unet_best.pth")



