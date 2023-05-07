import os
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from torchvision import transforms
from autoencoder import load_autoencoder
from dataset import LensingDataset
from tqdm import tqdm
import random


def set_all_seeds(num):
    random.seed(num)
    np.random.seed(num)
    torch.manual_seed(num)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(num)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler = None, epochs=100, device='cuda', save_dir='checkpoints', early_stopping_patience=10, report_interval=5):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if device == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model.to(device)
        
    else:
        model.to(device)
    
    print("Model Loaded to Device!")
    best_val_loss = 0
    no_improvement_count = 0
    train_losses, val_losses, steps = [], [], []
    print("Training Started!")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for i, batch in tqdm(enumerate(train_dataloader, 0), unit="batch", total=len(train_dataloader)):
            original_image, lensed_image, label, params = batch
            original_image, lensed_image, params = original_image.to(device), lensed_image.to(device), params.to(device)
            
            optimizer.zero_grad()
            img_output, param_output = model(lensed_image)
            loss = nn.MSELoss()
            img_loss = loss(img_output, lensed_image)
            param_loss = loss(param_output, params)
            sum_loss = img_loss + param_loss
            sum_loss.backward()
            optimizer.step()

            train_loss += sum_loss.item()
            train_losses.append(sum_loss.item())
            steps.append(epoch * len(train_dataloader) + i + 1)

        train_loss /= len(train_dataloader)
        print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.4e}")

        if scheduler is not None:
            scheduler.step()

        if (epoch + 1) % report_interval == 0:
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch in val_dataloader:
                    original_image, lensed_image, label, params = batch
                    original_image, lensed_image, params = original_image.to(device), lensed_image.to(device), params.to(device)
                    img_output, param_output = model(lensed_image)
                    loss = nn.MSELoss()
                    img_loss = loss(img_output, lensed_image)
                    param_loss = loss(param_output, params)
                    loss = img_loss + param_loss
                    sum_loss = img_loss + param_loss
                    val_loss += sum_loss.item()
                    val_losses.append(sum_loss.item())

            val_loss /= len(val_dataloader)
            lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
            print(f"Epoch: {epoch + 1}, Validation Loss: {val_loss:.4f}, Learning rate: {lr}")

            if val_loss > best_val_loss:
                best_val_loss= val_loss
                no_improvement_count = 0
                best_val_epoch = epoch + 1
                torch.save(model.module.state_dict(), os.path.join(save_dir, f"best_model.pt"))
            else:
                no_improvement_count += 1

            if no_improvement_count >= early_stopping_patience:
                print(f"Early stopping after {early_stopping_patience} epochs without improvement.")
                break

    torch.save(model.module.state_dict(), os.path.join(save_dir, "final_model.pt"))
    
    # Plot loss vs. training step graph
    plt.figure(figsize=(10, 5))
    plt.plot(steps, train_losses)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Train Loss vs. Training Steps')
    plt.savefig(os.path.join(save_dir, "train_loss_vs_training_steps.png"), bbox_inches='tight')
    
    plt.figure(figsize=(10, 5))
    plt.plot(steps, val_losses)
    plt.xlabel('Training Steps')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss vs. Training Steps')
    plt.savefig(os.path.join(save_dir, "val_loss_vs_training_steps.png"), bbox_inches='tight')
    
    return best_val_epoch, best_val_loss, train_losses[-1]


def main(config):
    model = load_autoencoder()
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params_to_optimize, lr = config['parameters']['lr'], 
                            weight_decay = config['parameters']['weight_decay'])

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = config['parameters']['milestones'],gamma=config['parameters']['lr_decay'])
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(180),
        transforms.CenterCrop(180),
        transforms.Resize(256, antialias=True),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    print("Loading train dataset!")
    start = time.time()
    train_dataset = LensingDataset(config['dataset'], transform)
    end = time.time()
    print(f"dataset loaded in {end - start:.3f} s")
    
    val_len = int(config['parameters']['val_size'] * len(train_dataset))
    train_len = len(train_dataset) - val_len
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_len, val_len])
    train_dataloader = DataLoader(train_dataset, batch_size=config['parameters']['batch_size'], shuffle=True, pin_memory = (True if torch.cuda.is_available() else False))
    val_dataloader = DataLoader(val_dataset, batch_size=config['parameters']['batch_size'], shuffle=True, pin_memory = (True if torch.cuda.is_available() else False))

    timestr = time.strftime("%Y%m%d-%H%M%S")
    save_dir = config['save_dir'] + config['model'] + '_' + timestr
    best_val_epoch, best_val_acc, final_loss = train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs=config['parameters']['epochs'], device=device, save_dir=save_dir,early_stopping_patience=config['parameters']['early_stopping'], report_interval=config['parameters']['report_interval'])
    print('Training Done')
    
    config['best_val_acc'] = best_val_acc
    config['best_val_epoch'] = best_val_epoch
    config['final_loss'] = final_loss
    # config['feature_fields'] = feature_fields

    file = open(f'{save_dir}/config.yaml',"w")
    yaml.dump(config, file)
    file.close()

    
if __name__ == '__main__':

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
        
    set_all_seeds(42)

    parser = argparse.ArgumentParser(description = 'Train the models')
    parser.add_argument('--config', metavar = 'config', required=True,
                    help='Location of the config file')

    args = parser.parse_args()


    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
   
    main(config)
