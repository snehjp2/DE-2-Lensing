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
from socket import gethostname
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def set_all_seeds(num):
    random.seed(num)
    np.random.seed(num)
    torch.manual_seed(num)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(num)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train(model, train_dataloader, optimizer, epoch, device='cuda', save_dir='checkpoints'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_losses, steps = [], [], []
    print("Training Started!")

    train_loss = 0.0
    model.train()
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

    train_loss /= len(train_dataloader)
    print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.4e}")
    return train_loss


def validate(model, val_dataloader, device='cuda'):
    model.eval()
    val_loss = 0.0
    val_losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            original_image, lensed_image, label, params = batch
            original_image, lensed_image, params = original_image.to(device), lensed_image.to(device), params.to(device)
            img_output, param_output = model(lensed_image)
            loss = nn.MSELoss()
            img_loss = loss(img_output, lensed_image)
            param_loss = loss(param_output, params)
            sum_loss = img_loss + param_loss
            val_loss += sum_loss.item()
            val_losses.append(sum_loss.item())

    val_loss /= len(val_dataloader)
    return val_loss


def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
def main(config):
    
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
    timestr = time.strftime("%Y%m%d-%H%M%S")
    save_dir = config['save_dir'] + 'autoencoder_' + timestr
    
    print("Loading train dataset!")
    start = time.time()
    train_dataset = LensingDataset(config['dataset'], transform)
    end = time.time()
    print(f"dataset loaded in {end - start:.3f} s")
    
    val_len = int(config['parameters']['val_size'] * len(train_dataset))
    train_len = len(train_dataset) - val_len
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_len, val_len])


    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["SLURM_PROCID"])
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    assert gpus_per_node == torch.cuda.device_count()
    print(f"Hello from rank {rank} of {world_size} on {gethostname()} where there are" \
          f" {gpus_per_node} allocated GPUs per node.", flush=True)
    setup(rank, world_size)
    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)
    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)
    print(f"host: {gethostname()}, rank: {rank}, local_rank: {local_rank}")
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(train_dataset, batch_size=config['parameters']['batch_size'], sampler = train_sampler,
                                  num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]), pin_memory=True)                
    val_dataloader = DataLoader(val_dataset, batch_size=config['parameters']['batch_size'], 
                                num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]), pin_memory=True)
    
    model = load_autoencoder().to(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank])
    optimizer = optim.AdamW(ddp_model.parameters(), lr = config['parameters']['lr'], 
                            weight_decay = config['parameters']['weight_decay'])

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = config['parameters']['milestones'], gamma=config['parameters']['lr_decay'])
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_val_epoch = 0
    final_train_loss = None
    early_stopping_patience = config['parameters']['early_stopping']
    early_stopping_counter = 0
    
    for epoch in range(1, config['parameters']['epochs'] + 1):
        train_loss = train(model = ddp_model, device=local_rank, train_loader=train_dataloader, optimizer=optimizer, epoch=epoch)
        
        if epoch % config['parameters']['report_interval'] == 0:
            if rank == 0:
                val_loss = validate(model=ddp_model, val_dataloader=val_dataloader, device=local_rank)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_epoch = epoch
                    if rank == 0:
                        torch.save(model.state_dict(), os.path.join(save_dir, f"best_model.pt"))
                        
                    early_stopping_counter = 0
                    
                else:
                    early_stopping_counter += 1
                    
                    if early_stopping_counter >= early_stopping_patience:
                        print("Early stopping triggered. No improvement in validation loss.")
                        break
                    
        scheduler.step()
        
        if epoch == config['parameters']['epochs']:
            final_train_loss = train_loss  # Capture the final training loss at the last epoch
            if rank == 0:
                torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pt"))

    if rank == 0:
        
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, config['parameters']['epochs'] + 1), train_losses, label='Train Loss')
        plt.plot(range(config['parameters']['report_interval'], config['parameters']['epochs'] + 1, config['parameters']['report_interval']),
                 val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'loss_curve.png'), bbox_inches='tight')
        plt.close()

    dist.destroy_process_group()
    
    config['best_val_loss'] = best_val_loss
    config['best_val_epoch'] = best_val_epoch
    config['final_train_loss'] = final_train_loss

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
