import os
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from tqdm import tqdm

def train(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0

    model.train()
    #Progress bar
    pbar = tqdm(enumerate(loader), 'Training', total=len(loader),leave=False)
    
    for x, y in loader:
        #x = x.to(device, dtype=torch.float32)
        x = x.to(device)

        #y = y.to(device, dtype=torch.float32)

        #y = y.to(device, dtype=torch.int64)
        y = y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        #print(y)
        loss = loss_fn(y_pred, y)
        loss.backward()

        optimizer.step()
        epoch_loss += loss.item()
        
        # update progressbar
        pbar.set_description(f'Training: (loss {loss.item():.4f})')
        pbar.update()

    epoch_loss = epoch_loss/len(loader)
    
    pbar.close()
    return epoch_loss

def evaluate(model, loader, loss_fn, device):
    epoch_loss = 0.0

    model.eval()
    #Progress bar
    pbar= tqdm(enumerate(loader), 'Validation', total=len(loader),leave=False)
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()
            
            # update progressbar
            pbar.set_description(f'Validation: (loss {loss.item():.4f})')
            pbar.update()

        epoch_loss = epoch_loss/len(loader)
        
        pbar.close()
    return epoch_loss