import os
import pandas as pd
import numpy as np
from datetime import datetime
import time
import json
from IPython.display import clear_output
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.models import resnet50, ResNet50_Weights

from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.ndimage import gaussian_filter
from PIL import Image


import warnings
warnings.filterwarnings('ignore')

# Import helper scripts
import sys
sys.path.append('/zhome/ac/d/174101/thesis')
from utils.ResNet import ResNet
from utils.Hyperparams import Hyperparams
from utils.train import  train, evaluate
from utils.DenseNet import DenseNet
from utils.CombineChannels import CombinedChannelDataset

# Paths and classes
img_path = '/work3/s220243/Thesis'
base_path = '/zhome/ac/d/174101/thesis'

# resnetX = (Num of channels, repetition, Bottleneck_expansion , Bottleneck_layer)
model_parameters={}
model_parameters['resnet18'] = ([64,128,256,512],[2,2,2,2],1,False)
model_parameters['resnet34'] = ([64,128,256,512],[3,4,6,3],1,False)
model_parameters['resnet50'] = ([64,128,256,512],[3,4,6,3],4,True)
model_parameters['resnet101'] = ([64,128,256,512],[3,4,23,3],4,True)
model_parameters['resnet152'] = ([64,128,256,512],[3,8,36,3],4,True)
# DensNetX
model_parameters['densenet121'] = [6,12,24,16]
model_parameters['densenet169'] = [6,12,32,32]
model_parameters['densenet201'] = [6,12,48,32]
model_parameters['densenet264'] = [6,12,64,48]

# Helper functions
class LBP:
    def __init__(self, radius=1, n_points=8):
        self.radius = radius
        self.n_points = n_points

    def __call__(self, image):
        image_gray = rgb2gray(image)
        image_gaussian = gaussian_filter(image_gray, sigma=0.5)
        lbp_image = local_binary_pattern(image_gaussian, self.n_points, self.radius, method='uniform')

        x = np.stack([lbp_image] * 3, axis=-1)
        pseudo_rgb_image = (x-np.min(x))/(np.max(x)-np.min(x))
        pseudo_rgb_image = (pseudo_rgb_image * 255).astype(np.uint8)  # Convert to uint8 for PIL compatibility
        pseudo_rgb_image = Image.fromarray(pseudo_rgb_image)

        return pseudo_rgb_image

def stack_channels(dataloader1, dataloader2):
    stacked_data = []
    for (data1, target1), (data2, target2) in zip(dataloader1, dataloader2):
        # Extract the first channel from data2
        first_channel = data2[:, 0:1, :, :]  # Shape: (batch_size, 1, height, width)
        
        # Stack the first channel with data2
        stacked_data_batch = torch.cat((data2, first_channel), dim=1)  # Shape: (batch_size, 4, height, width)
        
        # Append to list
        stacked_data.append((stacked_data_batch, target2))
    
    return stacked_data

def load_model(architecture, model):
  if architecture == 'pretrained':
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(1000, num_classes)
    #Freeze all but final layer
    for name, param in model.named_parameters():
      if "fc" in name:
        param.requires_grad = True
      else:
        param.requires_grad = False
  if model == "ResNet":
    model = ResNet(model_parameters[architecture] , in_channels=4, num_classes=num_classes)
  if model == "DenseNet":
     model = DenseNet(model_parameters[architecture] , in_channels=4, num_classes=num_classes)
  return model    

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

data_transforms = {
    'train': v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.7306, 0.6204, 0.5511], [0.1087, 0.1948, 0.1759])
    ]),
    'test': v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.7288, 0.6202, 0.5511], [0.1076, 0.1934, 0.1758])
    ]),
    'validation': v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_transforms_lbp = {
    'train': v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.5791, 0.5791, 0.5791], [0.2315, 0.2315, 0.2315])
    ]),
    'test': v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.5799, 0.5799, 0.5799], [0.2307, 0.2307, 0.2307])
    ]),
    'validation': v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Checkpoint paths
base = Path(f"{img_path}")
checkpoint_dst = base / "checkpoints"
checkpoint_dst.mkdir(exist_ok=True)
checkpoint_path = checkpoint_dst / "checkpoint.pth"

# Create DataFrame
trained_models_df = pd.DataFrame(columns=[
    'filename',
    'train_start_datetime',
    'batch_size',
    'lr',
    'layers',
    'optimizer',
    'loss',
    'trained_epochs',
    'last_saved_epoch',
    'best_val_loss',
    'early_stopping',
    'epoch_train_losses',
    'epoch_val_losses',
    'epoch_train_times',
    'total_train_time',
])

model_dst = base / "models"
model_dst.mkdir(exist_ok=True)
TRAINED_MODELS_CSV = base / "models/trained_models.csv"

isExist = os.path.exists(TRAINED_MODELS_CSV)
if not isExist:
    trained_models_df.to_csv(TRAINED_MODELS_CSV)
    print("Created trained_models.csv")

# Hyperparameters
model_type = "DenseNet"
architecture = 'densenet264'
hyperparams = Hyperparams(Path(base_path) / "data/train_conf.toml", str("4th_channel"), str(architecture))

train_start_datetime = datetime.now()

# Metric scores
metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
jaccard_epoch = []
f1_epoch = []
recall_epoch = []
precision_epoch = []
acc_epoch = []

# Define the base data directory
data_dir = Path(img_path) / 'data_split_resized'

# Define the lbp data directory
data_dir_lbp = Path(img_path) / 'data_split_lbp'

## Stack the 4th channel

# Create the original datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}
image_datasets_lbp = {x: datasets.ImageFolder(os.path.join(data_dir_lbp, x), data_transforms_lbp[x]) for x in ['train', 'test']}

# Create combined datasets
combined_datasets = {x: CombinedChannelDataset(image_datasets[x], image_datasets_lbp[x]) for x in ['train', 'test']}

# Create data loaders
batch_size = hyperparams.batch_size
dataloaders = {x: torch.utils.data.DataLoader(combined_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True) for x in ['train', 'test']}

# Get dataset sizes
dataset_sizes = {x: len(combined_datasets[x]) for x in ['train', 'test']}
print(dataset_sizes)

# Class names
class_names = image_datasets['train'].classes


num_classes = len(class_names)
model = load_model(architecture, model_type) #use architecture for training a full model

# Hyperparameters
num_epochs = hyperparams.epochs
lr = hyperparams.lr
early_stopping_patience = 10
optimizer = hyperparams.optimizer_class(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
loss_fn = hyperparams.loss_fn

# Move the model to the GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training the model
train_start_datetime = datetime.now()

train_losses = []
test_losses = []
lr_epochs = []

epoch_times = []

trained_epochs = 0
last_saved_epoch = 0
early_stopping = False

best_test_loss = float("inf")

train_start = time.time()
for epoch in range(num_epochs):
    start_time = time.time()

    train_loss = train(model, dataloaders['train'], optimizer, loss_fn, device)
    test_loss = evaluate(model, dataloaders['test'], loss_fn, device)

    """ Saving the model """
    if test_loss < best_test_loss:
        torch.save(model.state_dict(), checkpoint_path)

    #if epoch % 5 == 4:
        #clear_output(wait=True)

    if test_loss < best_test_loss:
        data_str = f"Valid loss improved from {best_test_loss:2.4f} to {test_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
        print(data_str)

        best_test_loss = test_loss
        last_saved_epoch = epoch+1

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    trained_epochs = epoch+1
    data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
    data_str += f'\tTrain Loss: {train_loss:.3f}\n'
    data_str += f'\t Test. Loss: {test_loss:.3f}\n'
    print(data_str)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    epoch_times.append(end_time - start_time)
    lr_epochs.append(lr)


    if trained_epochs - last_saved_epoch >= early_stopping_patience:
        early_stopping = True
        break

train_end = time.time()
train_mins, train_secs = epoch_time(train_start, train_end)
data_str = f'Training completed:\n'
data_str += f'\tEpochs: \t{trained_epochs}\n'
data_str += f'\tTraining Time: \t{train_mins}m {train_secs}s\n'
# data_str += f'\tTrain Loss: \t{train_loss:.3f}\n'
# data_str += f'\t Test. Loss: \t{test_loss:.3f}\n'
print(data_str)

# Save results
dict_to_append = {
    'filename': [hyperparams.model_name()],
    'train_start_datetime': [str(train_start_datetime)],
    'batch_size': [hyperparams.batch_size],
    'lr': [hyperparams.lr],
    'optimizer': [hyperparams.optimizer],
    'loss': [hyperparams.loss],
    'trained_epochs': [trained_epochs],
    'last_saved_epoch': [last_saved_epoch],
    'best_val_loss': [best_test_loss],
    'early_stopping': [early_stopping],
    'epoch_train_losses': [json.dumps(train_losses)],
    'epoch_val_losses': [json.dumps(test_losses)],
    'epoch_train_times': [json.dumps(epoch_times)],
    'total_train_time': [train_end - train_start],
}

new_row = pd.DataFrame.from_dict(dict_to_append)
# trained_models_df = pd.concat([trained_models_df, new_row], ignore_index=True)
new_row.to_csv(TRAINED_MODELS_CSV, mode='a', header=False)

# Save model 
model_save_path = base / "models" / hyperparams.model_name()

isExist = os.path.exists(model_save_path)
if not isExist:
    torch.save(model.state_dict(), model_save_path)
    print(f"Saved model to {model_save_path}")
else:
    print(f"WARNING: filename already exists, couldn't save {model_save_path}")

# Traning curve
def plot_training(training_losses,
                  validation_losses,
                  gaussian=True,
                  sigma=2,
                  figsize=(8, 6)
                  ):
    """
    Returns a loss plot with training loss, validation loss
    """

    list_len = len(training_losses)
    x_range = list(range(1, list_len + 1))  # number of x values

    fig = plt.figure(figsize=figsize)
    grid = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)

    subfig1 = fig.add_subplot(grid[0])

    subfigures = fig.get_axes()

    for i, subfig in enumerate(subfigures, start=1):
        subfig.spines['top'].set_visible(False)
        subfig.spines['right'].set_visible(False)

    if gaussian:
        training_losses_gauss = gaussian_filter(training_losses, sigma=sigma)
        validation_losses_gauss = gaussian_filter(validation_losses, sigma=sigma)

        linestyle_original = '.'
        color_original_train = 'lightcoral'
        color_original_valid = 'lightgreen'
        color_smooth_train = 'red'
        color_smooth_valid = 'green'
        alpha = 0.25
    else:
        linestyle_original = '-'
        color_original_train = 'red'
        color_original_valid = 'green'
        alpha = 1.0

    # Subfig 1
    subfig1.plot(x_range, training_losses, linestyle_original, color=color_original_train, label='Training',
                 alpha=alpha)
    subfig1.plot(x_range, validation_losses, linestyle_original, color=color_original_valid, label='Validation',
                 alpha=alpha)
    if gaussian:
        subfig1.plot(x_range, training_losses_gauss, '-', color=color_smooth_train, label='Training', alpha=0.75)
        subfig1.plot(x_range, validation_losses_gauss, '-', color=color_smooth_valid, label='Validation', alpha=0.75)
    subfig1.title.set_text('Training & validation loss')
    subfig1.set_xlabel('Epoch')
    subfig1.set_ylabel('Loss')

    subfig1.legend(loc='upper right')


    return fig


fig = plot_training(train_losses, test_losses, gaussian=True, sigma=1, figsize=(4,4))
plt.savefig('/zhome/ac/d/174101/thesis/plots/'+str(hyperparams.model_name())+'.jpg')