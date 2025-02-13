import torch
from torch.utils.data import Dataset

class Combined4ChannelsDataset(Dataset):
    def __init__(self, base_dataset, lbp_dataset):
        self.base_dataset = base_dataset
        self.lbp_dataset = lbp_dataset
        self.classes = base_dataset.classes

    def __len__(self):
        return min(len(self.base_dataset), len(self.lbp_dataset))

    def __getitem__(self, idx):
        base_data, base_target = self.base_dataset[idx]
        lbp_data, lbp_target = self.lbp_dataset[idx]

        # Extract the first channel from lbp_data
        lbp_first_channel = lbp_data[0:1, :, :]  # Shape: (1, height, width)

        # Stack the lbp_first_channel with base_data
        combined_data = torch.cat((base_data, lbp_first_channel), dim=0)  # Shape: (4, height, width)

        # Assuming the targets are the same, use either base_target or lbp_target
        return combined_data, base_target
    
class Combined5ChannelsDataset(Dataset):
    def __init__(self, base_dataset, lbp_dataset, sobel_dataset):
        self.base_dataset = base_dataset
        self.lbp_dataset = lbp_dataset
        self.sobel_dataset = sobel_dataset
        self.classes = base_dataset.classes

    def __len__(self):
        return min(len(self.base_dataset), len(self.lbp_dataset), len(self.sobel_dataset))

    def __getitem__(self, idx):
        base_data, base_target = self.base_dataset[idx]
        lbp_data, lbp_target = self.lbp_dataset[idx]
        sobel_data, sobel_target = self.sobel_dataset[idx]

        # Extract the first channel from lbp_data
        lbp_first_channel = lbp_data[0:1, :, :]  # Shape: (1, height, width)

        # Extract the first channel from lbp_data
        sobel_first_channel = sobel_data[0:1, :, :]  # Shape: (1, height, width)

        # Stack lbp_first_channel and sobel_first_channel with base_data
        combined_data = torch.cat((base_data, lbp_first_channel, sobel_first_channel), dim=0)  # Shape: (4, height, width)

        # Assuming the targets are the same, use either base_target, lbp_target or sobel_target 
        return combined_data, base_target