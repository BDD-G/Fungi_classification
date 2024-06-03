import torch
from torch.utils.data import Dataset

class CombinedChannelDataset(Dataset):
    def __init__(self, base_dataset, lbp_dataset):
        self.base_dataset = base_dataset
        self.lbp_dataset = lbp_dataset

    def __len__(self):
        return min(len(self.base_dataset), len(self.lbp_dataset))

    def __getitem__(self, idx):
        base_data, base_target = self.base_dataset[idx]
        lbp_data, lbp_target = self.lbp_dataset[idx]

        # Extract the first channel from lbp_data
        lbp_first_channel = lbp_data[0:1, :, :]  # Shape: (1, height, width)

        # Stack the lbp_first_channel with base_data
        combined_data = torch.cat((base_data, lbp_first_channel), dim=0)  # Shape: (4, height, width)

        # Assuming the targets are the same, you can use either base_target or lbp_target
        return combined_data, base_target