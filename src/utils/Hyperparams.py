import toml
from pathlib import Path
import sys
sys.path.append('/zhome/ac/d/174101/thesis')
from utils.loss import DiceBCELoss, DiceLoss, IoULoss, BCElossFuntion, CrossEntropy, NLL, KLDiv
from torch.nn import BCELoss, CrossEntropyLoss, NLLLoss
import torch.nn as nn
from torch.optim import Adam, RMSprop, SGD, NAdam

class Hyperparams:
    def __init__(self, path: Path, preprocessing: str):
        toml_dict = toml.load(path)

        # To tune
        self.batch_size = toml_dict['batch_size']
        self.epochs = toml_dict['epochs']
        self.lr = toml_dict['lr']
        self.optimizer = toml_dict['optimizer']
        self.loss = toml_dict['loss']
        self.preprocessing = preprocessing

    def model_name(self):
        formatted_lr = "{:.3e}".format(self.lr)
        return f"fungi_model_{self.preprocessing}_B{self.batch_size}_E{self.epochs}_lr{formatted_lr}_{self.optimizer}_{self.loss}.pth"

    # using property decorator
    # a loss getter function
    @property
    def loss_fn(self):
        if self.loss == "CrossEntropy":
            return CrossEntropyLoss()
        if self.loss == "NLL":
            return NLL()
        if self.loss == "KLDIV":
            return KLDiv()
        if self.loss == "BCE":
            return nn.BCEWithLogitsLoss()

    # # using property decorator
    # # a optimizer getter function
    @property
    def optimizer_class(self):
        if self.optimizer == "SGD":
            return SGD
        if self.optimizer == "RMS":
            return RMSprop
        if self.optimizer == "Adam":
            return Adam
        if self.optimizer == "NAdam":
            return NAdam

if __name__ == "__main__":
    #base_path = Path(__file__).parent.parent
    base_path = '/zhome/ac/d/174101/thesis/data'
    h = Hyperparams(base_path / 'train_conf.toml')

    print(h.model_name())