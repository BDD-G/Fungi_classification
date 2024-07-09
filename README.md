# Fungi-classifcation

This project aims to accelarate Fungal indetification performed in BioEngineering department at DTU using Deep Learning.

## Setup

### Setup environment

First create a conda virtual environment and activate it:

```bash
conda create --name <my-env>
```

Then install the requirements:

```bash
# to install requirements
conda install --yes --file requirements.txt
```

The project currently uses the pytorch nightly build for Python 3.12 compatibility.

### Dataset

The dataset used for training and evaluating the models is a part of a research conducted at biolab at DTU, hence it's not avaiable for public use.

## Project structure

The project structure can be seen in the [structure.md](/data/structure.md) file:

```txt

├── README.md                       <- The top-level README for developers using this project.
├── data                            <- Basic data - configuration files, label files.
│
│
├── best_model                      <- Trained and serialized best performing model, model predictions, or model summaries.           
│
├── plots                
│   ├── Learning_Curves             <- Learning curves of all trained models.
│   │
│   └── Confusion_Matrices          <- Confusion matrices of all evaluated models.
│
├── requirements.txt                <- The requirements file for reproducing the analysis environment.
│
├── src                             <- Source code for use in this project.
│   ├── api
│   │   │       
│   │   ├── templates               <- Folder with html templates.
│   │   └── main.py                 <- Main api file.
│   ├── utils                       <- Convenience functions for converting, formatting, etc.
│   │   │
│   │   ├── CombineChannels.py      <- Helper class to create 4 or 5 channel images.
│   │   ├── DenseNet.py             <- DenseNet implementation.
│   │   ├── Hyperparams.py          <- Hyperparameters.
│   │   ├── loss.py                 <- Loss functions.
│   │   ├── ResNet.py               <- ResNet implementation.
│   │   ├── train.py                <- Training loop.
│   │   └── ViT.py                  <- Visual transformers implemetation.
│   │
│   ├── preprocessing               <- Image preprocessing files
│   │   │
│   │   ├── Image_gradient.py       <- Script that creates an image gradient dataset.
│   │   ├── LBP.py                  <- Script that creates a local binary patterns dataset.
│   │   └── Resizing.py             <- Script that creates a resized dataset.
│   │
│   ├── notebooks                   <- Jupyter notebooks.
│   │   │
│   │   ├── DataDistrubtion.ipynb   <- Different visualization regarding the dataset.
│   │   ├── mean&std.ipynb          <- Calculates mean and standard deviation of a dataset.
│   │   └── Model_evaluation.ipynb  <- Checks the model performence on unseen data.
│   │
│   └── main.py                     <- Script for training the model.
└── LICENSE                         <- Open-source license.
'''