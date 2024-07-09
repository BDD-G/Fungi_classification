The directory structure of the project looks like this:
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