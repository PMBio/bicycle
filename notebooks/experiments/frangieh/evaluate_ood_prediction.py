import warnings

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", category=FutureWarning)

import time
import os
from pathlib import Path
from os import environ
import pytorch_lightning as pl
import torch
from bicycle.dictlogger import DictLogger
from bicycle.model import BICYCLE
from bicycle.utils.data import (
    get_diagonal_mask,
    compute_inits,
)
from bicycle.utils.plotting import plot_training_results
from pytorch_lightning.callbacks import RichProgressBar, StochasticWeightAveraging
from bicycle.callbacks import CustomModelCheckpoint, GenerateCallback, MyLoggerCallback
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm.auto import tqdm
import shutil
import sys

n_genes = 5
n_samples = 1000

device = "cpu"

gt_interv = torch.zeros((n_genes,n_genes + 1))
for i in range(n_genes):
    gt_interv[i,i+1] = 1.0
    
model = BICYCLE(
    0.001,
    gt_interv,
    n_genes,
    n_samples=n_samples,
    lyapunov_penalty=True,
    perfect_interventions=True,
    rank_w_cov_factor=n_genes,
    optimizer="rmsprop",
    device=device,
    x_distribution="Multinomial",
    intervention_type ="dCas9",
)

model.to(device)
model.train()
model.predict_perturbation([0,1],[0.3,1.1],[0.01,0.01])