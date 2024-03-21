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


SEED = 1
pl.seed_everything(SEED)
torch.set_float32_matmul_precision("high")
device = torch.device("cpu")
user_dir = "/data/m015k/data/bicycle"
MODEL_PATH = Path(os.path.join(user_dir, "models"))
PLOT_PATH = Path(os.path.join(user_dir, "plots"))
MODEL_PATH.mkdir(parents=True, exist_ok=True)
PLOT_PATH.mkdir(parents=True, exist_ok=True)

DATA_PATH = Path("/data/m015k/data/bicycle/data")
train_loader = torch.load(DATA_PATH / "nodags_data/control/training_data/train_loader.pth")
validation_loader = torch.load(DATA_PATH / "nodags_data/control/validation_data/validation_loader.pth")
test_loader = torch.load(DATA_PATH / "nodags_data/control/validation_data/test_loader.pth")
labels = np.load(DATA_PATH / "nodags_data/control/training_data/labels.npy", allow_pickle=True)

adata_genes = sc.read_h5ad(DATA_PATH / "SCP1064/ready/control/gene_filtered_adata.h5ad")
genes = adata_genes.var.index.tolist()

# Construct necessary matrices
gt_interv = torch.tensor(np.eye(61), dtype=torch.float32)

results = pd.DataFrame()
for filename in MODEL_PATH.glob("perturb_*/last.ckpt"):
    if float(str(filename).split("_")[8]) != 1:
        # print("- Skipping")
        continue
    if "_15_" not in str(filename):
        continue
    print(filename)
    print("LOADING")

    try:
        model = BICYCLE.load_from_checkpoint(checkpoint_path=filename, map_location=device, strict=True)
    except Exception as e:
        print(f"Could not load model {filename}: {e}")
        continue
    model.eval()

    maes = []
    name = "control"
    test_loader = torch.load(DATA_PATH / f"nodags_data/{name}/validation_data/test_loader.pth")
    for batch in test_loader:
        samples, intervention, sample_id, datatype = batch

        # Select samples from the test set
        samples_test = samples[datatype == 2]
        interventions_test = intervention[datatype == 2]
        id_test = sample_id[datatype == 2]
        acts = samples_test[np.arange(len(samples_test)), :]

        # Compute MAE of perturbation
        s = model.state_dict()["z_loc"][id_test]
        preds = s[np.arange(len(s)), :]
     
        # MAE
        mae = torch.mean(torch.mean(torch.abs(preds - acts), axis=0))  # / 61
        maes.extend(mae.detach().cpu().numpy().flatten().tolist())

    # # Average
    mae = np.mean(np.array(maes))
    print(mae)

    # ids = np.concatenate(ids)
    # datatypes = np.concatenate(datatypes)
    # test_ids = ids[datatypes == 2]

    # # Test predictions

    # # Get estimate of beta
    # beta_val = model.state_dict()["beta_val"]

    # n_genes = 61
    # mask = get_diagonal_mask(n_genes, device)
    # n_entries = (mask > 0.5).sum()
    # beta_idx = torch.where(mask > 0.5)

    # beta = torch.zeros((n_genes, n_genes), device=device)
    # beta[beta_idx[0], beta_idx[1]] = beta_val
    # # Threshold values with an abs value of greate than 0.5
    # # beta[torch.abs(beta) < 0.05] = 0
    # # Abs beta
    # beta = torch.abs(beta)

    # import matplotlib.pyplot as plt

    # vmin, vmax = beta.min(), beta.max()

    # fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    # im = ax[0].imshow(beta, cmap="bwr", vmin=vmin, vmax=vmax)
    # ax[0].set_xticks(range(n_genes))
    # ax[0].set_xticklabels(genes, rotation=90, size="xx-small")
    # ax[0].set_yticks(range(n_genes))
    # ax[0].set_yticklabels(genes, size="xx-small")
    # cbar = plt.colorbar(im)

    # im = ax[1].imshow(beta[21:25, 21:25], cmap="bwr", vmin=vmin, vmax=vmax)
    # ax[1].set_xticks(range(4))
    # ax[1].set_xticklabels(genes[21:25], rotation=90, size="xx-small")
    # ax[1].set_yticks(range(4))
    # ax[1].set_yticklabels(genes[21:25], size="xx-small")
    # cbar = plt.colorbar(im)
    # plt.show()
