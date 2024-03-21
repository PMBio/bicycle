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
from bicycle.model_eval import BICYCLE_EVAL
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


SEED = 1
pl.seed_everything(SEED)
torch.set_float32_matmul_precision("high")
device = torch.device("cpu")
user_dir = "/g/stegle/ueltzhoe/frangieh/bicycle"
MODEL_PATH = Path(os.path.join(user_dir, "models"))
PLOT_PATH = Path(os.path.join(user_dir, "plots"))
MODEL_PATH.mkdir(parents=True, exist_ok=True)
PLOT_PATH.mkdir(parents=True, exist_ok=True)
GPU_DEVICE = 0

DATA_PATH = DATA_PATH = Path("/scratch/ueltzhoe/frangieh")
train_loader = torch.load(DATA_PATH / "nodags_data/control/training_data/train_loader.pth")
validation_loader = torch.load(DATA_PATH / "nodags_data/control/validation_data/validation_loader.pth")
test_loader = torch.load(DATA_PATH / "nodags_data/control/validation_data/test_loader.pth")
labels = np.load(DATA_PATH / "nodags_data/control/training_data/labels.npy", allow_pickle=True)

#### THIS SHOULD BE INCREASED TO ~1000, I THINK (MORE EXPERIMENTATION NEEDED)
max_epochs = 1

adata_genes = sc.read_h5ad(DATA_PATH / "SCP1064/ready/control/gene_filtered_adata.h5ad")
genes = adata_genes.var.index.tolist()

# Construct necessary matrices
gt_interv = np.zeros((61, 61))
for i in range(61):
    gt_interv[i, i] = 1
gt_interv = torch.tensor(gt_interv, dtype=torch.float32)

results = pd.DataFrame()
for filename in MODEL_PATH.glob("perturb_*/last.ckpt"):
    if float(str(filename).split("_")[8]) != 1:
        # print("- Skipping")
        continue
    if "_15_" not in str(filename):
        continue
    print(filename)
    print("LOADING")

    maes = []
    name = "control"
    test_loader = torch.load(DATA_PATH / f"nodags_data/{name}/validation_data/test_loader.pth")
    
    ref_idx = test_loader.dataset[0][2]
    
    print('REF_IDX:',ref_idx)
    
    # N_SAMPLES X N_GENES TEST DATA MATRIX
    data_mat = test_loader.dataset[:][0]
    
    # N_SAMPLES x N_GENES MATRIX CONTAINING THE PREDICTION FOR EACH GENE AND SAMPLE BASED ON
    # THE EXPRESSION OF ALL THE OTHER GENES
    pred_mat = torch.zeros( data_mat.shape )

    print('SHAPE OF PREDICTION MATRIX:')
    print(pred_mat.shape)

    # CONDITION FOR EACH CELL FROM THE TEST DATASET
    pred_regime = test_loader.dataset[:][1]    
    
    # N_SAMPLES x N_GENES MASK OF UN-INTERVENED GENES
    pred_unintervened_mat = (1.0 - gt_interv[:,pred_regime]).transpose(0,1)
    
    print('SHAPE OF UNINTERVENED-PREDICTION MATRIX')    
    print(pred_unintervened_mat.shape)
    
    print('TOTAL NUMBER OF ENTRIES IN MATRIX:')
    print(pred_unintervened_mat.shape[0]*pred_unintervened_mat.shape[1])
    
    print('NUMBER OF NONZERO MASKING ELEMENTS:')
    print(pred_unintervened_mat.sum())
    
    # INDEX INTO Z_LOC AND Z_SCALE FOR EACH CELL FROM THE TEST DATASET
    pred_z_idx = test_loader.dataset[:][2]
    
    for gene in tqdm(range(61)): #tqdm(range(gt_interv.shape[0])):
        
        model = BICYCLE_EVAL.load_from_checkpoint(checkpoint_path=filename, map_location=device, strict=True)
        model.pred_gene = gene
        model.train()
        
        z_before = model.z_loc[pred_z_idx,gene].detach()
        
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="gpu",  # if str(device).startswith("cuda") else "cpu",
            devices=[GPU_DEVICE],  # if str(device).startswith("cuda") else 1,
            num_sanity_val_steps=0,
        )
        
        try:
            start_time = time.time()
            trainer.fit(model, test_loader)
            end_time = time.time()
            print(f"Training took {end_time - start_time:.2f} seconds")
        except Exception as e:
            print(f"Training failed: {e}")
            continue
            
        z_after1 = model.z_loc[pred_z_idx,gene].detach()
                
        pred_mat[:,gene] = model.z_loc[pred_z_idx,gene]
        
        trainer2 = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="gpu",  # if str(device).startswith("cuda") else "cpu",
            devices=[GPU_DEVICE],  # if str(device).startswith("cuda") else 1,
            num_sanity_val_steps=0,
        )
        
        try:
            start_time = time.time()
            trainer2.fit(model, test_loader)
            end_time = time.time()
            print(f"Training took {end_time - start_time:.2f} seconds")
        except Exception as e:
            print(f"Training failed: {e}")
            continue
            
        z_after2 = model.z_loc[pred_z_idx,gene].detach()
        
        print('delta_z after first iter:')
        print(torch.pow(z_before - z_after1,2).mean())
        print('delta_z after second iter:')
        print(torch.pow(z_after2 - z_after1,2).mean())
        
    print(pred_mat)  
    
    print('I-MAE over UNINTERVENED GENES:')
    
    imae = torch.abs( (pred_mat - data_mat)*pred_unintervened_mat ).sum() / pred_unintervened_mat.sum()
    
    print(imae)

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
