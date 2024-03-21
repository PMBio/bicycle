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

scale_l1 = 0.0
SEED = 0
DATASET = "control"
RANDOMIZE = True
max_epochs_one = 50000
gene_start = 0
gene_end = 1
max_epochs_two = 100

OUT_PATH = Path("/g/stegle/ueltzhoe/bicycle/notebooks/experiments/frangieh/imae")

print('RANDOMIZE:',RANDOMIZE)
print(type(RANDOMIZE))

pl.seed_everything(SEED)
torch.set_float32_matmul_precision("high")
device = torch.device("cpu")
user_dir = "/g/stegle/ueltzhoe/bicycle/notebooks/experiments/frangieh"
MODEL_PATH = Path(os.path.join(user_dir, "models"))
PLOT_PATH = Path(os.path.join(user_dir, "plots"))
MODEL_PATH.mkdir(parents=True, exist_ok=True)
PLOT_PATH.mkdir(parents=True, exist_ok=True)
GPU_DEVICE = 0

# Load Count Data for Training
DATA_PATH = Path("/scratch/ueltzhoe/nodags_data_counts")
train_loader = torch.load(DATA_PATH / f"{DATASET}/training_data/train_loader.pth")
validation_loader = torch.load(DATA_PATH / f"{DATASET}/validation_data/validation_loader.pth")
test_loader = torch.load(DATA_PATH / f"{DATASET}/test_data/test_loader.pth")
labels = np.load(DATA_PATH / f"{DATASET}/labels.npy", allow_pickle=True)

# Load Original, Normalized Frangieh Data
ORIGINAL_DATA_PATH = Path("/scratch/ueltzhoe/nodags_data_normalized")
test_loader_normalized = torch.load(ORIGINAL_DATA_PATH / f"{DATASET}/test_data/test_loader.pth")
labels_normalized = np.load(DATA_PATH / f"{DATASET}/labels.npy", allow_pickle=True)

print('labels:',labels)
print('labels normalized:',labels_normalized)

# Frangieh Data
FRANGIEH_DATA_PATH = Path("/scratch/ueltzhoe/SCP1064")
adata_genes = sc.read_h5ad(FRANGIEH_DATA_PATH / f"ready/{DATASET}/gene_filtered_adata.h5ad")
genes = adata_genes.var.index.tolist()

print('Loaded:')
test_samples_counts, test_regimes_counts, test_idx_counts, _, = test_loader.dataset[:]

print('Test samples counts:', test_samples_counts)

test_samples_normalized, test_regimes_normalized, test_idx_normalized, _, = test_loader_normalized.dataset[:]

label_diff = 0

for cl, nl in zip(labels, labels_normalized):
    if cl != nl:
        print('Mismatch:',cl,nl)
        label_diff += 1
        
assert label_diff == 0, 'Labels were rearranged'
assert (test_regimes_counts - test_regimes_normalized).abs().sum() == 0, 'Splits might be different'
assert (test_idx_counts - test_idx_normalized).abs().sum() == 0, 'Splits might be different'

print('SIZE OF TEST DATA:',test_samples_counts.shape)

for i in range(min(3,test_samples_counts.shape[0])):
    print( test_samples_counts[i,:], test_samples_normalized[i,:] )
    print( torch.corrcoef( 
        torch.cat( 
            ( test_samples_counts[i,:].reshape(1,-1) , 
              torch.exp( test_samples_normalized[i,:].reshape(1,-1) ) - 1.0 ), 
            axis = 0 )
        ) 
    )
    
# Construct necessary matrices
gt_interv = np.zeros((61, 61))
for i in range(61):
    gt_interv[i, i] = 1
gt_interv = torch.tensor(gt_interv, dtype=torch.float32)

results = pd.DataFrame()

n = 0
n_max = 1

filename = MODEL_PATH / "nodags_split_False_adam_1024_True_61_0.1_1_0_1_1_0.001_0.1_1_0_1_1_0_False/last.ckpt"

print(filename)
print("LOADING")

shutil.copyfile(filename, OUT_PATH / f"temp_{DATASET}.ckpt")

model = BICYCLE.load_from_checkpoint(checkpoint_path=OUT_PATH / f"temp_{DATASET}.ckpt", map_location=device, strict=True)

print('Z_LOC_OPT:',model.z_loc.min(), model.z_loc.mean(), model.z_loc.max())
print('Z_SCALE_OPT:',model.z_scale.min(), model.z_scale.mean(), model.z_scale.max())

if RANDOMIZE == 1:
    z_loc_init = 0.1*torch.randn( model.z_loc.shape )
    z_scale_init = -1.0*torch.ones( model.z_scale.shape)    

    print('Z_LOC_INIT:',z_loc_init.min(), z_loc_init.mean(), z_loc_init.max())
    print('Z_SCALE_INIT:',z_scale_init.min(), z_scale_init.mean(), z_scale_init.max())

ref_idx = test_loader.dataset[0][2]

# N_SAMPLES X N_GENES TEST DATA MATRIX
data_mat = test_loader.dataset[:][0]
data_mat_normalized = test_loader_normalized.dataset[:][0]

marginal_counts = (torch.exp(data_mat_normalized) - 1).sum(axis = 1, keepdims = True)

print('MARGINAL COUNTS:', marginal_counts)

# N_SAMPLES x N_GENES MATRIX CONTAINING THE PREDICTION FOR EACH GENE AND SAMPLE BASED ON
# THE EXPRESSION OF ALL THE OTHER GENES
pred_mat = torch.zeros( data_mat.shape )
prior_pred_mat = torch.zeros( data_mat.shape )
single_gene_pred_mat = torch.zeros( data_mat.shape )

print('SHAPE OF PREDICTION MATRIX:')
print(pred_mat.shape)

# CONDITION FOR EACH CELL FROM THE TEST DATASET
pred_regime = test_loader.dataset[:][1]    

# N_SAMPLES x N_GENES MASK OF UN-INTERVENED GENES
pred_unintervened_mat = (1.0 - gt_interv[:,pred_regime]).transpose(0,1)

'''print('SHAPE OF UNINTERVENED-PREDICTION MATRIX')    
print(pred_unintervened_mat.shape)

print('TOTAL NUMBER OF ENTRIES IN MATRIX:')
print(pred_unintervened_mat.shape[0]*pred_unintervened_mat.shape[1])

print('NUMBER OF NONZERO MASKING ELEMENTS:')
print(pred_unintervened_mat.sum())'''

# INDEX INTO Z_LOC AND Z_SCALE FOR EACH CELL FROM THE TEST DATASET
pred_z_idx = test_loader.dataset[:][2]

for gene in range(61):
    prior_pred_mat[:,gene] = model.predict_percentages(test_loader.dataset[:])[:,gene]
prior_pred_mat = torch.log(prior_pred_mat*marginal_counts + 1)    
imae = torch.abs( (prior_pred_mat - data_mat_normalized)*pred_unintervened_mat ).sum() / pred_unintervened_mat.sum()
print('I-MAE for ALL GENES after INITIAL MODEL FIT:',imae)

if RANDOMIZE:
    model.z_loc.data = z_loc_init
    model.z_scale.data = z_scale_init
    for gene in range(61):
        prior_pred_mat[:,gene] = model.predict_percentages(test_loader.dataset[:])[:,gene]
    prior_pred_mat = torch.log(prior_pred_mat*marginal_counts + 1)    
    imae = torch.abs( (prior_pred_mat - data_mat_normalized)*pred_unintervened_mat ).sum() / pred_unintervened_mat.sum()
    print('I-MAE for ALL GENES after RANDOM INIT:',imae)

for gene in tqdm(range(gene_start, gene_end)): #tqdm(range(gt_interv.shape[0])):

    model = BICYCLE.load_from_checkpoint(checkpoint_path=OUT_PATH / f"temp_{DATASET}.ckpt", map_location=device, strict=True)
    model.optimizer = "adam"
    model.lr = 1e-3
    
    if RANDOMIZE == 1:
        model.z_loc.data = z_loc_init
        model.z_scale.data = z_scale_init

    model.train()
    
    for gene in range(61):
        prior_pred_mat[:,gene] = model.predict_percentages(test_loader.dataset[:])[:,gene]
    prior_pred_mat = torch.log(prior_pred_mat*marginal_counts + 1)    
    imae = torch.abs( (prior_pred_mat - data_mat_normalized)*pred_unintervened_mat ).sum() / pred_unintervened_mat.sum()
    print('PRE-TRAINING: I-MAE for ALL GENES after INITIAL MODEL FIT:',imae)

    trainer = pl.Trainer(
        max_epochs=max_epochs_one,
        accelerator="gpu",  # if str(device).startswith("cuda") else "cpu",
        #devices=[GPU_DEVICE],  # if str(device).startswith("cuda") else 1,
        num_sanity_val_steps=0,
        callbacks = [
            StochasticWeightAveraging(0.01, swa_epoch_start=250)
        ]
    )
    
    model.train_only_latents = True

    start_time = time.time()    
    trainer.fit(model, test_loader)    
    end_time = time.time()
    print(f"Training took {end_time - start_time:.2f} seconds")

    pred_mat[:,gene] = model.predict_percentages(test_loader.dataset[:])[:,gene]
    
for gene in range(61):
    prior_pred_mat[:,gene] = model.predict_percentages(test_loader.dataset[:])[:,gene]
prior_pred_mat = torch.log(prior_pred_mat*marginal_counts + 1)    
imae = torch.abs( (prior_pred_mat - data_mat_normalized)*pred_unintervened_mat ).sum() / pred_unintervened_mat.sum()
print('POST-TRAINING: I-MAE for ALL GENES after INITIAL MODEL FIT:',imae)
'''
pred_mat = torch.log(pred_mat*marginal_counts + 1)
np.save(OUT_PATH / f"pred_mat_{DATASET}_{scale_l1}_{SEED}_{gene_start}_{gene_end}.npy",pred_mat.detach().cpu().numpy())
np.save(OUT_PATH / f"data_mat_normalized_{DATASET}_{scale_l1}_{SEED}_{gene_start}_{gene_end}.npy",data_mat_normalized.detach().cpu().numpy())
np.save(OUT_PATH / f"pred_unintervened_mat_{DATASET}_{scale_l1}_{SEED}_{gene_start}_{gene_end}.npy",pred_unintervened_mat.detach().cpu().numpy())
np.save(OUT_PATH / f"marginal_counts_{DATASET}_{scale_l1}_{SEED}_{gene_start}_{gene_end}.npy",marginal_counts.detach().cpu().numpy())


imae = torch.abs( (pred_mat - data_mat_normalized)*pred_unintervened_mat ).sum() / pred_unintervened_mat.sum()
np.save(OUT_PATH / f"imae_{DATASET}_{scale_l1}_{SEED}_{gene_start}_{gene_end}.npy",imae.detach().cpu().numpy())
print('I-MAE for ALL GENES AFTER OPT:',imae)
'''