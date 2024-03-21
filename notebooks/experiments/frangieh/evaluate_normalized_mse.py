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
user_dir = "/g/stegle/ueltzhoe/bicycle_main/bicycle/notebooks/experiments/frangieh"
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
train_loader_normalized = torch.load(ORIGINAL_DATA_PATH / f"{DATASET}/training_data/train_loader.pth")
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

#filename = MODEL_PATH / "nodags_split_False_rmsprop_1024_True_61_0.1_1_0_0.1_1_0.001_0.1_1_0_0.1_1_0_False/last.ckpt"
filename = MODEL_PATH / "nodags_split_False_rmsprop_1024_True_61_0.01_1_0_0.1_1_0.001_0.01_1_0_0.1_1_0_False/epoch=9499.ckpt"#last.ckpt"

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

train_data_mat_normalized = train_loader_normalized.dataset[:][0]

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



prior_pred_mat = model.predict_mean_percentages(test_loader.dataset[:])
prior_pred_mat = torch.log(prior_pred_mat*marginal_counts + 1)    

print('PRIOR PRED MAT:', prior_pred_mat.shape)
print(prior_pred_mat[:10])

print('NORMALIZED DATA MAT:', prior_pred_mat.shape)
print(data_mat_normalized[:10])

mse = torch.pow( (prior_pred_mat - data_mat_normalized)*pred_unintervened_mat, 2 ).sum() / pred_unintervened_mat.sum()

pred_base = train_data_mat_normalized.mean(axis = 0, keepdims = True)
mse_base = torch.pow( (pred_base - data_mat_normalized)*pred_unintervened_mat, 2 ).sum() / pred_unintervened_mat.sum()
print('MSE for OOD MEAN PREDICTION:',mse)
print('MSE USING DATASET MEAN (no unintervened condition in these splits, otherwise use only unperturbed cells):',mse_base)

'''
if RANDOMIZE:
    model.z_loc.data = z_loc_init
    model.z_scale.data = z_scale_init
    prior_pred_mat = model.predict_mean_percentages(test_loader.dataset[:])
    prior_pred_mat = torch.log(prior_pred_mat*marginal_counts + 1)    
    mse = torch.pow( (prior_pred_mat - data_mat_normalized)*pred_unintervened_mat, 2 ).sum() / pred_unintervened_mat.sum()
    print('MSE for ALL GENES after RANDOM INIT:',mse)
'''