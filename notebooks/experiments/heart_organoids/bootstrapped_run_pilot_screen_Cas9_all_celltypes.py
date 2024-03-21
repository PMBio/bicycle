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
    create_data,
    create_loaders,
    get_diagonal_mask,
    compute_inits,
)
from bicycle.utils.general import get_full_name
from bicycle.utils.plotting import plot_training_results
from pytorch_lightning.callbacks import RichProgressBar, StochasticWeightAveraging
from bicycle.callbacks import ModelCheckpoint, GenerateCallback, MyLoggerCallback, CustomModelCheckpoint
import numpy as np
import yaml
import sys
import pickle

if len(sys.argv) > 1:
    
    print('Using supplied argument for SEED!')
    print('Setting SEED to: ', int(sys.argv[1]))

    SEED = int(sys.argv[1])

else:
    
    SEED = 1

BOOTSTRAP_DATA = False

pl.seed_everything(SEED)
torch.set_float32_matmul_precision("highest")
device = torch.device("cpu")
if environ["USER"] == "m015k":
    user_dir = "/home/m015k/code/bicycle/notebooks/data"
else:
    user_dir = "."
MODEL_PATH = Path(os.path.join(user_dir, "models"))
PLOT_PATH = Path(os.path.join(user_dir, "plots"))
MODEL_PATH.mkdir(parents=True, exist_ok=True)
PLOT_PATH.mkdir(parents=True, exist_ok=True)

# LEARNING
lr = 1e-3
batch_size = 1024
USE_INITS = False
n_epochs = 50000
early_stopping = False
early_stopping_patience = 500
early_stopping_min_delta = 0.01
optimizer = "adam"
gradient_clip_val = 1.0
swa = 250
plot_epoch_callback = 1000

# MODEL
use_encoder = False
x_distribution = "Multinomial"
x_distribution_kwargs = {}
lyapunov_penalty = True
use_latents = True
sigma_min = 1e-3

GPU_DEVICES = 1
if GPU_DEVICES > 1:
    # Initialize Linalg Module, so you don't get
    # "lazy wrapper should be called at most once" error
    # Following: https://github.com/pytorch/pytorch/issues/90613
    torch.inverse(torch.ones((0, 0), device="cuda:0"))


SAVE_PLOT = True
CHECKPOINTING = False
VERBOSE_CHECKPOINTING = False
OVERWRITE = True
# REST
check_val_every_n_epoch = 1
log_every_n_steps = 1

covariates = None
correct_covariates = False

# DATA

#
# Load preprocessed Norman data
#

base_dir = './data/all_celltypes'
split_dir = 'Cas9'
data_dir = os.path.join(base_dir,split_dir)

samples = torch.tensor( np.load(os.path.join(data_dir,'samples.npy')), dtype = torch.float)
regimes = torch.tensor( np.load(os.path.join(data_dir,'regimes.npy')), dtype = torch.long )

train_gene_ko = np.unique( regimes )
test_gene_ko = []

print('Training and validation regimes:')
print( train_gene_ko )

print('Test regimes:')
print( test_gene_ko )

gt_interv = torch.tensor(np.load(os.path.join(data_dir,'gt_interv.npy')), dtype = torch.float)

with open(os.path.join(data_dir,'genes.pkl'), 'rb') as f:
    labels = pickle.load(f)

n_samples = samples.shape[0]
n_genes = samples.shape[1]
n_contexts = gt_interv.shape[1]

if BOOTSTRAP_DATA:
    print('BEFORE BOOTSTRAPPING:',samples.shape, samples[:3,:].max(axis = 1))
    samples = samples[torch.randint(n_samples,(n_samples,)),:]
    print('AFTER BOOTSTRAPPING:',samples.shape, samples[:3,:].max(axis = 1))

train_gene_ko = [str(g) for g in train_gene_ko]
test_gene_ko = [str(g) for g in test_gene_ko]

train_loader, validation_loader, test_loader = create_loaders( 
    samples,
    regimes,
    0,
    batch_size,
    SEED,
    train_gene_ko = train_gene_ko,
    test_gene_ko = test_gene_ko)
 
print(train_loader)

# RESULTS
name_prefix = f"PILOT_SCREEN_ALL_CELLTYPES_{split_dir}_SIGMA_MIN{sigma_min}_BOOTSTRAPPED_MAGPIE_batchsize{batch_size}_lyapunovpenalty{lyapunov_penalty}_latents{use_latents}_swa{swa}_seed{SEED}"

# Model
n_factors = 0 # Not using factorized beta
rank_w_cov_factor = n_genes # Fitting full-rank multivariate normals  #n_factors # Same as dictys: #min(TFs, N_GENES-1)
perfect_interventions = True

# Create Mask
mask = get_diagonal_mask(n_genes, device)

if n_factors > 0:
    mask = None

if USE_INITS:
    init_tensors = compute_inits(train_loader.dataset, rank_w_cov_factor, n_contexts)

print("Training data:")
print(f"- Number of training samples: {len(train_loader.dataset)}")

print('TORCH DEVICES:', torch.cuda.device_count())
#device = torch.device(f"cuda:{GPU_DEVICE}")
gt_interv = gt_interv.to(device)
n_genes = samples.shape[1]

if covariates is not None and correct_covariates:
    covariates = covariates.to(device)

for scale_kl in [1.0]:
    for scale_l1 in [1e-2]:
        for scale_spectral in [0.0]: 
            for scale_lyapunov in [1.0]:
                file_dir = get_full_name(
                    name_prefix,
                    0,
                    SEED,
                    lr,
                    n_genes,
                    scale_l1,
                    scale_kl,
                    scale_spectral,
                    scale_lyapunov,
                    gradient_clip_val,
                    swa,
                )

                # If final plot or final model exists: do not overwrite by default
                print("Checking Model and Plot files...")
                final_file_name = os.path.join(MODEL_PATH, file_dir, "last.ckpt")
                final_plot_name = os.path.join(PLOT_PATH, file_dir, "last.png")
                if (Path(final_file_name).exists() & SAVE_PLOT & ~OVERWRITE) | (
                    Path(final_plot_name).exists() & CHECKPOINTING & ~OVERWRITE
                ):
                    print("- Files already exists, skipping...")
                    continue
                else:
                    print("- Not all files exist, fitting model...")
                    print("  - Deleting dirs")
                    # Delete directories of files
                    if Path(final_file_name).exists():
                        print(f"  - Deleting {final_file_name}")
                        # Delete all files in os.path.join(MODEL_PATH, file_name)
                        for f in os.listdir(os.path.join(MODEL_PATH, file_dir)):
                            os.remove(os.path.join(MODEL_PATH, file_dir, f))
                    if Path(final_plot_name).exists():
                        print(f"  - Deleting {final_plot_name}")
                        for f in os.listdir(os.path.join(PLOT_PATH, file_dir)):
                            os.remove(os.path.join(PLOT_PATH, file_dir, f))

                    print("  - Creating dirs")
                    # Create directories
                    Path(os.path.join(MODEL_PATH, file_dir)).mkdir(parents=True, exist_ok=True)
                    Path(os.path.join(PLOT_PATH, file_dir)).mkdir(parents=True, exist_ok=True)

                model = BICYCLE(
                    lr,
                    gt_interv,
                    n_genes,
                    n_samples=n_samples,
                    lyapunov_penalty=lyapunov_penalty,
                    perfect_interventions=perfect_interventions,
                    rank_w_cov_factor=rank_w_cov_factor,
                    init_tensors=init_tensors if USE_INITS else None,
                    optimizer=optimizer,
                    device=device,
                    scale_l1=scale_l1,
                    scale_lyapunov=scale_lyapunov,
                    scale_spectral=scale_spectral,
                    scale_kl=scale_kl,
                    early_stopping=early_stopping,
                    early_stopping_min_delta=early_stopping_min_delta,
                    early_stopping_patience=early_stopping_patience,
                    early_stopping_p_mode=True,
                    x_distribution=x_distribution,
                    x_distribution_kwargs=x_distribution_kwargs,
                    mask=mask,
                    use_encoder=use_encoder,
                    gt_beta=None,
                    train_gene_ko=train_gene_ko,
                    test_gene_ko=test_gene_ko,
                    use_latents=use_latents,
                    covariates=covariates,
                    n_factors = n_factors,
                    sigma_min = sigma_min,
                    intervention_type = split_dir
                )
                model.to(device)

                dlogger = DictLogger()
                loggers = [dlogger]

                callbacks = [
                    RichProgressBar(refresh_rate=1),
                    GenerateCallback(
                        final_plot_name, 
                        plot_epoch_callback=plot_epoch_callback,
                        labels=labels
                    ),
                ]
                if swa > 0:
                    callbacks.append(StochasticWeightAveraging(0.01, swa_epoch_start=swa))
                if CHECKPOINTING:
                    Path(os.path.join(MODEL_PATH, file_dir)).mkdir(parents=True, exist_ok=True)
                    callbacks.append(
                        CustomModelCheckpoint(
                            dirpath=os.path.join(MODEL_PATH, file_dir),
                            filename="{epoch}",
                            save_last=True,
                            save_top_k=1,
                            verbose=VERBOSE_CHECKPOINTING,
                            monitor="valid_loss",
                            mode="min",
                            save_weights_only=True,
                            start_after=0,
                            save_on_train_epoch_end=False,
                            every_n_epochs=1,
                        )
                    )
                    callbacks.append(MyLoggerCallback(dirpath=os.path.join(MODEL_PATH, file_dir)))

                trainer = pl.Trainer(
                    max_epochs=n_epochs,
                    accelerator="gpu",  # if str(device).startswith("cuda") else "cpu",
                    strategy = "dp",
                    logger=loggers,
                    log_every_n_steps=log_every_n_steps,
                    enable_model_summary=True,
                    enable_progress_bar=True,
                    enable_checkpointing=CHECKPOINTING,
                    check_val_every_n_epoch=check_val_every_n_epoch,
                    devices=GPU_DEVICES,  # if str(device).startswith("cuda") else 1,
                    num_sanity_val_steps=0,
                    callbacks=callbacks,
                    gradient_clip_val=gradient_clip_val,
                    default_root_dir=str(MODEL_PATH),
                    gradient_clip_algorithm="value",
                    deterministic=False, #"warn",
                )

                # try:
                start_time = time.time()
                # assert False
                trainer.fit(model, train_loader, validation_loader)
                end_time = time.time()
                print(f"Training took {end_time - start_time:.2f} seconds")
                
                with torch.no_grad():
                    model.eval()
                    if model.mask is None:
                        if model.n_factors == 0:
                            estimated_beta = model.beta.detach().cpu().numpy()
                        else:
                            estimated_beta = torch.einsum("ij,jk->ik", model.gene2factor, model.factor2gene).detach().cpu().numpy() 
                    else:
                        estimated_beta = torch.zeros(
                            (model.n_genes, model.n_genes), device=model.device
                        )
                        estimated_beta[model.beta_idx[0], model.beta_idx[1]] = model.beta_val
                        estimated_beta = estimated_beta.detach().cpu().numpy()
                
                plot_training_results(
                    trainer,
                    model,
                    estimated_beta,
                    None,
                    scale_l1,
                    scale_kl,
                    scale_spectral,
                    scale_lyapunov,
                    final_plot_name,
                    callback=False,
                )
                # except Exception as e:
                #     # Write Exception to file
                #     report_path = os.path.join(MODEL_PATH, file_dir, "report.yaml")
                #     # Write yaml
                #     with open(report_path, "w") as outfile:
                #         yaml.dump({"exception": str(e)}, outfile, default_flow_style=False)
