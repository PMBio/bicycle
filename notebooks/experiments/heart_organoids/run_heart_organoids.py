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

SEED = 1
pl.seed_everything(SEED)
torch.set_float32_matmul_precision("high")
device = torch.device("cpu")
if environ["USER"] == "m015k":
    user_dir = "/home/m015k/code/bicycle/notebooks/data"
else:
    user_dir = "."
MODEL_PATH = Path(os.path.join(user_dir, "models"))
PLOT_PATH = Path(os.path.join(user_dir, "plots"))
MODEL_PATH.mkdir(parents=True, exist_ok=True)
PLOT_PATH.mkdir(parents=True, exist_ok=True)

cell_type = 'CM' #, Choose from: ['Fib.', 'Prog.', 'CM']

# Model
n_factors = 50
rank_w_cov_factor = n_factors # Same as dictys: #min(TFs, N_GENES-1)
perfect_interventions = True

# LEARNING
lr = 1e-5
batch_size = 128#4096
USE_INITS = True
use_encoder = False
n_epochs = 51000
early_stopping = False
early_stopping_patience = 500
early_stopping_min_delta = 0.01
optimizer = "adam"
gradient_clip_val = 10.0
swa = 250
x_distribution = "Multinomial"
x_distribution_kwargs = {}

# MODEL
lyapunov_penalty = True
GPU_DEVICE = 0
plot_epoch_callback = 1000
use_latents = False
# RESULTS
name_prefix = f"HEART-ORGANOID-TEST-2_{cell_type}_{use_encoder}_{batch_size}_{lyapunov_penalty}"
SAVE_PLOT = True
CHECKPOINTING = False
VERBOSE_CHECKPOINTING = False
OVERWRITE = True
# REST
check_val_every_n_epoch = 1
log_every_n_steps = 1

covariates = None
correct_covariates = False

# N_SAMPLES X N_GENES DATA MATRIX
samples = torch.tensor(np.load("/g/stegle/ueltzhoe/bicycle/notebooks/data_preprocessing/heart_organoids/data_" + cell_type + ".npy"))

print('SAMPLES.shape:',samples.shape)

n_samples = samples.shape[0]
n_genes = samples.shape[1]

# ALL CELLS ARE UNPERTURBED
regime = torch.zeros( (samples.shape[0],1), dtype = torch.int)

# N_GENES x N_CONTEXTS (in this case, there is only a SINGLE, UNPERTURBED CONTEXT)!
gt_interv = torch.zeros( (n_genes,1), dtype = torch.int )
n_contexts = 1

train_gene_ko = []
test_gene_ko = []

train_loader, validation_loader, test_loader = create_loaders( 
    samples,
    regime,
    0,
    batch_size,
    SEED,
    train_gene_ko = train_gene_ko,
    test_gene_ko = test_gene_ko)
 
print(train_loader)

# Create Mask
mask = get_diagonal_mask(n_genes, device)

if n_factors > 0:
    mask = None

if USE_INITS:
    init_tensors = compute_inits(train_loader.dataset, rank_w_cov_factor, n_contexts)

print("Training data:")
print(f"- Number of training samples: {len(train_loader.dataset)}")

device = torch.device(f"cuda:{GPU_DEVICE}")
gt_interv = gt_interv.to(device)
n_genes = samples.shape[1]

if covariates is not None and correct_covariates:
    covariates = covariates.to(device)

for scale_kl in [1.0]:  # 1
    for scale_l1 in [1000.0]:
        for scale_spectral in [0.0]: # 1.0
            for scale_lyapunov in [1000.0]: # 0.1
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
                    n_factors = n_factors
                )
                model.to(device)

                dlogger = DictLogger()
                loggers = [dlogger]

                callbacks = [
                    RichProgressBar(refresh_rate=1),
                    GenerateCallback(
                        final_plot_name, plot_epoch_callback=plot_epoch_callback
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
                    logger=loggers,
                    log_every_n_steps=log_every_n_steps,
                    enable_model_summary=True,
                    enable_progress_bar=True,
                    enable_checkpointing=CHECKPOINTING,
                    check_val_every_n_epoch=check_val_every_n_epoch,
                    devices=[GPU_DEVICE],  # if str(device).startswith("cuda") else 1,
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

                plot_training_results(
                    trainer,
                    model,
                    model.beta.detach().cpu().numpy(),
                    beta,
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
