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


SEED = 1
pl.seed_everything(SEED)
torch.set_float32_matmul_precision("high")
device = torch.device("cpu")
user_dir = "/data/m015k/data/bicycle"
MODEL_PATH = Path(os.path.join(user_dir, "models"))
PLOT_PATH = Path(os.path.join(".", "plots"))
MODEL_PATH.mkdir(parents=True, exist_ok=True)
PLOT_PATH.mkdir(parents=True, exist_ok=True)

DATASET = "control"

DATA_PATH = Path("/data/m015k/data/bicycle/data")
train_loader = torch.load(DATA_PATH / f"nodags_data/{DATASET}/training_data/train_loader.pth")
validation_loader = torch.load(DATA_PATH / f"nodags_data/{DATASET}/validation_data/validation_loader.pth")
test_loader = torch.load(DATA_PATH / f"nodags_data/{DATASET}/validation_data/test_loader.pth")
labels = np.load(DATA_PATH / f"nodags_data/{DATASET}/training_data/labels.npy", allow_pickle=True)

# Construct necessary matrices
gt_interv = torch.tensor(np.eye(61), dtype=torch.float32)

# for batch in train_loader:
#     samples, _, _, _ = batch
#     break

# # Compute Covariance matrix of samples
# cov = torch.matmul(samples.T, samples) / len(samples)
# # Compute rank of cov
# rank = torch.linalg.matrix_rank(cov)
# # Compute SVD of cov
# u, s, v = torch.linalg.svd(cov)
# # Plot s
# import matplotlib.pyplot as plt
# plt.scatter(range(61), s)
# s

# results = pd.DataFrame()
# for filename in MODEL_PATH.glob("perturbseq_*/*.ckpt"):
#     print(filename)
#     try:
#         model = BICYCLE.load_from_checkpoint(checkpoint_path=filename, map_location=device, strict=True)
#     except Exception as e:
#         print(f"Could not load model {filename}: {e}")
#         continue
#     model.eval()

#     trainer = pl.Trainer(
#         accelerator="gpu",
#         enable_model_summary=False,
#         enable_progress_bar=False,
#         devices=[1],
#     )

#     try:
#         predictions = sum(trainer.predict(model, test_loader)) / len(test_loader)
#         predictions = predictions.item()
#         print(f"Loss Test: {predictions:.2f}")
#     except Exception as e:
#         print(f"Could not predict test set: {e}")
#         predictions = np.nan

#     results = pd.concat(
#         [results, pd.DataFrame({"filename": str(filename), "loss": predictions}, index=[0])], axis=0
#     )

# results["loss"].median()


#
# Settings
#

# DGP
n_genes = 61  # Number of modelled genes
rank_w_cov_factor = 15  # n_genes - 1  # Same as dictys: #min(TFs, N_GENES-1)
n_contexts = n_genes  # Number of contexts
# LEARNING
lr = 1e-3
batch_size = 1024
USE_INITS = False
use_encoder = False
n_epochs = 40000
early_stopping = False
early_stopping_patience = 500
early_stopping_min_delta = 0.01
optimizer = "adam"
gradient_clip_val = 1
swa = 0
x_distribution = "Normal"  # "Poisson" if "random" in graph else
# DATA
validation_size = 0.0
# MODEL
lyapunov_penalty = True
GPU_DEVICE = 1
plot_epoch_callback = 1000
use_latents = True
# RESULTS
name_prefix = f"perturb_seq_{use_encoder}_{optimizer}_{batch_size}_{lyapunov_penalty}_{rank_w_cov_factor}"
SAVE_PLOT = True
CHECKPOINTING = True
VERBOSE_CHECKPOINTING = True
OVERWRITE = False
# REST
check_val_every_n_epoch = 1
log_every_n_steps = 1


#
# Create Mask
#
mask = get_diagonal_mask(n_genes, device)

init_tensors = {}
if USE_INITS:
    init_tensors = compute_inits(train_loader.dataset, rank_w_cov_factor, n_contexts, normalized=True)

print(f"Number of training samples: {len(train_loader.dataset)}")
if validation_size > 0:
    print(f"Number of validation samples: {len(validation_loader.dataset)}")
print(f"Number of test samples: {len(test_loader.dataset)}")

# gt_interv = torch.zeros((n_genes, n_genes + 1), device=device)
device = torch.device(f"cuda:{GPU_DEVICE}")
gt_interv = gt_interv.to(device)

for scale_kl in [1]:
    for scale_l1 in [0.1, 1]:
        for scale_spectral in [0, 1]:
            for scale_lyapunov in [0.1, 1, 10]:
                file_dir = (
                    name_prefix
                    + f"_{scale_l1}_{scale_kl}_{scale_spectral}_{scale_lyapunov}_{SEED}_{lr}_{scale_l1}_{scale_kl}_{scale_spectral}_{scale_lyapunov}_{gradient_clip_val}_{swa}_{USE_INITS}"
                )
                print(file_dir)

                # If final plot or final model exists: do not overwrite by default
                final_file_name = os.path.join(MODEL_PATH, file_dir, "last.ckpt")
                final_plot_name = os.path.join(PLOT_PATH, file_dir, "last.png")
                if (Path(final_file_name).exists() & SAVE_PLOT & ~OVERWRITE) | (
                    Path(final_plot_name).exists() & CHECKPOINTING & ~OVERWRITE
                ):
                    print("Files already exists, skipping...")
                    continue
                else:
                    print("Files do not exist, fitting model...")
                    print("Deleting dirs")
                    # Delete directories of files
                    if Path(final_file_name).exists():
                        print(f"Deleting {final_file_name}")
                        # Delete all files in os.path.join(MODEL_PATH, file_name)
                        for f in os.listdir(os.path.join(MODEL_PATH, file_dir)):
                            os.remove(os.path.join(MODEL_PATH, file_dir, f))
                    if Path(final_plot_name).exists():
                        print(f"Deleting {final_plot_name}")
                        for f in os.listdir(os.path.join(PLOT_PATH, file_dir)):
                            os.remove(os.path.join(PLOT_PATH, file_dir, f))

                    print("Creating dirs")
                    # Create directories
                    Path(os.path.join(MODEL_PATH, file_dir)).mkdir(parents=True, exist_ok=True)
                    Path(os.path.join(PLOT_PATH, file_dir)).mkdir(parents=True, exist_ok=True)

                model = BICYCLE(
                    lr,
                    gt_interv,
                    n_genes,
                    n_samples=len(train_loader.dataset),
                    lyapunov_penalty=lyapunov_penalty,
                    perfect_interventions=True,
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
                    mask=mask,
                    use_encoder=use_encoder,
                    gt_beta=None,
                    use_latents=use_latents,
                )
                model.to(device)

                dlogger = DictLogger()
                loggers = [dlogger]

                callbacks = [
                    RichProgressBar(refresh_rate=1),
                    GenerateCallback(
                        final_plot_name,
                        plot_epoch_callback=plot_epoch_callback,
                        true_beta=None,
                        labels=labels,
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
                            monitor="train_loss", ### FIXME: valid_loss
                            mode="min",
                            save_weights_only=True,
                            start_after=1000,
                            save_on_train_epoch_end=True, ### FIXME: False
                            every_n_epochs=500,
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
                )

                try:
                    start_time = time.time()
                    trainer.fit(model, train_loader, validation_loader)
                    end_time = time.time()
                    print(f"Training took {end_time - start_time:.2f} seconds")
                except Exception as e:
                    print(f"Training failed: {e}")
                    continue

                plot_training_results(
                    trainer,
                    model,
                    model.beta.detach().cpu().numpy(),
                    None,
                    scale_l1,
                    scale_kl,
                    scale_spectral,
                    scale_lyapunov,
                    final_plot_name,
                    callback=False,
                )

#


# # #
# # # TEST SET PREDICTIONS
# # #
# scale_l1 = 0.1
# scale_kl = 1
# scale_spectral = 1
# scale_lyapunov = 0
# file_name = get_name(
#     name_prefix,
#     len(LOGO),
#     SEED,
#     lr,
#     n_genes,
#     scale_l1,
#     scale_kl,
#     scale_spectral,
#     scale_lyapunov,
#     gradient_clip_val,
#     swa,
# )
# final_file_name = os.path.join(MODEL_PATH, file_name, "last.ckpt")

# if Path(final_file_name).exists():
#     print("\n\nRunning test set predictions...")
#     model = BICYCLE.load_from_checkpoint(checkpoint_path=final_file_name)
#     model.eval()

#     trainer = pl.Trainer(
#         accelerator="gpu",
#         enable_model_summary=False,
#         enable_progress_bar=False,
#         devices=[GPU_DEVICE],
#     )

#     if not test_loader:
#         print("No test set, skipping...")
#     else:
#         predictions = trainer.predict(model, test_loader)
#         predictions = sum(predictions) / len(predictions)
#         print(f"Loss Test: {predictions:.2f}")

#     predictions = trainer.predict(model, train_loader)
#     predictions = sum(predictions) / len(predictions)
#     print(f"Loss Train: {predictions:.2f}")
#     predictions = trainer.predict(model, validation_loader)
#     predictions = sum(predictions) / len(predictions)
#     print(f"Loss Valid: {predictions:.2f}")

# # Load parquet file
# import pandas as pd

# pd.read_parquet(os.path.join(MODEL_PATH, file_name, "logger.parquet"))
