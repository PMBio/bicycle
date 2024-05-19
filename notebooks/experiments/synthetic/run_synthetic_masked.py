import warnings

warnings.filterwarnings("ignore", ".*does not have many workers.*")

import os
import pdb
import sys
import time
import traceback
from os import environ
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
from bicycle.dictlogger import DictLogger
from bicycle.model import BICYCLE
from bicycle.utils.data import (
    compute_inits,
    create_data,
    create_loaders,
    get_diagonal_mask,
    get_names,
    get_ring_mask,
)
from bicycle.utils.training import GenerateCallback
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    RichProgressBar,
    StochasticWeightAveraging,
)

SEED = 3141
pl.seed_everything(SEED)
torch.set_float32_matmul_precision("high")
device = torch.device("cpu")
if environ["USER"] == "m015k":
    user_dir = "/home/m015k/code/bicycle/notebooks/data"
else:
    user_dir = "."
DATA_PATH = Path(user_dir)
MODEL_PATH = Path(user_dir)
PLOT_PATH = Path(os.path.join(user_dir, "plots"))

#
# Settings
#
batch_size = 1024
n_genes = 5  # 20  #10  # Number of modelled genes
rank_w_cov_factor = n_genes - 1  # Same as dictys: #min(TFs, N_GENES-1)
n_contexts = n_genes + 1  # Number of contexts
n_activators = 10  # 20 # 10
n_repressors = 6  # 12 # 6
USE_INITS = False
n_samples_control = 500
n_samples_per_perturbation = 250
model_count_data = True
dgp_ring = True
add_mask = False
n_additional_entries = 12
use_encoder = True
n_gpus = 4

lr = 1e-3
n_epochs = 30_000
log_every_n_steps = 1
check_val_every_n_epoch = 1
perfect_interventions = True
early_stopping_patience = 500
early_stopping_min_delta = 0.01
SAVE_PLOT = True

normalise = False
optimizer = "adam"
early_stopping = False
lyapunov_penalty = True
validation_size = 0.2
LOGO = [4]
train_gene_ko = [str(x) for x in set(range(0, n_genes)) - set(LOGO)]  # We start counting at 0
test_gene_ko = [str(x) for x in LOGO]  # + ["3,5"]
n_samples_total = n_samples_control + (len(train_gene_ko) + len(test_gene_ko)) * n_samples_per_perturbation
SAVE_PREFIX = f"v1_encoder_{optimizer}_{normalise}_{lr}_{batch_size}_{lyapunov_penalty}"
GPU_DEVICE = 0
SAVE_MODEL = True
OVERWRITE = True
every_n_epochs = 1000

#
# Create Mask
#
if add_mask:
    if dgp_ring:
        mask = get_ring_mask(n_additional_entries, n_genes, device)
    else:
        raise NotImplementedError("Mask only implemented for DGP Ring")
else:
    mask = get_diagonal_mask(n_genes, device)

#
# Create synthetic data
#
gt_dyn, intervened_variables, samples, gt_interv, sim_regime, beta = create_data(
    n_genes,
    n_samples_control=n_samples_control,
    n_samples_per_perturbation=n_samples_per_perturbation,
    n_activators=n_activators,
    n_repressors=n_repressors,
    device=device,
    make_counts=model_count_data,
    dgp_ring=dgp_ring,
    train_gene_ko=train_gene_ko,
    test_gene_ko=test_gene_ko,
)

train_loader, validation_loader, test_loader = create_loaders(
    samples,
    sim_regime,
    validation_size,
    batch_size,
    LOGO,
    SEED,
    train_gene_ko,
    test_gene_ko,
)

if USE_INITS:
    init_tensors = compute_inits(train_loader.dataset, rank_w_cov_factor, n_contexts)

print(f"Number of training samples: {len(train_loader.dataset)}")
if validation_size > 0:
    print(f"Number of validation samples: {len(validation_loader.dataset)}")
if LOGO:
    print(f"Number of test samples: {len(test_loader.dataset)}")

device = torch.device(f"cuda:{GPU_DEVICE}")
gt_interv = gt_interv.to(device)
n_genes = samples.shape[1]


for scale_spectral in [0]:  # 1
    for scale_lyapunov in [1, 10]:  # 1
        for scale_l1 in [0.1, 0.5, 1, 5, 10]:  # 0.1,
            for scale_kl in [1]:  # , 10, 0.1
                print(f"Scale Lyapunov: {scale_lyapunov}")
                print(f"Scale spectral loss: {scale_spectral}")

                if scale_lyapunov != 0 and scale_spectral != 0:
                    # Either scale_spectral or scale_lyapunov can be 0
                    # We don't need both of them at the same time:
                    # One is to ensure a proper solution of the direct solver
                    # the other is to replace the direct solver
                    raise ValueError("Only one of scale_lyapunov or scale_spectral can be non-zero")

                file_name_model, file_name_plot = get_names(
                    SAVE_PREFIX,
                    n_genes,
                    scale_l1,
                    scale_kl,
                    scale_spectral,
                    scale_lyapunov,
                    SEED,
                    PLOT_PATH,
                    MODEL_PATH,
                )

                if (Path(file_name_plot).exists() & SAVE_PLOT & ~OVERWRITE) | (
                    Path(file_name_model).exists() & SAVE_MODEL & ~OVERWRITE
                ):
                    print(f"File {file_name_plot} already exists, skipping...")
                    continue
                else:
                    print(f"File {file_name_plot} does not exist, fitting model...")

                model = BICYCLE(
                    lr,
                    gt_interv,
                    n_genes,
                    n_samples=n_samples_total,
                    lyapunov_penalty=lyapunov_penalty,
                    perfect_interventions=perfect_interventions,
                    rank_w_cov_factor=rank_w_cov_factor,
                    init_tensors=init_tensors if USE_INITS else None,
                    optimizer=optimizer,
                    device=device,
                    normalise=normalise,
                    scale_l1=scale_l1,
                    scale_lyapunov=scale_lyapunov,
                    scale_spectral=scale_spectral,
                    scale_kl=scale_kl,
                    early_stopping=early_stopping,
                    early_stopping_min_delta=early_stopping_min_delta,
                    early_stopping_patience=early_stopping_patience,
                    early_stopping_p_mode=True,
                    model_latents=model_count_data,
                    x_distribution="Poisson",
                    mask=mask,
                    use_encoder=use_encoder,
                )
                model.to(device)

                dlogger = DictLogger()
                loggers = [dlogger]

                # checkpoint_callback = ModelCheckpoint(
                #     filepath=str(MODEL_PATH),
                #     save_top_k=1,
                #     verbose=True,
                #     monitor='train_loss',
                #     mode='min',
                #     prefix=''
                # )

                trainer = pl.Trainer(
                    max_epochs=n_epochs,
                    accelerator="cpu" if n_gpus < 1 else "gpu",
                    gpus=n_gpus,
                    strategy="dp" if n_gpus > 1 else "auto",
                    logger=loggers,
                    log_every_n_steps=log_every_n_steps,
                    enable_model_summary=True,
                    enable_progress_bar=True,
                    enable_checkpointing=False,
                    check_val_every_n_epoch=check_val_every_n_epoch,
                    devices=[GPU_DEVICE] if str(device).startswith("cuda") else 1,
                    num_sanity_val_steps=0,
                    callbacks=[
                        RichProgressBar(refresh_rate=1),
                        GenerateCallback(
                            file_name_plot, every_n_epochs=every_n_epochs, true_beta=beta.cpu().numpy()
                        ),
                        # ModelCheckpoint(),
                        StochasticWeightAveraging(0.01, swa_epoch_start=250),
                    ],
                    gradient_clip_val=0.1,
                    default_root_dir=str(MODEL_PATH),
                )

                try:
                    start_time = time.time()
                    trainer.fit(model, train_loader, validation_loader)
                    end_time = time.time()
                    print(f"Training took {end_time - start_time:.2f} seconds")

                    if SAVE_MODEL:
                        trainer.save_checkpoint(file_name_model)

                    # Plot training curve
                    fig, ax = plt.subplots(2, 2, figsize=(17.5, 11))
                    df_plot = pd.DataFrame(trainer.logger.history).reset_index(drop=True)
                    df_plot["epoch"] = df_plot.index
                    df_plot_train = df_plot[[x for x in df_plot.columns if "train" in x] + ["epoch"]]
                    df_plot_valid = df_plot[[x for x in df_plot.columns if "valid" in x] + ["epoch"]]

                    df_plot_train = df_plot_train.melt(
                        id_vars=["epoch"], value_vars=[x for x in df_plot.columns if "train_" in x]
                    )
                    df_plot_valid = df_plot_valid.melt(
                        id_vars=["epoch"], value_vars=[x for x in df_plot.columns if "valid_" in x]
                    )
                    sns.scatterplot(
                        df_plot_train,
                        x="epoch",
                        y="value",
                        hue="variable",
                        ax=ax[1, 0],
                        s=10,
                        edgecolor="none",
                        linewidth=0,
                    )
                    sns.scatterplot(
                        df_plot_valid,
                        x="epoch",
                        y="value",
                        hue="variable",
                        ax=ax[1, 1],
                        s=10,
                        edgecolor="none",
                        linewidth=0,
                    )
                    ax[1, 0].grid(True)
                    ax[1, 1].grid(True)
                    ax[1, 0].set_title("Training")
                    ax[1, 1].set_title("Validation")
                    ax[1, 0].set_yscale("log")
                    ax[1, 1].set_yscale("log")

                    kwargs = {
                        "center": 0,
                        "cmap": "vlag",
                        "annot": True,
                        "vmin": -1,
                        "vmax": 1,
                        "annot_kws": {"size": 6},
                        "fmt": ".2f",
                    }
                    sns.heatmap(beta.cpu().detach().numpy(), ax=ax[0, 0], **kwargs)
                    sns.heatmap(model.beta.detach().cpu().numpy(), ax=ax[0, 1], **kwargs)
                    ax[0, 0].set_title("True beta")
                    ax[0, 1].set_title("Estimated beta")
                    plt.suptitle(
                        f"time: {end_time - start_time:.2f}s | L1: {scale_l1}, KL: {scale_kl}, Spectral: {scale_spectral}, Lyapunov: {scale_lyapunov} | Final T: {model.T.item():.2f}"
                    )
                    plt.tight_layout()
                    # Save fig
                    if SAVE_PLOT:
                        fig.savefig(file_name_plot)
                    plt.show()
                    # Close figure
                    plt.close(fig)

                except Exception as e:
                    extype, value, tb = sys.exc_info()
                    traceback.print_exc()
                    pdb.post_mortem(tb)


#
# TEST SET PREDICTIONS
#
# file_name_model, file_name_plot = get_names(
#     SAVE_PREFIX, n_genes, scale_l1, scale_kl, scale_spectral, scale_lyapunov, SEED, PLOT_PATH, MODEL_PATH
# )

# # # Evaluate log likelihood on test data
# new_model = BICYCLE.load_from_checkpoint(checkpoint_path=file_name_model)
# new_model.eval()

# dlogger = DictLogger()
# loggers = [dlogger]
# trainer = pl.Trainer(
#     accelerator="gpu" if str(device).startswith("cuda") else "cpu",
#     enable_model_summary=False,
#     enable_progress_bar=False,
#     devices=[GPU_DEVICE] if str(device).startswith("cuda") else 1,
#     logger=loggers,
# )
# trainer.test(new_model, test_loader)

# print(f"Test Loss: {trainer.logger.history['test_loss'].values[0]:.2f}")
