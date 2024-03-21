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
from bicycle.utils_data import get_diagonal_mask, compute_inits
from bicycle.utils_plotting import plot_training_results
from pytorch_lightning.callbacks import RichProgressBar, StochasticWeightAveraging
from bicycle.callbacks import CustomModelCheckpoint, GenerateCallback, MyLoggerCallback
import click
import numpy as np


@click.command()
@click.option("--seed", default=1, type=int)
@click.option("--lr", default=1e-3, type=float)
@click.option("--scale-l1", default=1, type=float)
@click.option("--scale-kl", default=1, type=float)
@click.option("--scale-spectral", default=1, type=float)
@click.option("--scale-lyapunov", default=1, type=float)
@click.option("--gradient-clip-val", default=0.1, type=float)
@click.option("--swa", default=0, type=int)
@click.option("--use-inits", default=True, type=bool)
def run_bicycle_training(
    seed, lr, scale_l1, scale_kl, scale_spectral, scale_lyapunov, gradient_clip_val, swa, use_inits
):
    SEED = seed
    pl.seed_everything(SEED)
    torch.set_float32_matmul_precision("high")

    #
    # Paths
    #
    MODEL_PATH = Path("/omics/groups/OE0540/internal/users/rohbeck/bicycle/models/")
    DATA_PATH = Path("/omics/groups/OE0540/internal/users/rohbeck/bicycle/data/")
    PLOT_PATH = Path("/omics/groups/OE0540/internal/users/rohbeck/bicycle/plots/")

    if not MODEL_PATH.exists():
        MODEL_PATH.mkdir(parents=True, exist_ok=True)
    if not PLOT_PATH.exists():
        PLOT_PATH.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        DATA_PATH.mkdir(parents=True, exist_ok=True)

    print("Checking CUDA availability:")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    # Convert to int if possible
    if scale_l1.is_integer():
        scale_l1 = int(scale_l1)
    if scale_kl.is_integer():
        scale_kl = int(scale_kl)
    if scale_spectral.is_integer():
        scale_spectral = int(scale_spectral)
    if scale_lyapunov.is_integer():
        scale_lyapunov = int(scale_lyapunov)
    if gradient_clip_val.is_integer():
        gradient_clip_val = int(gradient_clip_val)

    #
    # Settings
    #
    device = torch.device("cpu")

    n_genes = 61

    # LEARNING
    batch_size = 1024
    USE_INITS = use_inits
    use_encoder = False
    n_epochs = 50_000
    optimizer = "adam"
    # DGP
    rank_w_cov_factor = n_genes - 1  # Same as dictys: #min(TFs, N_GENES-1)
    n_contexts = n_genes
    perfect_interventions = True
    # LEARNING
    early_stopping = False
    early_stopping_patience = 500
    early_stopping_min_delta = 0.01
    x_distribution = "Normal"  # "Poisson"
    # MODEL
    lyapunov_penalty = True
    plot_epoch_callback = 1000
    # RESULTS
    SAVE_PLOT = True
    CHECKPOINTING = True
    VERBOSE_CHECKPOINTING = False
    OVERWRITE = False
    check_val_every_n_epoch = 1
    log_every_n_steps = 1  # We don't need more on the server

    name_prefix = f"perturbseq_{batch_size}_"

    #
    # Create Mask
    #
    mask = get_diagonal_mask(n_genes, device)

    #
    # Create synthetic data
    #
    train_loader = torch.load(DATA_PATH / "nodags_data/control/training_data/train_loader.pth")
    validation_loader = torch.load(DATA_PATH / "nodags_data/control/validation_data/validation_loader.pth")
    test_loader = torch.load(DATA_PATH / "nodags_data/control/validation_data/test_loader.pth")
    labels = np.load(DATA_PATH / "nodags_data/control/training_data/labels.npy", allow_pickle=True)

    gt_interv = np.zeros((61, 61))
    for i in range(61):
        gt_interv[i, i] = 1
    gt_interv = torch.tensor(gt_interv, dtype=torch.float32)

    if use_inits:
        init_tensors = compute_inits(train_loader.dataset, rank_w_cov_factor, n_contexts)

    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of validation samples: {len(validation_loader.dataset)}")
    print(f"Number of test samples: {len(test_loader.dataset)}")

    device = torch.device("cuda")
    devices = "auto"
    gt_interv = gt_interv.to(device)

    file_dir = (
        name_prefix
        + f"{seed}_{lr}_{scale_l1}_{scale_kl}_{scale_spectral}_{scale_lyapunov}_{gradient_clip_val}_{swa}_{use_inits}"
    )

    # If final plot or final model exists: do not overwrite by default
    final_file_name = os.path.join(MODEL_PATH, file_dir, "last.ckpt")
    final_plot_name = os.path.join(PLOT_PATH, file_dir, "last.png")
    if (Path(final_file_name).exists() & SAVE_PLOT & ~OVERWRITE) | (
        Path(final_plot_name).exists() & CHECKPOINTING & ~OVERWRITE
    ):
        print("Files already exists, skipping...")
        pass
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

        # Save pickle of train_loader into final_file_name
        torch.save(train_loader, os.path.join(MODEL_PATH, file_dir, "train_loader.pth"))
        torch.save(validation_loader, os.path.join(MODEL_PATH, file_dir, "validation_loader.pth"))
        torch.save(test_loader, os.path.join(MODEL_PATH, file_dir, "test_loader.pth"))

        model = BICYCLE(
            lr,
            gt_interv,
            n_genes,
            n_samples=len(train_loader.dataset),
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
            mask=mask,
            use_encoder=use_encoder,
        )
        model.to(device)

        dlogger = DictLogger()
        loggers = [dlogger]

        callbacks = [
            RichProgressBar(refresh_rate=1),
            GenerateCallback(
                final_plot_name, plot_epoch_callback=plot_epoch_callback, true_beta=None, labels=labels
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
                    start_after=1000,
                    save_on_train_epoch_end=False,
                    every_n_epochs=100,
                )
            )
            callbacks.append(MyLoggerCallback(dirpath=os.path.join(MODEL_PATH, file_dir)))

        trainer = pl.Trainer(
            max_epochs=n_epochs,
            accelerator="gpu",  # ONLY RUN THIS ON GPU
            logger=loggers,
            log_every_n_steps=log_every_n_steps,
            enable_model_summary=True,
            enable_progress_bar=True,
            enable_checkpointing=CHECKPOINTING,
            check_val_every_n_epoch=check_val_every_n_epoch,
            devices=devices,
            num_sanity_val_steps=0,
            callbacks=callbacks,
            gradient_clip_val=gradient_clip_val,
            default_root_dir=str(MODEL_PATH),
            gradient_clip_algorithm="value",
        )

        # try:
        start_time = time.time()
        trainer.fit(model, train_loader, validation_loader)
        end_time = time.time()
        print(f"Training took {end_time - start_time:.2f} seconds")

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


if __name__ == "__main__":
    run_bicycle_training()
