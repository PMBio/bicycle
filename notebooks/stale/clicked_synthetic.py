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
from bicycle.utils_data import (
    create_data,
    create_loaders,
    get_name,
    get_ring_mask,
    get_diagonal_mask,
    compute_inits,
)
from bicycle.utils_plotting import plot_training_results
from pytorch_lightning.callbacks import RichProgressBar, StochasticWeightAveraging
from bicycle.callbacks import CustomModelCheckpoint, GenerateCallback, MyLoggerCallback
import click
import numpy as np


@click.command()
@click.option("--nlogo", default=1, type=int)
@click.option("--seed", default=1, type=int)
@click.option("--lr", default=1e-3, type=float)
@click.option("--n-genes", default=5, type=int)
@click.option("--scale-l1", default=1, type=float)
@click.option("--scale-kl", default=1, type=float)
@click.option("--scale-spectral", default=1, type=float)
@click.option("--scale-lyapunov", default=1, type=float)
@click.option("--gradient-clip-val", default=0.1, type=float)
@click.option("--swa", default=0, type=int)
def run_bicycle_training(
    nlogo, seed, lr, n_genes, scale_l1, scale_kl, scale_spectral, scale_lyapunov, gradient_clip_val, swa
):
    SEED = seed
    SERVER = True
    pl.seed_everything(SEED)
    torch.set_float32_matmul_precision("high")

    #
    # Paths
    #
    if SERVER:
        MODEL_PATH = Path("/omics/groups/OE0540/internal/users/rohbeck/bicycle/models/")
        DATA_PATH = Path("/omics/groups/OE0540/internal/users/rohbeck/bicycle/data/")
        PLOT_PATH = Path("/omics/groups/OE0540/internal/users/rohbeck/bicycle/plots/")
    else:
        if environ["USER"] == "m015k":
            user_dir = "/home/m015k/code/bicycle/notebooks/data"
        else:
            user_dir = "."
        MODEL_PATH = Path(user_dir)
        DATA_PATH = Path(user_dir)
        PLOT_PATH = Path(os.path.join(user_dir, "plots"))
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

    graph = "cycle-random"
    graph_kwargs = {"abs_weight_low": 0.25, "abs_weight_high": 0.95, "p_success": 0.4}
    graph_kwargs_str = "_".join([f"{v}" for v in graph_kwargs.values()])
    n_additional_entries = 12

    # LEARNING
    batch_size = 1024
    USE_INITS = False
    use_encoder = False
    n_epochs = 20_000
    optimizer = "adam"
    # DATA
    # nlogo REPRESENTS THE NUMBER OF GROUPS THAT SHOULD BE LEFT OUT DURING TRAINING
    LOGO = sorted(list(np.random.choice(n_genes, nlogo, replace=False)))
    train_gene_ko = [str(x) for x in set(range(0, n_genes)) - set(LOGO)]  # We start counting at 0
    # FIXME: There might be duplicates...
    ho_perturbations = sorted(list(set([tuple(sorted(np.random.choice(n_genes, 2, replace=False))) for _ in range(0, 20)])))
    test_gene_ko = [f"{x[0]},{x[1]}" for x in ho_perturbations]

    # DGP
    rank_w_cov_factor = n_genes - 1  # Same as dictys: #min(TFs, N_GENES-1)
    add_mask = False
    n_contexts = n_genes + 1  # Number of contexts
    n_samples_control = 250
    n_samples_per_perturbation = 250
    perfect_interventions = True
    make_counts = False  # True | also set x_distribution
    # LEARNING
    early_stopping = False
    early_stopping_patience = 500
    early_stopping_min_delta = 0.01
    x_distribution = "Normal"  # "Poisson"
    # DATA
    validation_size = 0.2
    # MODEL
    lyapunov_penalty = False
    GPU_DEVICE = 1
    plot_epoch_callback = 1000
    # RESULTS
    name_prefix = f"v3_inc_{graph}_{graph_kwargs_str}_{use_encoder}_{optimizer}_{batch_size}_{lyapunov_penalty}_{x_distribution}"
    SAVE_PLOT = True
    CHECKPOINTING = True
    VERBOSE_CHECKPOINTING = False
    OVERWRITE = False
    # REST
    n_samples_total = (
        n_samples_control + (len(train_gene_ko) + len(test_gene_ko)) * n_samples_per_perturbation
    )
    check_val_every_n_epoch = 1
    log_every_n_steps = 1  # We don't need more on the server

    #
    # Create Mask
    #
    if add_mask:
        if graph == "cycle":
            mask = get_ring_mask(n_additional_entries, n_genes, device)
        else:
            raise NotImplementedError("Mask only implemented for DGP cycle")
    else:
        mask = get_diagonal_mask(n_genes, device)

    #
    # Create synthetic data
    #
    _, _, samples, gt_interv, sim_regime, beta = create_data(
        n_genes,
        n_samples_control=n_samples_control,
        n_samples_per_perturbation=n_samples_per_perturbation,
        device=device,
        make_counts=make_counts,
        train_gene_ko=train_gene_ko,
        test_gene_ko=test_gene_ko,
        graph=graph,
        **graph_kwargs,
    )

    train_loader, validation_loader, test_loader = create_loaders(
        samples,
        sim_regime,
        validation_size,
        batch_size,
        SEED,
        train_gene_ko,
        test_gene_ko,
    )

    # Check if eig value of identity matrix minus beta in all contexts are < 0
    B = torch.eye(n_genes) - (1.0 - torch.eye(n_genes)) * beta.T
    for k in range(0, 11):
        B_p = B.clone()
        if k < 10:
            B_p[:, k] = 0
        eig_values = torch.real(torch.linalg.eigvals(-B_p))
        if torch.any(eig_values > 0):
            raise ValueError("Eigenvalues of identity matrix minus beta are not all negative.")

    if USE_INITS:
        init_tensors = compute_inits(train_loader.dataset, rank_w_cov_factor, n_contexts)

    print(f"Number of training samples: {len(train_loader.dataset)}")
    if validation_size > 0:
        print(f"Number of validation samples: {len(validation_loader.dataset)}")
    if LOGO:
        print(f"Number of test samples: {len(test_loader.dataset)}")

    if SERVER:
        device = torch.device("cuda")
        devices = "auto"
    else:
        device = torch.device(f"cuda:{GPU_DEVICE}")
        devices = [GPU_DEVICE] if str(device).startswith("cuda") else 1
    print(device)
    gt_interv = gt_interv.to(device)
    n_genes = samples.shape[1]

    file_dir = get_name(
        name_prefix,
        nlogo,
        seed,
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
            n_samples=n_samples_total,
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
            gt_beta=beta,
            train_gene_ko=train_gene_ko,
            test_gene_ko=test_gene_ko,
        )
        model.to(device)

        dlogger = DictLogger()
        loggers = [dlogger]

        callbacks = [
            RichProgressBar(refresh_rate=1),
            GenerateCallback(
                final_plot_name, plot_epoch_callback=plot_epoch_callback, true_beta=beta.cpu().numpy()
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
            beta,
            scale_l1,
            scale_kl,
            scale_spectral,
            scale_lyapunov,
            final_plot_name,
            callback=False,
        )


if __name__ == "__main__":
    run_bicycle_training()
