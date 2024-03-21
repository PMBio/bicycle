import warnings

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", category=FutureWarning)

import time
import os
from pathlib import Path
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
import click
import numpy as np
import yaml


@click.command()
@click.option("--nlogo", default=1, type=int)
@click.option("--seed", default=1, type=int)
@click.option("--lr", default=1e-3, type=float)
@click.option("--n-genes", default=5, type=int)
@click.option("--scale-l1", default=1, type=float)
@click.option("--scale-kl", default=1, type=float)
@click.option("--scale-spectral", default=1, type=float)
@click.option("--scale-lyapunov", default=1, type=float)
@click.option("--swa", default=0, type=int)
@click.option("--n-samples-control", default=250, type=int)
@click.option("--n-samples-per-perturbation", default=250, type=int)
@click.option("--validation-size", default=0.0, type=float)
@click.option("--sem", default="linear-ou", type=str)
@click.option("--use-latents", default=True, type=bool)
@click.option("--intervention-scale", default=1.0, type=float)
@click.option("--rank-w-cov-factor", default=10, type=int)
def run_bicycle_training(
    nlogo,
    seed,
    lr,
    n_genes,
    scale_l1,
    scale_kl,
    scale_spectral,
    scale_lyapunov,
    swa,
    n_samples_control,
    n_samples_per_perturbation,
    validation_size,
    sem,
    use_latents,
    intervention_scale,
    rank_w_cov_factor,
):
    SEED = seed
    SERVER = True
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

    #
    # Settings
    #
    device = torch.device("cpu")

    graph_type = "erdos-renyi"
    edge_assignment = "random-uniform"
    make_contractive = True
    graph_kwargs = {
        "abs_weight_low": 0.25,
        "abs_weight_high": 0.95,
        "p_success": 0.5,
        "expected_density": 2,
        "noise_scale": 0.5,
    }
    graph_kwargs_str = graph_kwargs["noise_scale"]
    graph_kwargs["intervention_scale"] = intervention_scale

    # LEARNING
    batch_size = 1024
    USE_INITS = False
    use_encoder = False
    n_epochs = 25_000
    optimizer = "adam"
    # DATA
    # nlogo REPRESENTS THE NUMBER OF GROUPS THAT SHOULD BE LEFT OUT DURING TRAINING
    LOGO = sorted(list(np.random.choice(n_genes, nlogo, replace=False)))
    train_gene_ko = [str(x) for x in set(range(0, n_genes)) - set(LOGO)]  # We start counting at 0
    # FIXME: There might be duplicates...
    ho_perturbations = sorted(
        list(set([tuple(sorted(np.random.choice(n_genes, 2, replace=False))) for _ in range(0, 20)]))
    )
    test_gene_ko = [f"{x[0]},{x[1]}" for x in ho_perturbations]

    # DGP
    # rank_w_cov_factor = n_genes - 1  # Same as dictys: #min(TFs, N_GENES-1)
    n_contexts = n_genes + 1  # Number of contexts
    perfect_interventions = True
    make_counts = False  # True | also set x_distribution
    # LEARNING
    early_stopping = True
    early_stopping_patience = 500
    early_stopping_min_delta = 0.01
    x_distribution = "Normal"  # "Poisson"
    x_distribution_kwargs = {"std": 0.1}
    # MODEL
    lyapunov_penalty = False
    GPU_DEVICE = 1
    plot_epoch_callback = 1000
    # RESULTS
    SAVE_PLOT = True
    CHECKPOINTING = True
    VERBOSE_CHECKPOINTING = False
    OVERWRITE = False
    # REST
    n_samples_total = (
        n_samples_control + (len(train_gene_ko) + len(test_gene_ko)) * n_samples_per_perturbation
    )
    check_val_every_n_epoch = 1
    log_every_n_steps = 1

    name_prefix = f"v1_{graph_type}_{graph_kwargs_str}_{early_stopping}_{early_stopping_patience}_{early_stopping_min_delta}_{x_distribution}"

    #
    # Create Mask
    #
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
        graph_type=graph_type,
        edge_assignment=edge_assignment,
        sem=sem,
        make_contractive=make_contractive,
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

    if USE_INITS:
        init_tensors = compute_inits(train_loader.dataset, rank_w_cov_factor, n_contexts)

    print("Training data:")
    print(f"- Number of training samples: {len(train_loader.dataset)}")
    if validation_size > 0:
        print(f"- Number of validation samples: {len(validation_loader.dataset)}")
    if LOGO:
        print(f"- Number of test samples: {len(test_loader.dataset)}")

    if SERVER:
        device = torch.device("cuda")
        devices = "auto"
    else:
        device = torch.device(f"cuda:{GPU_DEVICE}")
        devices = [GPU_DEVICE] if str(device).startswith("cuda") else 1
    gt_interv = gt_interv.to(device)
    n_genes = samples.shape[1]

    file_dir = (
        name_prefix
        + f"_{nlogo}_{seed}_{lr}_{n_genes}_{scale_l1}_{scale_kl}_{scale_spectral}_{scale_lyapunov}_{swa}_{n_samples_control}_{n_samples_per_perturbation}_{validation_size}_{sem}_{use_latents}_{intervention_scale}_{rank_w_cov_factor}"
    )

    # If final plot or final model exists: do not overwrite by default
    print("Checking Model and Plot files...")
    report_file_name = os.path.join(MODEL_PATH, file_dir, "report.yaml")
    final_file_name = os.path.join(MODEL_PATH, file_dir, "last.ckpt")
    final_plot_name = os.path.join(PLOT_PATH, file_dir, "last.png")
    if Path(report_file_name).exists() & ~OVERWRITE:
        print("- Files already exists, skipping...")
        pass
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
            x_distribution_kwargs=x_distribution_kwargs,
            mask=mask,
            use_encoder=use_encoder,
            gt_beta=beta,
            train_gene_ko=train_gene_ko,
            test_gene_ko=test_gene_ko,
            use_latents=use_latents,
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
                    start_after=0,
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
            gradient_clip_val=1,
            default_root_dir=str(MODEL_PATH),
            gradient_clip_algorithm="value",
            deterministic=True,
        )

        try:
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

        except Exception as e:
            # Write Exception to file
            report_path = os.path.join(MODEL_PATH, file_dir, "report.yaml")
            print(f"Raised Exception, writing to file: {report_path}")
            # Write yaml
            with open(report_path, "w") as outfile:
                yaml.dump({"exception": str(e)}, outfile, default_flow_style=False)


if __name__ == "__main__":
    run_bicycle_training()
