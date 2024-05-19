import warnings

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import time
from os import environ
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from bicycle.callbacks import (
    CustomModelCheckpoint,
    GenerateCallback,
    MyLoggerCallback,
)
from bicycle.dictlogger import DictLogger
from bicycle.model import BICYCLE
from bicycle.utils.data import (
    compute_inits,
    create_data,
    create_loaders,
    get_diagonal_mask,
)
from bicycle.utils.general import get_full_name
from bicycle.utils.plotting import plot_training_results
from pytorch_lightning.callbacks import RichProgressBar, StochasticWeightAveraging
from pytorch_lightning.tuner.tuning import Tuner

n_factors = 0

add_covariates = False
n_covariates = 0  # Number of covariates
covariate_strength = 5.0
correct_covariates = False

intervention_type_simulation = "Cas9"
intervention_type_inference = "Cas9"

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

#
# Settings
#

# DATA GENERATION
n_genes = 10  # Number of modelled genes
rank_w_cov_factor = n_genes  # Same as dictys: #min(TFs, N_GENES-1)
graph_type = "erdos-renyi"
edge_assignment = "random-uniform"
sem = "linear-ou"
graph_kwargs = {
    "abs_weight_low": 0.25,
    "abs_weight_high": 0.95,
    "p_success": 0.5,
    "expected_density": 2,
    "noise_scale": 0.5,
    "intervention_scale": 0.1,
}
n_additional_entries = 12
n_contexts = n_genes + 1  # Number of contexts
n_samples_control = 500
n_samples_per_perturbation = 250
perfect_interventions = True
make_contractive = True
make_counts = True
synthetic_T = 1.0
library_size_range = [10 * n_genes, 100 * n_genes]

# TRAINING
lr = 1e-3  # 3e-4
batch_size = 10_000
USE_INITS = False
use_encoder = False
n_epochs = 51000
early_stopping = False
early_stopping_patience = 500
early_stopping_min_delta = 0.01
# Maybe this helps to stop the loss from growing late during training (see current version
# of Plot_Diagnostics.ipynb)
optimizer = "adam"  # "rmsprop" #"adam"
optimizer_kwargs = {"betas": [0.5, 0.9]}  # Faster decay for estimates of gradient and gradient squared
gradient_clip_val = 1.0
GPU_DEVICE = 0
plot_epoch_callback = 500
validation_size = 0.2
lyapunov_penalty = True
swa = 250
n_epochs_pretrain_latents = 10000

LOGO = []
train_gene_ko = [str(x) for x in set(range(0, n_genes)) - set(LOGO)]  # We start counting at 0
# FIXME: There might be duplicates...
ho_perturbations = sorted(
    list(set([tuple(sorted(np.random.choice(n_genes, 2, replace=False))) for _ in range(0, 20)]))
)
test_gene_ko = [f"{x[0]},{x[1]}" for x in ho_perturbations]

# MODEL
x_distribution = "Multinomial"
x_distribution_kwargs = {}
model_T = 1.0
learn_T = False
use_latents = make_counts

# RESULTS
name_prefix = f"LATENT_SYNTHETIC_optim{optimizer}_b1_0.5_b2_0.9_pretrain_epochs{n_epochs_pretrain_latents}synthetic_T{synthetic_T}_GRAD-CLIP_SIM:{intervention_type_simulation}INF:{intervention_type_inference}-slow_lr_{graph_type}_{edge_assignment}_{use_encoder}_{batch_size}_{lyapunov_penalty}"
SAVE_PLOT = True
CHECKPOINTING = False
VERBOSE_CHECKPOINTING = False
OVERWRITE = True
# REST
n_samples_total = n_samples_control + (len(train_gene_ko) + len(test_gene_ko)) * n_samples_per_perturbation
check_val_every_n_epoch = 1
log_every_n_steps = 1


# Create Mask
mask = get_diagonal_mask(n_genes, device)

if n_factors > 0:
    mask = None

#
# Create synthetic data
#
gt_dyn, intervened_variables, samples, gt_interv, sim_regime, beta = create_data(
    n_genes,
    n_samples_control=n_samples_control,
    n_samples_per_perturbation=n_samples_per_perturbation,
    device=device,
    make_counts=make_counts,
    T=synthetic_T,
    train_gene_ko=train_gene_ko,
    test_gene_ko=test_gene_ko,
    graph_type=graph_type,
    edge_assignment=edge_assignment,
    sem=sem,
    make_contractive=make_contractive,
    intervention_type=intervention_type_simulation,
    library_size_range=library_size_range,
    **graph_kwargs,
)

check_samples, check_gt_interv, check_sim_regime, check_beta = (
    np.copy(samples),
    np.copy(gt_interv),
    np.copy(sim_regime),
    np.copy(beta),
)

print("eigvals B:", torch.max(torch.real(torch.linalg.eigvals(beta - torch.eye(n_genes)))))

if add_covariates:

    print("ADDING COVARIATES")
    # Create some artificial covariates and add them to the simulated data
    covariates = torch.randn((n_samples_total, n_covariates)).to(device)
    covariate_weights = torch.zeros((n_genes, n_covariates)).to(device)

    """covariate_weights[0,0] = covariate_strength
    covariate_weights[1,0] = covariate_strength
    
    covariate_weights[2,1] = -covariate_strength
    covariate_weights[3,1] = covariate_strength
    
    covariate_weights[4,2] = covariate_strength
    covariate_weights[5,2] = covariate_strength
    covariate_weights[6,2] = -covariate_strength"""

    covariate_weights[:, 0] = covariate_strength

    print("covariates.shape", covariates.shape)
    print("covariate_weights", covariate_weights)

    print("samples before:", samples[:2])

    samples = samples + torch.mm(covariates, covariate_weights.transpose(0, 1))

    print("samples after:", samples[:2])

    train_loader, validation_loader, test_loader, covariates = create_loaders(
        samples,
        sim_regime,
        validation_size,
        batch_size,
        SEED,
        train_gene_ko,
        test_gene_ko,
        covariates=covariates,
    )

    if correct_covariates == False:
        print("NOT CORRECTING FOR COVARIATES!")
        covariates = None

else:

    train_loader, validation_loader, test_loader = create_loaders(
        samples, sim_regime, validation_size, batch_size, SEED, train_gene_ko, test_gene_ko
    )

    covariates = None

if USE_INITS:
    init_tensors = compute_inits(train_loader.dataset, rank_w_cov_factor, n_contexts)

print("Training data:")
print(f"- Number of training samples: {len(train_loader.dataset)}")
if validation_size > 0:
    print(f"- Number of validation samples: {len(validation_loader.dataset)}")
if LOGO:
    print(f"- Number of test samples: {len(test_loader.dataset)}")

device = torch.device(f"cuda:{GPU_DEVICE}")
gt_interv = gt_interv.to(device)
n_genes = samples.shape[1]

if covariates is not None and correct_covariates:
    covariates = covariates.to(device)

for scale_kl in [1.0]:  # 1
    for scale_l1 in [0.1]:
        for scale_spectral in [0.0]:  # 1.0
            for scale_lyapunov in [1.0]:  # 0.1
                file_dir = get_full_name(
                    name_prefix,
                    len(LOGO),
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

                # Save simulated data for inspection and debugging
                final_data_path = os.path.join(PLOT_PATH, file_dir)

                if os.path.isdir(final_data_path):
                    print(final_data_path, "exists")
                else:
                    print("Creating", final_data_path)
                    os.mkdir(final_data_path)

                np.save(os.path.join(final_data_path, "check_sim_samples.npy"), check_samples)
                np.save(os.path.join(final_data_path, "check_sim_regimes.npy"), check_sim_regime)
                np.save(os.path.join(final_data_path, "check_sim_beta.npy"), check_beta)
                np.save(os.path.join(final_data_path, "check_sim_gt_interv.npy"), check_gt_interv)

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
                    n_samples=n_samples_total,
                    lyapunov_penalty=lyapunov_penalty,
                    perfect_interventions=perfect_interventions,
                    rank_w_cov_factor=rank_w_cov_factor,
                    init_tensors=init_tensors if USE_INITS else None,
                    optimizer=optimizer,
                    optimizer_kwargs=optimizer_kwargs,
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
                    covariates=covariates,
                    n_factors=n_factors,
                    intervention_type=intervention_type_inference,
                    T=model_T,
                    learn_T=learn_T,
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
                    deterministic=False,  # "warn",
                )

                """print('Optimizing learning rates')
                
                tuner = Tuner(trainer)

                # Run learning rate finder
                lr_finder = tuner.lr_find(model)

                # Results can be found in
                print(lr_finder.results)

                # Plot with
                fig = lr_finder.plot(suggest=True)
                fig.save('lr_finder.png')

                # Pick point based on plot, or get suggestion
                new_lr = lr_finder.suggestion()
                
                print('Using learning rate of:',new_lr)

                # update hparams of the model
                model.hparams.lr = new_lr"""

                if use_latents and n_epochs_pretrain_latents > 0:

                    pretrain_callbacks = [
                        RichProgressBar(refresh_rate=1),
                        GenerateCallback(
                            str(Path(final_plot_name).with_suffix("")) + "_pretrain",
                            plot_epoch_callback=plot_epoch_callback,
                            true_beta=beta.cpu().numpy(),
                        ),
                    ]

                    if swa > 0:
                        pretrain_callbacks.append(StochasticWeightAveraging(0.01, swa_epoch_start=swa))

                    pretrain_callbacks.append(MyLoggerCallback(dirpath=os.path.join(MODEL_PATH, file_dir)))

                    pretrainer = pl.Trainer(
                        max_epochs=n_epochs_pretrain_latents,
                        accelerator="gpu",  # if str(device).startswith("cuda") else "cpu",
                        logger=loggers,
                        log_every_n_steps=log_every_n_steps,
                        enable_model_summary=True,
                        enable_progress_bar=True,
                        enable_checkpointing=CHECKPOINTING,
                        check_val_every_n_epoch=check_val_every_n_epoch,
                        devices=[GPU_DEVICE],  # if str(device).startswith("cuda") else 1,
                        num_sanity_val_steps=0,
                        callbacks=pretrain_callbacks,
                        gradient_clip_val=gradient_clip_val,
                        default_root_dir=str(MODEL_PATH),
                        gradient_clip_algorithm="value",
                        deterministic=False,  # "warn",
                    )

                    print("PRETRAINING LATENTS!")
                    start_time = time.time()
                    model.train_only_likelihood = True
                    # assert False
                    pretrainer.fit(model, train_loader, validation_loader)
                    end_time = time.time()
                    model.train_only_likelihood = False

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
