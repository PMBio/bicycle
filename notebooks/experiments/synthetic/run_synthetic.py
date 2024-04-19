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
from bicycle.utils.training import EarlyStopping_mod
from pytorch_lightning.callbacks import RichProgressBar, StochasticWeightAveraging
from bicycle.callbacks import ModelCheckpoint, GenerateCallback, MyLoggerCallback, CustomModelCheckpoint
import numpy as np
import yaml
from pytorch_lightning.tuner.tuning import Tuner
import click

SEED = 1
pl.seed_everything(SEED)
torch.set_float32_matmul_precision("high")
DEVICE = torch.device("cpu")
if environ["USER"] == "m015k":
    user_dir = "/home/m015k/code/bicycle/notebooks/data"
else:
    user_dir = "."
MODEL_PATH = Path(os.path.join(user_dir, "models"))
PLOT_PATH = Path(os.path.join(user_dir, "plots"))
MODEL_PATH.mkdir(parents=True, exist_ok=True)
PLOT_PATH.mkdir(parents=True, exist_ok=True)

@click.group()
def cli():
    pass

@cli.command()
@click.argument("config-file", type=click.Path(exists=True, path_type=Path))
def run_synthetic_experiment(
    config_file: str,
):
    with open(config_file, "r") as fd:
        config = yaml.safe_load(fd)

    # Settings
    n_contexts = config["n_contexts"] + 1  # Number of contexts
    library_size_range = [config["library_size_range_factors"][0]*config["n_genes"], config["library_size_range_factors"][1]*config["n_genes"]]

    LOGO = []
    train_gene_ko = [str(x) for x in set(range(0, config["n_genes"])) - set(LOGO)]  # We start counting at 0
    # FIXME: There might be duplicates...
    ho_perturbations = sorted(
        list(set([tuple(sorted(np.random.choice(config["n_genes"], 2, replace=False))) for _ in range(0, 20)]))
    )
    test_gene_ko = [f"{x[0]},{x[1]}" for x in ho_perturbations]
    n_samples_total = config["n_samples_control"] + (len(train_gene_ko) + len(test_gene_ko)) * config["n_samples_per_perturbation"]

    # Create Mask
    mask = get_diagonal_mask(config["n_genes"], DEVICE)

    if config["n_factors"] > 0:
        mask = None

    # Create synthetic data
    gt_dyn, intervened_variables, samples, gt_interv, sim_regime, beta = create_data(
        config["n_genes"],
        n_samples_control=config["n_samples_control"],
        n_samples_per_perturbation=config["n_samples_per_perturbation"],
        device=DEVICE,
        make_counts=config["make_counts"],
        T = config["synthetic_T"],
        train_gene_ko=train_gene_ko,
        test_gene_ko=test_gene_ko,
        graph_type=config["graph_type"],
        edge_assignment=config["edge_assignment"],
        sem=config["sem"],
        make_contractive=config["make_contractive"],
        intervention_type = config["intervention_type_simulation"],
        library_size_range = library_size_range,
        **config["graph_kwargs"],
    )

    check_samples, check_gt_interv, check_sim_regime, check_beta = (
            np.copy(samples), np.copy(gt_interv), np.copy(sim_regime), np.copy(beta)
    )

    print("eigvals B:",torch.max(torch.real(torch.linalg.eigvals(beta - torch.eye(config["n_genes"])))))

    if config["add_covariates"]:
        print('ADDING COVARIATES')
        # Create some artificial covariates and add them to the simulated data
        covariates = torch.randn((n_samples_total, config["n_covariates"])).to(DEVICE)
        covariate_weights = torch.zeros((config["n_genes"], config["n_covariates"])).to(DEVICE)
        
        '''covariate_weights[0,0] = covariate_strength
        covariate_weights[1,0] = covariate_strength
        
        covariate_weights[2,1] = -covariate_strength
        covariate_weights[3,1] = covariate_strength
        
        covariate_weights[4,2] = covariate_strength
        covariate_weights[5,2] = covariate_strength
        covariate_weights[6,2] = -covariate_strength'''
        
        covariate_weights[:,0] = config["covariate_strength"]
        
        print('covariates.shape',covariates.shape)
        print('covariate_weights',covariate_weights)
        
        print('samples before:',samples[:2])
        
        samples = samples + torch.mm(covariates, covariate_weights.transpose(0,1)) 
        
        print('samples after:',samples[:2])
        
        train_loader, validation_loader, test_loader, covariates = create_loaders(
            samples,
            sim_regime,
            config["validation_size"],
            config["batch_size"],
            SEED,
            train_gene_ko,
            test_gene_ko,
            covariates = covariates
        )
        
        if config["correct_covariates"] == False:
            print('NOT CORRECTING FOR COVARIATES!')
            covariates = None
    else:
        train_loader, validation_loader, test_loader = create_loaders(
            samples,
            sim_regime,
            config["validation_size"],
            config["batch_size"],
            SEED,
            train_gene_ko,
            test_gene_ko
        )
        covariates = None

    if config["USE_INITS"]:
        init_tensors = compute_inits(train_loader.dataset, config["rank_w_cov_factor"], n_contexts)

    print("Training data:")
    print(f"- Number of training samples: {len(train_loader.dataset)}")
    if config["validation_size"] > 0:
        print(f"- Number of validation samples: {len(validation_loader.dataset)}")
    if LOGO:
        print(f"- Number of test samples: {len(test_loader.dataset)}")

    device = torch.device(f"cuda:{config['GPU_DEVICE']}")
    gt_interv = gt_interv.to(device)
    n_genes = samples.shape[1]

    if covariates is not None and config["correct_covariates"]:
        covariates = covariates.to(device)
    
    for scale_l1, scale_kl, scale_spectral, scale_lyapunov in zip(
        config["scale_l1"],config["scale_kl"],config["scale_spectral"],config["scale_lyapunov"]
    ):
        file_dir = get_full_name(
            ''.join(str(i) for i in config["name_prefix"]),
            len(LOGO),
            SEED,
            config["lr"],
            n_genes,
            scale_l1,
            scale_kl,
            scale_spectral,
            scale_lyapunov,
            config["gradient_clip_val"],
            config["swa"],
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

        np.save(os.path.join(final_data_path,'check_sim_samples.npy'), check_samples)
        np.save(os.path.join(final_data_path,'check_sim_regimes.npy'), check_sim_regime)
        np.save(os.path.join(final_data_path,'check_sim_beta.npy'), check_beta)
        np.save(os.path.join(final_data_path,'check_sim_gt_interv.npy'), check_gt_interv)

        if (Path(final_file_name).exists() & config["SAVE_PLOT"] & ~config["OVERWRITE"]) | (
            Path(final_plot_name).exists() & config["CHECKPOINTING"] & ~config["OVERWRITE"]
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
            config["lr"],
            gt_interv,
            n_genes,
            n_samples=n_samples_total,
            lyapunov_penalty=config["lyapunov_penalty"],
            perfect_interventions=config["perfect_interventions"],
            rank_w_cov_factor=config["rank_w_cov_factor"],
            init_tensors=init_tensors if config["USE_INITS"] else None,
            optimizer=config["optimizer"],
            optimizer_kwargs = config["optimizer_kwargs"],
            device=device,
            scale_l1=scale_l1,
            scale_lyapunov=scale_lyapunov,
            scale_spectral=scale_spectral,
            scale_kl=scale_kl,
            early_stopping=True if "early_stopping" in config else False,
            # early_stopping_min_delta=config["early_stopping_min_delta"],
            # early_stopping_patience=config["early_stopping_patience"],
            # early_stopping_threshold_mode=config["early_stopping_threshold_mode"],
            #early_stopping_p_mode=True, # relative percent change
            x_distribution=config["x_distribution"],
            x_distribution_kwargs=config["x_distribution_kwargs"],
            mask=mask,
            use_encoder=config["use_encoder"],
            gt_beta=beta,
            train_gene_ko=train_gene_ko,
            test_gene_ko=test_gene_ko,
            use_latents=config["use_latents"],
            covariates=covariates,
            n_factors = config["n_factors"],
            intervention_type = config["intervention_type_inference"],
            T = config["model_T"],
            learn_T = config["learn_T"]
        )
        model.to(device)

        dlogger = DictLogger()
        loggers = [dlogger]

        callbacks = [
            RichProgressBar(refresh_rate=1),
            GenerateCallback(
                final_plot_name, plot_epoch_callback=config["plot_epoch_callback"], true_beta=beta.cpu().numpy()
            ),
        ]
        if config["swa"] > 0:
            callbacks.append(StochasticWeightAveraging(0.01, swa_epoch_start=config["swa"]))
        if config["CHECKPOINTING"]:
            Path(os.path.join(MODEL_PATH, file_dir)).mkdir(parents=True, exist_ok=True)
            callbacks.append(
                CustomModelCheckpoint(
                    dirpath=os.path.join(MODEL_PATH, file_dir),
                    filename="{epoch}",
                    save_last=True,
                    save_top_k=1,
                    verbose=config["VERBOSE_CHECKPOINTING"],
                    monitor="valid_loss",
                    mode="min",
                    save_weights_only=True,
                    start_after=0,
                    save_on_train_epoch_end=False,
                    every_n_epochs=1,
                )
            )
            callbacks.append(MyLoggerCallback(dirpath=os.path.join(MODEL_PATH, file_dir)))

        if "early_stopping" in config:
            callbacks.append(
                EarlyStopping_mod(
                    monitor="avg_valid_loss",
                    **config["early_stopping"])
            )

        trainer = pl.Trainer(
            min_epochs=config["min_epochs_train"],
            max_epochs=config["n_epochs"],
            accelerator="gpu",  # if str(device).startswith("cuda") else "cpu",
            logger=loggers,
            log_every_n_steps=config["log_every_n_steps"],
            enable_model_summary=True,
            enable_progress_bar=True,
            enable_checkpointing=config["CHECKPOINTING"],
            check_val_every_n_epoch=config["check_val_every_n_epoch"],
            devices=[config["GPU_DEVICE"]],  # if str(device).startswith("cuda") else 1,
            num_sanity_val_steps=0,
            callbacks=callbacks,
            gradient_clip_val=config["gradient_clip_val"],
            default_root_dir=str(MODEL_PATH),
            gradient_clip_algorithm="value",
            deterministic=False, #"warn",
        )

        '''print('Optimizing learning rates')

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
        model.hparams.lr = new_lr'''


        if config["use_latents"] and config["n_epochs_pretrain_latents"] > 0:
            
            pretrain_callbacks = [
                RichProgressBar(refresh_rate=1),
                GenerateCallback(
                    str(Path(final_plot_name).with_suffix("")) + '_pretrain', plot_epoch_callback=config["plot_epoch_callback"], true_beta=beta.cpu().numpy()
                ),                    
            ]
            
            if config["swa"] > 0:
                pretrain_callbacks.append(StochasticWeightAveraging(0.01, swa_epoch_start=config["swa"]))

            pretrain_callbacks.append(MyLoggerCallback(dirpath=os.path.join(MODEL_PATH, file_dir)))

            if "early_stopping" in config:
                pretrain_callbacks.append(
                    EarlyStopping_mod(
                        monitor="avg_valid_loss",
                        **config["early_stopping"])
            )
            
            pretrainer = pl.Trainer(
                min_epochs=config["min_epochs_pretrain_latents"],
                max_epochs=config["n_epochs_pretrain_latents"],
                accelerator="gpu",  # if str(device).startswith("cuda") else "cpu",
                logger=loggers,
                log_every_n_steps=config["log_every_n_steps"],
                enable_model_summary=True,
                enable_progress_bar=True,
                enable_checkpointing=config["CHECKPOINTING"],
                check_val_every_n_epoch=config["check_val_every_n_epoch"],
                devices=[config["GPU_DEVICE"]],  # if str(device).startswith("cuda") else 1,
                num_sanity_val_steps=0,
                callbacks=pretrain_callbacks,
                gradient_clip_val=config["gradient_clip_val"],
                default_root_dir=str(MODEL_PATH),
                gradient_clip_algorithm="value",
                deterministic=False, #"warn",
            )
            
            print('PRETRAINING LATENTS!')
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

if __name__ == "__main__":
    cli()