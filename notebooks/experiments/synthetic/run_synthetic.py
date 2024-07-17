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
from bicycle.utils.training import EarlyStopping_mod, suggest_hparams
from pytorch_lightning.callbacks import RichProgressBar, StochasticWeightAveraging
from bicycle.callbacks import ModelCheckpoint, GenerateCallback, MyLoggerCallback, CustomModelCheckpoint
import numpy as np
import yaml
#from pytorch_lightning.tuner.tuning import Tuner
import click
import optuna
from pprint import pformat, pprint
from typing import Optional
import copy

SEED = 1
pl.seed_everything(SEED)
torch.set_float32_matmul_precision("high")
DEVICE = torch.device("cpu")
if environ["USER"] == "m015k":
    user_dir = "/home/m015k/code/bicycle/notebooks/data"
else:
    user_dir = "."
#MODEL_PATH = Path(os.path.join(user_dir, "models"))
#PLOT_PATH = Path(os.path.join(user_dir, "plots"))
DATA_PATH = Path(os.path.join(user_dir, "data"))
#MODEL_PATH.mkdir(parents=True, exist_ok=True)
#PLOT_PATH.mkdir(parents=True, exist_ok=True)
DATA_PATH.mkdir(parents=True, exist_ok=True)
LOG_DIR = Path(user_dir)

@click.group()
def cli():
    pass


def run_pretrain(
        config: dict,
        model,
        train_loader,
        validation_loader,
        trial_id,
        beta,
        loggers,
):
        
    #Create directories for plot and model saving
    pretrain_model_dir = f"trial_{trial_id}/pretrain/model"
    Path(os.path.join(LOG_DIR, pretrain_model_dir)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(LOG_DIR, f"trial_{trial_id}", "plot")).mkdir(parents=True, exist_ok=True)
    pretrain_plot_dir = f"trial_{trial_id}/pretrain/plot"
    Path(os.path.join(LOG_DIR, pretrain_plot_dir)).mkdir(parents=True, exist_ok=True)
    final_plot_name = os.path.join(LOG_DIR, pretrain_plot_dir,"last_pretrain.png")
    pretrain_callbacks = [
        RichProgressBar(refresh_rate=1),
        GenerateCallback(
            final_plot_name, plot_epoch_callback=config["training"]["plot_epoch_callback"], true_beta=beta.cpu().numpy()
        ),                    
    ]
    
    if config["model"]["config"]["swa"] > 0:
        pretrain_callbacks.append(StochasticWeightAveraging(0.01, swa_epoch_start=config["model"]["config"]["swa"]))

    pretrain_callbacks.append(MyLoggerCallback(dirpath=os.path.join(LOG_DIR, pretrain_model_dir)))

    if "early_stopping" in config:
        pretrain_callbacks.append(
            EarlyStopping_mod(
                monitor="avg_valid_loss",
                **config["training"]["early_stopping"])
    )
    
    pretrainer = pl.Trainer(
        min_epochs=config["training"]["min_epochs_pretrain_latents"],
        max_epochs=config["training"]["n_epochs_pretrain_latents"],
        accelerator="gpu",  # if str(device).startswith("cuda") else "cpu",
        logger=loggers,
        log_every_n_steps=config["training"]["log_every_n_steps"],
        enable_model_summary=True,
        enable_progress_bar=True,
        enable_checkpointing=config["training"]["CHECKPOINTING"],
        check_val_every_n_epoch=config["training"]["check_val_every_n_epoch"],
        devices=[config["training"]["GPU_DEVICE"]],  # if str(device).startswith("cuda") else 1,
        num_sanity_val_steps=0,
        callbacks=pretrain_callbacks,
        gradient_clip_val=config["training"]["gradient_clip_val"],
        default_root_dir=str(os.path.join(LOG_DIR, pretrain_model_dir)),
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

    return model


def run_training(
       config: dict, 
       train_loader,
       validation_loader,
       n_samples_total,
       device, gt_interv, n_genes,
       train_gene_ko, test_gene_ko, beta,
       covariates,
       trial: Optional[optuna.trial.Trial] = None,
       trial_id: Optional[int] = None, 
):

    config_train = copy.deepcopy(config)
    if trial is not None:
        # Parameters set in config can be used to indicate hyperparameter optimization.
        # Set as the following:
        # lr: 
        #     hparam:
        #         type: 'float'  
        #             args:
        #                 - 1.0e-5
        #                 - 1.0e-1
        #             kwargs:
        #                 log: True
        # This translates to Optuna's suggest_float
        # lr = optuna.suggest_float(name="lr", low=1.0e-3, high=1.0e-1, log=True)
        # and afterward replace the respective area in config to the suggestion.
        config_train["model"]["config"] = suggest_hparams(config["model"]["config"], trial)
        print("Model hyperparameters this trial:")
        pprint(config_train["model"]["config"])
        Path(LOG_DIR / f"trial_{trial_id}").mkdir(parents=True, exist_ok=True)
        config_out = Path(LOG_DIR) / f"trial_{trial_id}" / "hpopt_config.yaml"
        with open(config_out, "w") as f:
            yaml.dump(config_train, f)

    if config_train["model"]["config"]["USE_INITS"]:
        init_tensors = compute_inits(train_loader.dataset, config_train["rank_w_cov_factor"], config_train["n_contexts"] + 1 )

    # Create Mask
    mask = get_diagonal_mask(config_train["n_genes"], DEVICE)
    if config_train["n_factors"] > 0:
        mask = None

    model = BICYCLE(
        config_train["model"]["config"]["lr"],
        gt_interv,
        n_genes,
        n_samples=n_samples_total,
        lyapunov_penalty=config_train["model"]["config"]["lyapunov_penalty"],
        perfect_interventions=config_train["perfect_interventions"],
        rank_w_cov_factor=config_train["rank_w_cov_factor"],
        init_tensors=init_tensors if config_train["model"]["config"]["USE_INITS"] else None,
        optimizer=config_train["model"]["config"]["optimizer"],
        optimizer_kwargs = config_train["model"]["config"]["optimizer_kwargs"],
        device=device,
        scale_l1=config_train["model"]["config"]["scale_l1"],
        scale_lyapunov=config_train["model"]["config"]["scale_lyapunov"],
        scale_spectral=config_train["model"]["config"]["scale_spectral"],
        scale_kl=config_train["model"]["config"]["scale_kl"],
        early_stopping=True if "early_stopping" in config_train["training"] else False,
        x_distribution=config_train["model"]["config"]["x_distribution"],
        x_distribution_kwargs=config_train["model"]["config"]["x_distribution_kwargs"],
        mask=mask,
        use_encoder=config_train["model"]["config"]["use_encoder"],
        gt_beta=beta,
        train_gene_ko=train_gene_ko,
        test_gene_ko=test_gene_ko,
        use_latents=config_train["model"]["config"]["use_latents"],
        covariates=covariates,
        n_factors = config_train["n_factors"],
        intervention_type = config_train["intervention_type_inference"],
        T = config_train["model"]["config"]["model_T"],
        learn_T = config_train["model"]["config"]["learn_T"]
    )
    model.to(device)

    dlogger = DictLogger()
    loggers = [dlogger]

    Path(os.path.join(LOG_DIR, f"trial_{trial_id}", "plot")).mkdir(parents=True, exist_ok=True)
    callbacks = [
        RichProgressBar(refresh_rate=1),
        GenerateCallback(
            Path(os.path.join(LOG_DIR, f"trial_{trial_id}", "plot")), plot_epoch_callback=config_train["training"]["plot_epoch_callback"], true_beta=beta.cpu().numpy()
        ),
    ]
    
    if "early_stopping" in config_train["training"]:
        callbacks.append(
            EarlyStopping_mod(
                monitor="avg_valid_loss",
                **config_train["training"]["early_stopping"])
        )
    
    if config_train["model"]["config"]["swa"] > 0:
        callbacks.append(StochasticWeightAveraging(0.01, swa_epoch_start=config_train["model"]["config"]["swa"]))
    if config_train["training"]["CHECKPOINTING"]:
        Path(os.path.join(LOG_DIR, f"trial_{trial_id}", "model")).mkdir(parents=True, exist_ok=True)

        callbacks.append(MyLoggerCallback(dirpath=os.path.join(LOG_DIR, f"trial_{trial_id}", "model")))

        callbacks.append(
            CustomModelCheckpoint(
                dirpath=os.path.join(LOG_DIR, f"trial_{trial_id}", "model"),
                filename="{epoch}",
                save_last=True,
                save_top_k=1,
                verbose=config_train["training"]["VERBOSE_CHECKPOINTING"],
                monitor="valid_loss",
                mode="min",
                save_weights_only=True,
                start_after=0,
                save_on_train_epoch_end=False,
                every_n_epochs=10,
            )
        )

    trainer = pl.Trainer(
        min_epochs=config_train["training"]["min_epochs_train"],
        max_epochs=config_train["training"]["n_epochs"],
        accelerator="gpu",  # if str(device).startswith("cuda") else "cpu",
        logger=loggers,
        log_every_n_steps=config_train["training"]["log_every_n_steps"],
        enable_model_summary=True,
        enable_progress_bar=True,
        enable_checkpointing=config_train["training"]["CHECKPOINTING"],
        check_val_every_n_epoch=config_train["training"]["check_val_every_n_epoch"],
        devices=[config_train["training"]["GPU_DEVICE"]],  # if str(device).startswith("cuda") else 1,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        gradient_clip_val=config_train["training"]["gradient_clip_val"],
        default_root_dir=str(os.path.join(LOG_DIR, f"trial_{trial_id}", "model")),
        gradient_clip_algorithm="value",
        deterministic=False, #"warn",
    )
    
    # Pre-train model latent parameters if specified in config
    if config_train["model"]["config"]["use_latents"] and config_train["training"]["n_epochs_pretrain_latents"] > 0:
        model = run_pretrain(config_train,
                             model,
                             train_loader,
                             validation_loader,
                             trial_id,
                             beta,
                             loggers,
        )

    start_time = time.time()
    # assert False
    trainer.fit(model, train_loader, validation_loader)
    end_time = time.time()
    print(f"Training took {end_time - start_time:.2f} seconds")

    final_plot_name = os.path.join(LOG_DIR, f"trial_{trial_id}", "plot","last.png")
    plot_training_results(
        trainer,
        model,
        model.beta.detach().cpu().numpy(),
        beta,
        config_train["model"]["config"]["scale_l1"],
        config_train["model"]["config"]["scale_kl"],
        config_train["model"]["config"]["scale_spectral"],
        config_train["model"]["config"]["scale_lyapunov"],
        final_plot_name,
        callback=False,
    )

    if config_train["training"]["CHECKPOINTING"]:
        trial.set_user_attr( #TODO: change callbacks to dictionary to reference explicitly "CustomModelCheckpoint"
            "checkpoint_path", callbacks[-2].best_model_path
        )

    return model.avg_valid_loss

@cli.command()
@click.argument("config-file", type=click.Path(exists=True, path_type=Path))
def run_synthetic_experiment(
    config_file: str,
):
    with open(config_file, "r") as fd:
        config = yaml.safe_load(fd)

    # Settings
    library_size_range = [config["library_size_range_factors"][0]*config["n_genes"], config["library_size_range_factors"][1]*config["n_genes"]]

    LOGO = []
    train_gene_ko = [str(x) for x in set(range(0, config["n_genes"])) - set(LOGO)]  # We start counting at 0
    # FIXME: There might be duplicates...
    ho_perturbations = sorted(
        list(set([tuple(sorted(np.random.choice(config["n_genes"], 2, replace=False))) for _ in range(0, 20)]))
    )
    test_gene_ko = [f"{x[0]},{x[1]}" for x in ho_perturbations]
    n_samples_total = config["n_samples_control"] + (len(train_gene_ko) + len(test_gene_ko)) * config["n_samples_per_perturbation"]

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
            config["training"]["validation_size"],
            config["training"]["batch_size"],
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
            config["training"]["validation_size"],
            config["training"]["batch_size"],
            SEED,
            train_gene_ko,
            test_gene_ko
        )
        covariates = None

    print("Training data:")
    print(f"- Number of training samples: {len(train_loader.dataset)}")
    if config["training"]["validation_size"] > 0:
        print(f"- Number of validation samples: {len(validation_loader.dataset)}")
    if LOGO:
        print(f"- Number of test samples: {len(test_loader.dataset)}")

    device = torch.device(f"cuda:{config['training']['GPU_DEVICE']}")
    gt_interv = gt_interv.to(device)
    n_genes = samples.shape[1]

    if covariates is not None and config["correct_covariates"]:
        covariates = covariates.to(device)
    
    file_dir = config['name_prefix'] #get_full_name(
        # ''.join(str(i) for i in config["name_prefix"]))
        # len(LOGO),
    #     SEED,
    #     config["lr"],
    #     n_genes,
    #     config["scale_l1"],
    #     config["scale_kl"],
    #     config["scale_spectral"],
    #     config["scale_lyapunov"],
    #     config["gradient_clip_val"],
    #     config["swa"],
    # )

    # # If final plot or final model exists: do not overwrite by default
    # print("Checking Model and Plot files...")
    # final_file_name = os.path.join(MODEL_PATH, file_dir, "last.ckpt")
    # final_plot_name = os.path.join(PLOT_PATH, file_dir, "last.png")

    # Save simulated data for inspection and debugging
    final_data_path = os.path.join(DATA_PATH, file_dir)

    if os.path.isdir(final_data_path):
        print(final_data_path, "exists")
    else:
        print("Creating", final_data_path)
        os.mkdir(final_data_path)

    np.save(os.path.join(final_data_path,'check_sim_samples.npy'), check_samples)
    np.save(os.path.join(final_data_path,'check_sim_regimes.npy'), check_sim_regime)
    np.save(os.path.join(final_data_path,'check_sim_beta.npy'), check_beta)
    np.save(os.path.join(final_data_path,'check_sim_gt_interv.npy'), check_gt_interv)

    # if (Path(final_file_name).exists() & config["SAVE_PLOT"] & ~config["OVERWRITE"]) | (
    #     Path(final_plot_name).exists() & config["CHECKPOINTING"] & ~config["OVERWRITE"]
    # ):
    #     print("- Files already exists, skipping...")
    #     pass
    # else:
    #     print("- Not all files exist, fitting model...")
    #     print("  - Deleting dirs")
    #     # Delete directories of files
    #     if Path(final_file_name).exists():
    #         print(f"  - Deleting {final_file_name}")
    #         # Delete all files in os.path.join(MODEL_PATH, file_name)
    #         for f in os.listdir(os.path.join(MODEL_PATH, file_dir)):
    #             os.remove(os.path.join(MODEL_PATH, file_dir, f))
    #     if Path(final_plot_name).exists():
    #         print(f"  - Deleting {final_plot_name}")
    #         for f in os.listdir(os.path.join(PLOT_PATH, file_dir)):
    #             os.remove(os.path.join(PLOT_PATH, file_dir, f))

    #     print("  - Creating dirs")
    #     # Create directories
    #     Path(os.path.join(MODEL_PATH, file_dir)).mkdir(parents=True, exist_ok=True)
    #     Path(os.path.join(PLOT_PATH, file_dir)).mkdir(parents=True, exist_ok=True)

    hparam_optim = config.get("hyperparameter_optimization", None)
    if hparam_optim is None:
        run_training(config, train_loader, validation_loader, 
                     n_samples_total,
                     device, gt_interv, n_genes,
                     train_gene_ko, test_gene_ko, beta,
                     covariates
        )
    else:
        pruner_config = config["hyperparameter_optimization"].get("pruning", None)
        if pruner_config is not None:
            pruner: optuna.pruners.BasePruner = getattr(
                optuna.pruners, pruner_config["type"]
            )(**pruner_config["config"])
        else:
            pruner = optuna.pruners.NopPruner()

        objective_direction = config["hyperparameter_optimization"].get(
            "direction", "minimize"
        )

        sampler_config = config["hyperparameter_optimization"].get("sampler", None)
        if sampler_config is not None:
            sampler: optuna.samplers._base.BaseSampler = getattr(
                optuna.samplers, sampler_config["type"]
            )(**sampler_config["config"])
        else:
            sampler = None

        hpopt_file = Path(os.path.join(LOG_DIR, "hyperparameter_optimization.db"))
        study = optuna.create_study(
            study_name=Path(hpopt_file).stem,
            direction=objective_direction,
            sampler=sampler,
            pruner=pruner,
            storage=f"sqlite:///{hpopt_file}",
            load_if_exists=True,
        )
        study.optimize(
            lambda trial: run_training(
                config, train_loader, validation_loader, 
                n_samples_total,
                device, gt_interv, n_genes,
                train_gene_ko, test_gene_ko, beta,
                covariates,
                trial=trial,
                trial_id=trial.number,
            ),
            n_trials=config["hyperparameter_optimization"]["n_trials"],
            timeout=hparam_optim.get("timeout", None),
            #n_jobs=25,
        )

        print(f"Number of finished trials: {len(study.trials)}")
        trial = study.best_trial
        print(f'Best trial: {trial.number}')
        print(
            #f'  Mean {config["model"]["config"]["metrics"]["objective"]}: '
            f"Mean neg_log_likelihood + loss_l1 + z_kl + loss_spectral + loss_lyapunov : {trial.value}"
        )
        print(f"  Params:\n{pformat(trial.params)}")


if __name__ == "__main__":
    cli()