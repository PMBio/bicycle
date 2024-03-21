import warnings

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", category=FutureWarning)

from pathlib import Path
import pytorch_lightning as pl
import torch
import numpy as np
import pandas as pd
from bicycle.model import BICYCLE
from bicycle.nodags_files.nodags import resflow_train_test_wrapper
from bicycle.utils.metrics import compute_auprc
from bicycle.utils.data import process_data_for_nodags

SEED = 0
pl.seed_everything(SEED)
device = torch.device("cpu")
MODEL_PATH = Path("/data/m015k/data/bicycle/models")
RESULTS_PATH = Path("/home/m015k/code/bicycle/notebooks/data/results")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

NHIDDEN = 0
FUNTYPE = "lin-mlp"  ###"lin-mlp"    "gst-mlp"
LAMBDA_C = 0.001   # 0.01, 0.1

if (RESULTS_PATH / f"results_new_{NHIDDEN}_{FUNTYPE}_{LAMBDA_C}.parquet").exists():
    print("Loading existing results")
    df_models = pd.read_parquet(RESULTS_PATH / f"results_new_{NHIDDEN}_{FUNTYPE}_{LAMBDA_C}.parquet")
else:
    print("Creating new results df")
    df_models = pd.DataFrame(columns=["filename", "fun_type", "lambda_c", "lr", "n_hidden"])
n_genes = 10
for filename in MODEL_PATH.glob("v6_*/last.ckpt"):
    if "NormalNormal" in str(filename.parent):
        n_samples_control = int(str(filename.parent).split("_")[21])
        nlogo = int(str(filename.parent).split("_")[11])
        seed = int(str(filename.parent).split("_")[12])
        n_samples_per_perturbation = int(str(filename.parent).split("_")[22])
        sem = str(filename.parent).split("_")[24]
        intervention_scale = float(str(filename.parent).split("_")[-1])
        use_latents = "True"  # FIXME: STUPID BUG
        x_distribution = str(filename.parent).split("_")[9]
    else:
        n_samples_control = int(str(filename.parent).split("_")[19])
        nlogo = int(str(filename.parent).split("_")[9])
        seed = int(str(filename.parent).split("_")[10])
        n_samples_per_perturbation = int(str(filename.parent).split("_")[20])
        sem = str(filename.parent).split("_")[22]
        intervention_scale = float(str(filename.parent).split("_")[-1])
        use_latents = "True"  # FIXME: STUPID BUG
        x_distribution = "Normal"

    try:
        model = BICYCLE.load_from_checkpoint(checkpoint_path=filename, map_location="cuda", strict=True)
    except Exception as e:
        print(f"Could not load model {filename}: {e}")
        continue

    if "NormalNormal" in str(filename.parent):
        continue
    if str(filename.parent).split("_")[-2] == "True":
        continue

    gt_beta = model.gt_beta.detach().cpu()
    gt_beta_copy = gt_beta.clone()
    gt_interv = model.gt_interv.detach().cpu()
    gt_beta = (gt_beta != 0).float()

    # Load train_loader from folder
    train_loader = torch.load(str(filename).replace("last.ckpt", "train_loader.pth"))
    test_loader = torch.load(str(filename).replace("last.ckpt", "test_loader.pth"))

    # assert False

    # # Extract control samples from train_loader
    # dataset_train = []
    # for batch in train_loader:
    #     data = batch[0]
    #     data = data[batch[1] == 0]
    #     dataset_train.append(data)
    # dataset_train = [torch.cat(dataset_train, dim=0).detach().cpu().numpy()]
    # dataset_train_targets = [np.array([0])]

    dataset_train, dataset_train_targets = process_data_for_nodags(
        train_loader, gt_interv, model.train_gene_ko, n_samples_control
    )
    dataset_test, dataset_test_targets = process_data_for_nodags(
        test_loader, gt_interv, model.test_gene_ko, n_samples_control
    )

    # Remove 20% validation data (this should actually be the same as in the original training)
    # Uncomment if you want to use all data for training
    dataset_valid = []
    dataset_valid_targets = []
    for k in range(len(dataset_train)):
        train_idx = np.random.choice(len(dataset_train[k]), int(0.8 * len(dataset_train[k])), replace=False)
        valid_idx = np.setdiff1d(np.arange(len(dataset_train[k])), train_idx)

        dataset_valid.append(dataset_train[k][valid_idx])
        dataset_valid_targets.append(dataset_train_targets[k])

        dataset_train[k] = dataset_train[k][train_idx]
        
    break

    batch_size = 1024
    epochs = 500
    for fun_type in [FUNTYPE]:  #    |                  | "mul-mlp"
        for lambda_c in [LAMBDA_C]:  # , 1e-1, 1e-2, 1e-3, 1e-4  | , 1e-3, , 1e-1, 1, 10, 100
            for lr in [1e-2, 1e-1, 1e-3]:  #  1, 1e-1, 1e-2, 1e-3,
                for n_hidden in [NHIDDEN]:  #   , 1, 2, 3
                    print(f"fun_type: {fun_type}, lambda_c: {lambda_c}, lr: {lr}, n_hidden: {n_hidden}")

                    # Check if model already exists
                    if (
                        df_models[
                            (df_models["filename"] == str(filename))
                            & (df_models["fun_type"] == fun_type)
                            & (df_models["lambda_c"] == lambda_c)
                            & (df_models["lr"] == lr)
                            & (df_models["n_hidden"] == n_hidden)
                        ].shape[0]
                        > 0
                    ):
                        print("Parameter set for model already exists, skipping...")
                        continue

                    notears_wrapper = resflow_train_test_wrapper(
                        n_nodes=10,
                        lambda_c=lambda_c,
                        n_hidden=n_hidden,
                        lr=lr,
                        v=False,
                        l1_reg=True,
                        fun_type=fun_type,
                        act_fun="none",
                        epochs=epochs,
                        optim="adam",
                        inline=True,
                        lin_logdet=False,
                        batch_size=batch_size,
                        lip_const=0.99,
                        full_input=False,
                        upd_lip=True,
                        n_lip_iter=5,
                        dag_input=False,
                    )
                    notears_wrapper.train(dataset_train, dataset_train_targets, batch_size=batch_size)
                    nll_pred_valid = notears_wrapper.predictLikelihood(dataset_valid, dataset_valid_targets)

                    # notears_wrapper.train(dataset_train, dataset_train_targets, batch_size=batch_size)
                    est_beta = notears_wrapper.get_adjacency()
                    est_beta = torch.from_numpy(est_beta).to(device).detach().cpu()

                    area_test = compute_auprc(gt_beta, est_beta)
                    nll_pred_test = notears_wrapper.predictLikelihood(dataset_test, dataset_test_targets)

                    # Append results to df_models
                    # try:
                    df_models = pd.concat(
                        [
                            df_models,
                            pd.DataFrame(
                                {
                                    "filename": str(filename),
                                    "nlogo": nlogo,
                                    "seed": seed,
                                    "sem": sem,
                                    "n_samples_control": n_samples_control,
                                    "intervention_scale": intervention_scale,
                                    "fun_type": fun_type,
                                    "lr": lr,
                                    "lambda_c": lambda_c,
                                    "n_hidden": n_hidden,
                                    "fun_type": fun_type,
                                    "area_test": area_test,
                                    "nll_pred_valid": np.mean(nll_pred_valid),
                                    "nll_pred_test": np.mean(nll_pred_test),
                                    "ntrain": n_genes - nlogo,
                                    "epochs": epochs,
                                },
                                index=[0],
                            ),
                        ]
                    )

                    df_models.to_parquet(RESULTS_PATH / f"results_new_{NHIDDEN}_{FUNTYPE}_{LAMBDA_C}.parquet")
                    print(f"Saved df_models with nrows: {len(df_models)}")
