import warnings

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", category=FutureWarning)

from pathlib import Path
import pytorch_lightning as pl
import torch
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from bicycle.model import BICYCLE
from bicycle.nodags_files.notears import NotearsClassWrapper
from bicycle.utils.metrics import compute_auprc
from bicycle.utils.data import process_data_for_llc

SEED = 0
pl.seed_everything(SEED)
device = torch.device("cpu")
MODEL_PATH = Path("/data/m015k/data/bicycle/models")
RESULTS_PATH = Path("/home/m015k/code/bicycle/notebooks/data/results")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

N_GENES = 10
noise_scale = 0.5
if Path(RESULTS_PATH / "results_synthetic_notears.parquet").exists():
    df_models = pd.read_parquet(RESULTS_PATH / "results_synthetic_notears.parquet")
else:
    df_models = pd.DataFrame(
        columns=[
            "filename",
            "l1",
            "noise_scale"
        ]
    )
for filename in MODEL_PATH.glob("v6_*/last.ckpt"):
    if "NormalNormal" in str(filename.parent):
        n_samples_control = int(str(filename.parent).split("_")[21])
        nlogo = int(str(filename.parent).split("_")[11])
        seed = int(str(filename.parent).split("_")[12])
        n_samples_per_perturbation = int(str(filename.parent).split("_")[22])
        sem = str(filename.parent).split("_")[24]
        intervention_scale = float(str(filename.parent).split("_")[-1])
        use_latents = bool(str(filename.parent).split("_")[-2])
        x_distribution = str(filename.parent).split("_")[9]
    else:
        n_samples_control = int(str(filename.parent).split("_")[19])
        nlogo = int(str(filename.parent).split("_")[9])
        seed = int(str(filename.parent).split("_")[10])
        n_samples_per_perturbation = int(str(filename.parent).split("_")[20])
        sem = str(filename.parent).split("_")[22]
        intervention_scale = float(str(filename.parent).split("_")[-1])
        use_latents = bool(str(filename.parent).split("_")[-2])
        x_distribution = "Normal"
    if n_samples_control == 0:
        print("Need control data...")
        continue
    try:
        model = BICYCLE.load_from_checkpoint(checkpoint_path=filename, map_location="cuda", strict=True)
    except Exception as e:
        print(f"Could not load model {filename}: {e}")
        continue

    gt_beta = model.gt_beta.detach().cpu()
    gt_beta_copy = gt_beta.clone()
    gt_interv = model.gt_interv.detach().cpu()
    gt_beta = (gt_beta != 0).float()

    # Load train_loader from folder
    train_loader = torch.load(str(filename).replace("last.ckpt", "train_loader.pth"))
    test_loader = torch.load(str(filename).replace("last.ckpt", "test_loader.pth"))

    # Extract control samples from train_loader
    dataset_train = []
    for batch in train_loader:
        data = batch[0][batch[1] == 0]  # Context 0 is intervention samples
        dataset_train.append(data)
    dataset_train = [torch.cat(dataset_train, dim=0).detach().cpu().numpy()]
    dataset_train_targets = [np.array([])]

    # dataset_train, dataset_train_targets = process_data_for_llc(train_loader, gt_interv, model.train_gene_ko)
    dataset_test, dataset_test_targets = process_data_for_llc(test_loader, gt_interv, model.test_gene_ko)

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

    results = pd.DataFrame()
    for loss_type in ["l2"]:
        for l1 in [1e-4, 1e-3, 1e-2, 1e-1, 1]:  # [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
            # Check if model is already in df_models
            if (
                df_models[
                    (df_models["filename"] == str(filename))
                    & (df_models["l1"] == l1)
                    & (df_models["noise_scale"] == noise_scale)
                ].shape[0]
                > 0
            ):
                print("Parameter set for model already exists, skipping...")
                continue

            notears_wrapper = NotearsClassWrapper(lambda1=l1, loss_type=loss_type, noise_scale=noise_scale)
            est_beta = notears_wrapper.train(dataset_train, dataset_train_targets, return_weights=True)
            nll_pred_valid = notears_wrapper.predictLikelihood(dataset_valid, dataset_valid_targets)
            est_beta = torch.from_numpy(est_beta).to(device).detach().cpu()

            area_test = compute_auprc(gt_beta, est_beta)
            nll_pred_test = notears_wrapper.predictLikelihood(dataset_test, dataset_test_targets)

            # Append results to df_models
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
                            "nll_pred_valid": np.mean(nll_pred_valid),
                            "nll_pred_test": np.mean(nll_pred_test),
                            "noise_scale": noise_scale,
                            "area_test": area_test,
                            "l1": l1,
                        },
                        index=[0],
                    ),
                ],
                ignore_index=True,
            )

            df_models.to_parquet(RESULTS_PATH / "results_synthetic_notears.parquet")
            print(f"Saved df_models with nrows: {len(df_models)}")


# Mean data across nlogo (this is actually just representing more seeds)
# df_models = df_models.reset_index(drop=True)
# df_models = df_models.drop(columns=["nlogo"])
# df_models = df_models.groupby(["sem", "intervention_scale", "seed", "n_samples_control"]).mean().reset_index()
