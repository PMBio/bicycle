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
from bicycle.nodags_files.llc import LLCClassWrapper
from bicycle.utils.metrics import compute_auprc, compute_shd
from bicycle.utils.data import process_data_for_llc

SEED = 0
pl.seed_everything(SEED)
device = torch.device("cpu")
MODEL_PATH = Path("/data/m015k/data/bicycle/models")
RESULTS_PATH = Path("/home/m015k/code/bicycle/notebooks/data/results")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

N_GENES = 10
if Path(RESULTS_PATH / "results_synthetic_llc.parquet").exists():
    df_models = pd.read_parquet(RESULTS_PATH / "results_synthetic_llc.parquet")
else:
    df_models = pd.DataFrame(
        columns=[
            "filename",
            "area_test",
            "nlogo",
            "seed",
            "nll_pred_test",
            "ntrain",
            "sem",
            "n_samples_control",
            "intervention_scale",
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
    if n_samples_control != 0:
        print("Found control data. LLC cannot use control data. Skipping...")
        continue
    if str(filename) in df_models["filename"].values:
        print(f"Skipping {filename}")
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

    dataset_train, dataset_train_targets = process_data_for_llc(train_loader, gt_interv, model.train_gene_ko)
    dataset_test, dataset_test_targets = process_data_for_llc(test_loader, gt_interv, model.test_gene_ko)

    # We do not remove the validation data here, because we want to compare to the original training

    # Remove 20% validation data (this should actually be the same as in the original training)
    # Uncomment if you want to use all data for training
    # for k in range(len(dataset_train)):
    #     random_idx = np.random.choice(len(dataset_train[k]), int(0.8 * len(dataset_train[k])), replace=False)
    #     dataset_train[k] = dataset_train[k][random_idx]

    llc_wrapper = LLCClassWrapper()
    est_beta = llc_wrapper.train(dataset_train, dataset_train_targets, return_weights=True)
    est_beta = torch.from_numpy(est_beta).to(device).detach().cpu()

    area_test = compute_auprc(gt_beta, est_beta)
    nll_pred_test = llc_wrapper.predictLikelihood(dataset_test, dataset_test_targets)

    # Append results to df_models
    df_models = pd.concat(
        [
            df_models,
            pd.DataFrame(
                {
                    "filename": str(filename),
                    "area_test": area_test,
                    "nlogo": nlogo,
                    "seed": seed,
                    "nll_pred_test": np.mean(nll_pred_test),
                    "ntrain": N_GENES - nlogo,
                    "sem": sem,
                    "n_samples_control": n_samples_control,
                    "intervention_scale": intervention_scale,
                },
                index=[0],
            ),
        ]
    )

df_models.reset_index(drop=True).to_parquet(RESULTS_PATH / "results_synthetic_llc.parquet")

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
sns.boxplot(
    data=df_models[df_models["sem"] == "linear"], x="ntrain", y="area_test", hue="intervention_scale", ax=ax
)
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
sns.boxplot(
    data=df_models[df_models["sem"] == "linear-ou"],
    x="ntrain",
    y="area_test",
    hue="intervention_scale",
    ax=ax,
)
