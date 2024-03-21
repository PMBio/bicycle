# %%
import warnings

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*The loaded checkpoint was .*")
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import pytorch_lightning as pl
import torch
from bicycle.model import BICYCLE
import numpy as np
from bicycle.utils.metrics import compute_auprc
from bicycle.utils.plotting import plot_style

plot_style(minimal=True)

torch.set_float32_matmul_precision("high")
device = torch.device("cpu")
MODEL_PATH = Path("/data/m015k/data/bicycle/models")
PLOT_PATH = Path("/home/m015k/code/bicycle/notebooks/data/plots")
RESULTS_PATH = Path("/home/m015k/code/bicycle/notebooks/data/results")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
PLOT_PATH.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# TODO: Split by type of intervention

n_genes = 10

# %%
# Loop over all models in folder
# Compte Negative Loglikelihood on validation data
df_models = pd.DataFrame()
for filename in MODEL_PATH.glob("v1_*/last.ckpt"):
    if "NormalNormal" in str(filename.parent):
        n_samples_control = int(str(filename.parent).split("_")[21])
        nlogo = int(str(filename.parent).split("_")[11])
        seed = int(str(filename.parent).split("_")[12])
        n_samples_per_perturbation = int(str(filename.parent).split("_")[22])
        sem = str(filename.parent).split("_")[24]
        intervention_scale = float(str(filename.parent).split("_")[-1])
        use_latents = str(filename.parent).split("_")[-2]
        x_distribution = str(filename.parent).split("_")[9]
    else:
        n_samples_control = int(str(filename.parent).split("_")[19])
        nlogo = int(str(filename.parent).split("_")[9])
        seed = int(str(filename.parent).split("_")[10])
        n_samples_per_perturbation = int(str(filename.parent).split("_")[20])
        sem = str(filename.parent).split("_")[22]
        intervention_scale = float(str(filename.parent).split("_")[-1])
        use_latents = str(filename.parent).split("_")[-2]
        x_distribution = "Normal"
    print(filename)
    try:
        model = BICYCLE.load_from_checkpoint(checkpoint_path=filename, map_location=device, strict=True)
    except Exception as e:
        print(f"Could not load model {filename}: {e}")
        continue
    model.eval()

    # Compute reconstruction quality of beta for models
    gt_beta = model.gt_beta
    gt_beta = (gt_beta != 0).float()

    est_beta = torch.zeros((n_genes, n_genes), device=device)
    est_beta[model.beta_idx[0], model.beta_idx[1]] = model.beta_val.detach().cpu()

    area = compute_auprc(gt_beta, est_beta)

    trainer = pl.Trainer(
        accelerator="gpu",
        enable_model_summary=False,
        enable_progress_bar=False,
        devices=[1],
    )

    # Load train, valid and test_loader
    train_loader = torch.load(filename.parent / "train_loader.pth")
    valid_loader = torch.load(filename.parent / "validation_loader.pth")
    test_loader = torch.load(filename.parent / "test_loader.pth")

    if not test_loader:
        print("No test set, skipping...")
        predictions = np.nan
    else:
        try:
            predictions_valid = sum(trainer.predict(model, valid_loader)) / len(valid_loader)
            predictions_valid = predictions_valid.item() / n_genes
            print(f"Loss Valid: {predictions_valid:.2f}")
            predictions = sum(trainer.predict(model, test_loader)) / len(test_loader)
            predictions = predictions.item() / n_genes
            print(f"Loss Test: {predictions:.2f}")
        except Exception as e:
            print(f"Could not predict test set: {e}")
            predictions_valid = np.nan
            predictions = np.nan

    df_temp = pd.DataFrame(
        {
            "lr": model.hparams.lr,
            "scale_l1": model.hparams.scale_l1,
            "scale_kl": model.hparams.scale_kl,
            "scale_spectral": model.hparams.scale_spectral,
            "scale_lyapunov": model.hparams.scale_lyapunov,
            "filename": filename,
            "auprc": area,
            "nlogo": nlogo,
            "seed": seed,
            "ho_nll_valid": predictions_valid,
            "ho_nll": predictions,
            "n_samples_control": n_samples_control,
            "sem": sem,
            "intervention_scale": intervention_scale,
            "use_latents": use_latents,
            "x_distribution": x_distribution,
        },
        index=[0],
    )
    df_models = pd.concat([df_models, df_temp], axis=0, ignore_index=True)

df_bicycle = df_models.copy()
# Pick best model for each train_gene_ko and test_gene_ko combination
df_bicycle["nlogo"] = df_bicycle["nlogo"].astype(int)
df_bicycle["ntrain"] = (10 - df_bicycle["nlogo"]).astype(int)
df_bicycle["n_samples_control"] = df_bicycle["n_samples_control"].astype(str)
df_bicycle["model"] = "Bicycle" + " (+" + df_bicycle["n_samples_control"] + " Ctrl.)"
# Choose only best model for each seed and test_gene_ko nlogo and intervention_scale and sem combination
# The best model has the lowest NLL on the validation set
df_bicycle = (
    df_bicycle.groupby(
        ["nlogo", "seed", "n_samples_control", "sem", "intervention_scale", "use_latents", "x_distribution"]
    )
    .apply(lambda x: x.sort_values("ho_nll_valid", ascending=True).head(1))
    .reset_index(drop=True)
)

# %%
# Load bicycle results
if False:
    df_bicycle["filename"] = df_bicycle["filename"].astype(str)
    df_bicycle.to_parquet(RESULTS_PATH / "results_synthetic_bicycle.parquet")
if False:
    df_bicycle = pd.read_parquet(RESULTS_PATH / "results_synthetic_bicycle.parquet")


# %%
# LLC
df_llc = pd.read_parquet(RESULTS_PATH / "results_synthetic_llc.parquet")
df_llc = df_llc.drop(columns=["filename"])
df_llc = df_llc.reset_index(drop=True)
df_llc["model"] = "LLC"
df_llc["ho_nll"] = df_llc["nll_pred_test"]
df_llc["auprc"] = df_llc["area_test"]

# %%
# NOTEARS
df_notears = pd.read_parquet(RESULTS_PATH / "results_synthetic_notears.parquet")
df_notears = df_notears.assign(model="Notears", ntrain=0)
df_notears["model"] = "NOTEARS" + " (+" + df_notears["n_samples_control"].astype(int).astype(str) + " Ctrl.)"
df_notears = (
    df_notears.groupby(
        ["filename", "model", "nlogo", "sem", "intervention_scale", "n_samples_control", "noise_scale"]
    )
    .apply(lambda x: x.sort_values("nll_pred_valid", ascending=True).head(1))
    .reset_index(drop=True)
)
df_notears = df_notears.drop(columns=["filename"])
df_notears["ho_nll"] = df_notears["nll_pred_test"]
df_notears["auprc"] = df_notears["area_test"]

# %%
# NODAGS
# TODO: NOTE THERE IS ALSO THE _ALL VERSION
# Load all files in RESULTS_PATH that start with "results_new_"
df_nodags = pd.DataFrame()
for file in RESULTS_PATH.glob("results_new_*.parquet"):
    df_nodags = pd.concat([df_nodags, pd.read_parquet(file)], axis=0, ignore_index=True)
    print(df_nodags.shape)
print(df_nodags.shape)
df_nodags = df_nodags.drop_duplicates().reset_index(drop=True)
print(df_nodags.shape)

# df_nodags0 = pd.read_parquet(RESULTS_PATH / "results_synthetic_nodags0.parquet")
# df_nodags1 = pd.read_parquet(RESULTS_PATH / "results_synthetic_nodags1.parquet")
# df_nodags2 = pd.read_parquet(RESULTS_PATH / "results_synthetic_nodags2.parquet")
# df_nodags3 = pd.read_parquet(RESULTS_PATH / "results_synthetic_nodags3.parquet")
# print(f"Loaded df_nodags0 with nrows: {len(df_nodags0)}")
# print(f"Loaded df_nodags1 with nrows: {len(df_nodags1)}")
# print(f"Loaded df_nodags2 with nrows: {len(df_nodags2)}")
# print(f"Loaded df_nodags3 with nrows: {len(df_nodags3)}")
# df_nodags = pd.concat(
#     [df_nodags0, df_nodags1, df_nodags2, df_nodags3], axis=0, ignore_index=True
# ).reset_index(drop=True)

# Only use best performing model on validation set
# Group by fun_type, lambda_c, lr, n_hidden and select best model on validaten nll
df_nodags = (
    df_nodags.groupby(["filename", "nlogo", "sem", "intervention_scale", "n_samples_control", "epochs"])
    .apply(lambda x: x.sort_values("nll_pred_valid", ascending=True).head(1))
    .reset_index(drop=True)
)
df_nodags["ho_nll"] = df_nodags["nll_pred_test"]
df_nodags["auprc"] = df_nodags["area_test"]
df_nodags["ntrain"] = 10 - df_nodags["nlogo"].astype(int)
df_nodags = df_nodags.drop(columns=["filename", "nll_pred_valid", "nll_pred_test", "area_test"])
df_nodags["model"] = "NODAGS" + " (+" + df_nodags["n_samples_control"].astype(int).astype(str) + " Ctrl.)"
print(df_nodags.shape)

# for k in df_nodags.columns:
#     # Print all value counts for each column
#     print(df_nodags[k].value_counts())

# %%
# df_nodags
df_plot = (
    pd.concat([df_bicycle, df_llc, df_notears, df_nodags], axis=0, ignore_index=True)
    .reset_index(drop=True)
    .copy()
)
df_plot = df_plot.drop(columns=["filename"])
df_plot["n_samples_control"] = df_plot["n_samples_control"].astype(int)


# %%
def make_plot(df_p, filename):
    fig, ax = plt.subplots(1, 2, figsize=(2 * 1.66 * 5.52, 5))
    sns.barplot(
        data=df_p,
        x="ntrain",
        y="auprc",
        hue="model",
        ax=ax[0],
        errwidth=1.5,
        hue_order=sorted([x for x in df_p["model"].unique()]),
        palette=[f"C{x}" for x in range(len(df_p["model"].unique()))],  #
    )
    sns.barplot(
        data=df_p,
        x="ntrain",
        y="ho_nll",
        hue="model",
        ax=ax[1],
        errwidth=1.5,
        hue_order=sorted([x for x in df_p["model"].unique()]),
        palette=[f"C{x}" for x in range(len(df_p["model"].unique()))],  #
    )
    # Make x labels real integers
    ax[0].set_xticklabels([int(float(x.get_text())) for x in ax[0].get_xticklabels()])
    ax[1].set_xticklabels([int(float(x.get_text())) for x in ax[1].get_xticklabels()])
    # Remove legend title
    ax[0].get_legend().set_title("")
    ax[1].get_legend().set_title("")
    ax[0].get_legend().remove()
    lgd = ax[1].legend(loc="lower center", bbox_to_anchor=(-0.1, -0.6), frameon=True, ncol=3)
    handles, labels = ax[1].get_legend_handles_labels()
    ax[0].set_xlabel("No. Interv. in Training Set")
    ax[0].set_ylabel("AUPRC")
    ax[1].set_xlabel("No. Interv. in Training Set")
    ax[1].set_ylabel("NLL")
    ax[0].set_ylim([0, None])
    ax[0].set_axisbelow(True)
    ax[1].set_axisbelow(True)
    ax[0].grid(axis="y")
    ax[1].grid(axis="y")
    ax[1].set_ylim([-3, 4])
    # Add vertical lines between bars
    for i in range(1, 8):
        ax[0].axvline(i - 0.5, color="black", linestyle="--", linewidth=0.95)
        ax[1].axvline(i - 0.5, color="black", linestyle="--", linewidth=0.95)

    # plt.tight_layout()
    # plt.savefig(f"{filename}.png", bbox_extra_artists=(lgd,), bbox_inches="tight")
    # plt.savefig(f"{filename}.svg", bbox_extra_artists=(lgd,), bbox_inches="tight")
    plt.show()


# %%
def make_plot1(df_p, filename):
    df_print = df_p.groupby(["ntrain", "model"], as_index=False).agg(
        {"auprc": ["mean", "std"], "ho_nll": ["mean", "std"]}
    )
    df_print = df_print.round(2)
    df_print["AUPRC"] = df_print["auprc"]["mean"].astype(str) + " ± " + df_print["auprc"]["std"].astype(str)
    df_print["NLL"] = df_print["ho_nll"]["mean"].astype(str) + " ± " + df_print["ho_nll"]["std"].astype(str)
    df_print = df_print.droplevel(1, axis=1)
    df_print = df_print.drop(columns=["auprc", "ho_nll"])
    df_print.columns = ["Int. Contexts", "Model", "AUPRC", "NLL"]

    print(df_print)

    fig = plt.figure(figsize=(2 * 5.52, 3.5))
    ax1 = fig.add_subplot(2, 2, (1, 3))
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 4, sharex=ax2)
    model_list = [
        "Bicycle (+0 Ctrl.)",
        "Bicycle (+500 Ctrl.)",
        "NODAGS (+0 Ctrl.)",
        "NODAGS (+500 Ctrl.)",
        "LLC",
        "NOTEARS (+500 Ctrl.)",
    ]

    palette = [
        "#a6cee3",
        "#1f78b4",
        "#b2df8a",
        "#33a02c",
        "#e31a1c",
        "#fdbf6f",
    ]
    sns.barplot(
        data=df_p,
        x="ntrain",
        y="auprc",
        hue="model",
        ax=ax1,
        errwidth=1.5,
        hue_order=model_list,
        palette=palette,
    )
    sns.barplot(
        data=df_p,
        x="ntrain",
        y="ho_nll",
        hue="model",
        ax=ax2,
        errwidth=1.5,
        hue_order=model_list,
        palette=palette,
    )
    sns.barplot(
        data=df_p,
        x="ntrain",
        y="ho_nll",
        hue="model",
        ax=ax3,
        errwidth=1.5,
        hue_order=model_list,
        palette=palette,
    )
    # Make x labels real integers
    ax1.set_xticklabels([int(float(x.get_text())) for x in ax1.get_xticklabels()])
    ax3.set_xticklabels([int(float(x.get_text())) for x in ax2.get_xticklabels()])
    # Remove legend title
    ax1.get_legend().remove()
    ax2.get_legend().remove()
    ax3.get_legend().set_title("")
    lgd = ax3.legend(loc="lower center", bbox_to_anchor=(-0.1, -1), frameon=True, ncol=3)
    handles, labels = ax3.get_legend_handles_labels()
    ax1.set_xlabel("No. Interv. in Training Set")
    ax1.set_ylabel("AUPRC")
    ax3.set_xlabel("No. Interv. in Training Set")
    ax3.set_ylabel("NLL")
    ax2.set_ylabel("")
    ax2.set_xlabel("")
    # ax1.set_ylim([0, None])
    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)
    ax3.set_axisbelow(True)
    ax1.grid(axis="y")
    ax2.grid(axis="y")
    ax3.grid(axis="y")
    ax3.set_ylim([-2.5, 7])
    ax2.set_ylim([60, 100])
    ax3.yaxis.set_label_coords(-0.075, 1.0)

    ax2.spines["bottom"].set_visible(False)
    ax3.spines["top"].set_visible(False)
    ax2.xaxis.tick_top()
    ax2.tick_params(labeltop=False)  # don't put tick labels at the top
    ax3.xaxis.tick_bottom()

    d = 0.01  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax2.transAxes, color="k", clip_on=False)
    ax2.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
    ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax3.transAxes)  # switch to the bottom axes
    ax3.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax3.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    # fig.subplots_adjust(hspace=2)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.1)

    # Add vertical lines between bars
    for i in range(1, 7):
        ax1.axvline(i - 0.5, color="black", linestyle="--", linewidth=0.95)
        ax2.axvline(i - 0.5, color="black", linestyle="--", linewidth=0.95)
        ax3.axvline(i - 0.5, color="black", linestyle="--", linewidth=0.95)

    # plt.tight_layout()
    # plt.savefig(f"{filename}.png", bbox_extra_artists=(lgd,), bbox_inches="tight")
    # plt.savefig(f"{filename}.svg", bbox_extra_artists=(lgd,), bbox_inches="tight")
    plt.show()

    return df_print


# %%
def make_plot2(df_p, filename):
    df_print = df_p.groupby(["ntrain", "model"], as_index=False).agg(
        {"auprc": ["mean", "std"], "ho_nll": ["mean", "std"]}
    )
    df_print = df_print.round(2)
    df_print["AUPRC"] = df_print["auprc"]["mean"].astype(str) + " ± " + df_print["auprc"]["std"].astype(str)
    df_print["NLL"] = df_print["ho_nll"]["mean"].astype(str) + " ± " + df_print["ho_nll"]["std"].astype(str)
    df_print = df_print.droplevel(1, axis=1)
    df_print = df_print.drop(columns=["auprc", "ho_nll"])
    df_print.columns = ["Int. Contexts", "Model", "AUPRC", "NLL"]

    print(df_print)

    fig, ax = plt.subplots(1, 2, figsize=(2 * 5.52, 3.5))

    palette = [
        "#a6cee3",
        "#1f78b4",
        "black",
    ]

    sns.barplot(
        data=df_p,
        x="ntrain",
        y="auprc",
        hue="model",
        ax=ax[0],
        errwidth=1.5,
        hue_order=sorted([x for x in df_p["model"].unique()]),
        palette=palette,
    )
    sns.barplot(
        data=df_p,
        x="ntrain",
        y="ho_nll",
        hue="model",
        ax=ax[1],
        errwidth=1.5,
        hue_order=sorted([x for x in df_p["model"].unique()]),
        palette=palette,
    )
    # Make x labels real integers
    ax[0].set_xticklabels([int(float(x.get_text())) for x in ax[0].get_xticklabels()])
    ax[1].set_xticklabels([int(float(x.get_text())) for x in ax[1].get_xticklabels()])
    # Remove legend title
    ax[0].get_legend().set_title("")
    ax[1].get_legend().set_title("")
    ax[0].get_legend().remove()
    lgd = ax[1].legend(loc="lower center", bbox_to_anchor=(-0.1, -0.35), frameon=True, ncol=3)
    handles, labels = ax[1].get_legend_handles_labels()
    ax[0].set_xlabel("No. Interv. in Training Set")
    ax[0].set_ylabel("AUPRC")
    ax[1].set_xlabel("No. Interv. in Training Set")
    ax[1].set_ylabel("NLL")
    ax[0].set_ylim([0, None])
    ax[0].set_axisbelow(True)
    ax[1].set_axisbelow(True)
    ax[0].grid(axis="y")
    ax[1].grid(axis="y")
    ax[1].set_ylim([-3, 4])
    # Add vertical lines between bars
    for i in range(1, 8):
        ax[0].axvline(i - 0.5, color="black", linestyle="--", linewidth=0.95)
        ax[1].axvline(i - 0.5, color="black", linestyle="--", linewidth=0.95)

    # plt.tight_layout()
    # plt.savefig(f"{filename}.png", bbox_extra_artists=(lgd,), bbox_inches="tight")
    # plt.savefig(f"{filename}.svg", bbox_extra_artists=(lgd,), bbox_inches="tight")
    plt.show()

    return df_print


# %%
def make_plot3(df_p, filename):
    df_print = df_p.groupby(["ntrain", "model"], as_index=False).agg(
        {"auprc": ["mean", "std"], "ho_nll": ["mean", "std"]}
    )
    df_print = df_print.round(2)
    df_print["AUPRC"] = df_print["auprc"]["mean"].astype(str) + " ± " + df_print["auprc"]["std"].astype(str)
    df_print["NLL"] = df_print["ho_nll"]["mean"].astype(str) + " ± " + df_print["ho_nll"]["std"].astype(str)
    df_print = df_print.droplevel(1, axis=1)
    df_print = df_print.drop(columns=["auprc", "ho_nll"])
    df_print.columns = ["Int. Contexts", "Model", "AUPRC", "NLL"]

    print(df_print)

    fig = plt.figure(figsize=(2 * 5.52, 3.5))
    ax1 = fig.add_subplot(2, 2, (1, 3))
    ax2 = fig.add_subplot(2, 2, (2, 4))
    model_list = [
        "Bicycle (+0 Ctrl.)",
        "NODAGS (+0 Ctrl.)",
        "LLC",
    ]

    palette = [
        "#1f78b4",
        "#33a02c",
        "#e31a1c",
    ]
    sns.barplot(
        data=df_p,
        x="ntrain",
        y="auprc",
        hue="model",
        ax=ax1,
        errwidth=1.5,
        hue_order=model_list,
        palette=palette,
    )
    sns.barplot(
        data=df_p,
        x="ntrain",
        y="ho_nll",
        hue="model",
        ax=ax2,
        errwidth=1.5,
        hue_order=model_list,
        palette=palette,
    )
    # Make x labels real integers
    ax1.set_xticklabels([int(float(x.get_text())) for x in ax1.get_xticklabels()])
    ax2.set_xticklabels([int(float(x.get_text())) for x in ax2.get_xticklabels()])
    # Remove legend title
    ax1.get_legend().remove()
    # ax2.get_legend().remove()
    ax2.get_legend().set_title("")
    lgd = ax2.legend(loc="lower center", bbox_to_anchor=(-0.1, -0.35), frameon=True, ncol=3)
    handles, labels = ax2.get_legend_handles_labels()
    ax1.set_xlabel("No. Interv. in Training Set")
    ax1.set_ylabel("AUPRC")
    ax2.set_xlabel("No. Interv. in Training Set")
    ax2.set_ylabel("NLL")
    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)
    ax1.grid(axis="y")
    ax2.grid(axis="y")
    ax1.set_ylim([0, 1.05])
    # ax2.set_ylim([60, 100])

    # Add vertical lines between bars
    for i in range(1, 6):
        ax1.axvline(i - 0.5, color="black", linestyle="--", linewidth=0.95)
        ax2.axvline(i - 0.5, color="black", linestyle="--", linewidth=0.95)

    # plt.tight_layout()
    # plt.savefig(f"{filename}.png", bbox_extra_artists=(lgd,), bbox_inches="tight")
    # plt.savefig(f"{filename}.svg", bbox_extra_artists=(lgd,), bbox_inches="tight")
    plt.show()

    return df_print


# %% Plot 1
if True:
    plt.style.use("default")
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"
    plt.rcParams["xtick.major.size"] = 2
    plt.rcParams["xtick.minor.size"] = 1
    plt.rcParams["ytick.major.size"] = 2
    plt.rcParams["ytick.minor.size"] = 1
    plt.rcParams["xtick.major.width"] = 0.5
    plt.rcParams["xtick.minor.width"] = 0.5
    plt.rcParams["ytick.major.width"] = 0.5
    plt.rcParams["ytick.minor.width"] = 0.5
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["legend.handlelength"] = 2
    plt.rcParams["figure.titlesize"] = 12

    x = df_plot.loc[
        (df_plot["sem"] == "linear-ou")
        & (df_plot["intervention_scale"] == 1)
        & ((df_plot["x_distribution"] == "Normal") | (df_plot["use_latents"].isna()))
        & ((df_plot["use_latents"] == "False") | (df_plot["use_latents"].isna()))
        & (
            df_plot["model"].isin(
                [
                    "Bicycle (+0 Ctrl.)",
                    "Bicycle (+500 Ctrl.)",
                    "NODAGS (+0 Ctrl.)",
                    "NODAGS (+500 Ctrl.)",
                    "LLC",
                    "NOTEARS (+500 Ctrl.)",
                ]
            )
        ),
        :,
    ].copy()
    df_print = make_plot1(x, filename="synthetic_linearou_fig1")

print(
    df_print.to_latex(
        index=False, formatters={"name": str.upper}, float_format="{:.1f}".format, bold_rows=True
    )
)


# %% Plot 2
if True:
    plt.style.use("default")
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"
    plt.rcParams["xtick.major.size"] = 2
    plt.rcParams["xtick.minor.size"] = 1
    plt.rcParams["ytick.major.size"] = 2
    plt.rcParams["ytick.minor.size"] = 1
    plt.rcParams["xtick.major.width"] = 0.5
    plt.rcParams["xtick.minor.width"] = 0.5
    plt.rcParams["ytick.major.width"] = 0.5
    plt.rcParams["ytick.minor.width"] = 0.5
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["legend.handlelength"] = 2
    plt.rcParams["figure.titlesize"] = 12

    x = df_plot.loc[
        (df_plot["sem"] == "linear-ou")
        & (df_plot["intervention_scale"] == 1)
        & ((df_plot["x_distribution"] == "Normal") | (df_plot["use_latents"].isna()))
        & ((df_plot["use_latents"] == "False") | (df_plot["use_latents"].isna()))
        & (
            df_plot["model"].isin(
                [
                    "Bicycle (+0 Ctrl.)",
                    "Bicycle (+500 Ctrl.)",
                    "Bicycle (+5000 Ctrl.)",
                ]
            )
        ),
        :,
    ].copy()
    df_print = make_plot2(x, filename="synthetic_linearou_fig2")


print(
    df_print.to_latex(
        index=False, formatters={"name": str.upper}, float_format="{:.1f}".format, bold_rows=True
    )
)

# %% Plot 3
if True:
    plt.style.use("default")
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"
    plt.rcParams["xtick.major.size"] = 2
    plt.rcParams["xtick.minor.size"] = 1
    plt.rcParams["ytick.major.size"] = 2
    plt.rcParams["ytick.minor.size"] = 1
    plt.rcParams["xtick.major.width"] = 0.5
    plt.rcParams["xtick.minor.width"] = 0.5
    plt.rcParams["ytick.major.width"] = 0.5
    plt.rcParams["ytick.minor.width"] = 0.5
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["legend.handlelength"] = 2
    plt.rcParams["figure.titlesize"] = 12

    x = df_plot.loc[
        (df_plot["sem"] == "linear")
        & (df_plot["intervention_scale"] == 1)
        & (df_plot["n_samples_control"] == 0)
        & ((df_plot["x_distribution"] == "Normal") | (df_plot["use_latents"].isna()))
        & ((df_plot["use_latents"] == "False") | (df_plot["use_latents"].isna())),
        :,
    ].copy()
    df_print = make_plot3(x, filename="synthetic_linear_1_fig3")

    print(
        df_print.to_latex(
            index=False, formatters={"name": str.upper}, float_format="{:.1f}".format, bold_rows=True
        )
    )

    df_notears.loc[
        (df_notears["sem"] == "linear")
        & (df_notears.intervention_scale == 1.0)
        & (df_notears.n_samples_control == 500),
        :,
    ]["auprc"].mean()

    x = df_plot.loc[
        (df_plot["sem"] == "linear")
        & (df_plot["intervention_scale"] == 0.5)
        & (df_plot["n_samples_control"] == 0)
        & ((df_plot["x_distribution"] == "Normal") | (df_plot["use_latents"].isna()))
        & ((df_plot["use_latents"] == "False") | (df_plot["use_latents"].isna())),
        :,
    ].copy()
    df_print = make_plot3(x, filename="synthetic_linear_05_fig4")

    print(
        df_print.to_latex(
            index=False,
            formatters={"name": str.upper},
            float_format="{:.1f}".format,
            bold_rows=True
        )
    )

    x = df_plot.loc[
        (df_plot["sem"] == "linear")
        & (df_plot["intervention_scale"] == 0.1)
        & (df_plot["n_samples_control"] == 0)
        & ((df_plot["x_distribution"] == "Normal") | (df_plot["use_latents"].isna()))
        & ((df_plot["use_latents"] == "False") | (df_plot["use_latents"].isna())),
        :,
    ].copy()
    df_print = make_plot3(x, filename="synthetic_linear_01_fig5")

    print(
        df_print.to_latex(
            index=False,
            formatters={"name": str.upper},
            float_format="{:.1f}".format,
            bold_rows=True
        )
    )


# %% Plot 4
df_plot_bicycle = df_plot.copy()
df_plot_bicycle["model"] = df_plot_bicycle["model"] + " " + df_plot_bicycle["x_distribution"].astype(str)
x = df_plot_bicycle.loc[
    (df_plot_bicycle["sem"] == "linear-ou")
    & (df_plot_bicycle["intervention_scale"] == 1)
    & (
        (df_plot_bicycle["x_distribution"] == "NormalNormal")
        | (df_plot_bicycle["x_distribution"] == "Normal")
    )
    # & ((df_plot_bicycle["x_distribution"] == "Normal") | (df_plot_bicycle["use_latents"].isna()))
    & ((df_plot_bicycle["use_latents"] == "False") | (df_plot_bicycle["use_latents"].isna())),
    :,
].copy()
make_plot(x)


# %%
