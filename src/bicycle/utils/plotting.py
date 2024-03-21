import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import numpy as np


def plot_training_results(
    trainer,
    pl_module,
    estimated_beta,
    true_beta,
    scale_l1,
    scale_kl,
    scale_spectral,
    scale_lyapunov,
    file_name_plot,
    callback=True,
    labels=None,
):
    fig, ax = plt.subplots(3, 2, figsize=(14, 16))
    df_plot = pd.DataFrame(trainer.logger.history).reset_index(drop=True)
    df_plot["epoch"] = df_plot.index
    df_plot_train = df_plot[[x for x in df_plot.columns if "train" in x] + ["epoch"]]
    df_plot_valid = df_plot[[x for x in df_plot.columns if "valid" in x] + ["epoch"]]

    df_plot_train = df_plot_train.melt(
        id_vars=["epoch"], value_vars=[x for x in df_plot.columns if "train_" in x]
    )
    df_plot_valid = df_plot_valid.melt(
        id_vars=["epoch"], value_vars=[x for x in df_plot.columns if "valid_" in x]
    )

    df_plot_train.to_csv( str(Path(file_name_plot).with_suffix("")) + f"_log_train.csv" )
    df_plot_valid.to_csv( str(Path(file_name_plot).with_suffix("")) + f"_log_valid.csv" )

    sns.scatterplot(
        df_plot_train,
        x="epoch",
        y="value",
        hue="variable",
        ax=ax[1, 0],
        s=10,
        edgecolor="none",
        linewidth=0,
    )
    sns.scatterplot(
        df_plot_train,
        x="epoch",
        y="value",
        hue="variable",
        ax=ax[2, 0],
        s=10,
        edgecolor="none",
        linewidth=0,
    )
    sns.scatterplot(
        df_plot_valid,
        x="epoch",
        y="value",
        hue="variable",
        ax=ax[1, 1],
        s=10,
        edgecolor="none",
        linewidth=0,
    )
    ax[1, 0].grid(True)
    ax[1, 1].grid(True)
    ax[2, 0].grid(True)
    ax[2, 0].set_title("Training")
    ax[1, 0].set_title("Training")
    ax[1, 1].set_title("Validation")
    ax[1, 0].set_yscale("log")
    ax[1, 1].set_yscale("log")

    if pl_module.n_genes <= 20:
        annot = True
    else:
        annot = False

    if true_beta is not None:
        sns.heatmap(
            true_beta,
            annot=annot,
            center=0,
            cmap="vlag",
            square=True,
            annot_kws={"fontsize": 7},
            ax=ax[0, 0],
            cbar=False,
        )
        
        np.save(str(Path(file_name_plot).with_suffix("")) + f"_true_beta.npy", true_beta )

    sns.heatmap(
        estimated_beta,
        annot=annot,
        center=0,
        cmap="vlag",
        square=True,
        annot_kws={"fontsize": 7},
        ax=ax[0, 1],
        cbar=False,
    )
    ax[0, 0].set_title("True")
    ax[0, 1].set_title("Estimated")
    
    np.save(str(Path(file_name_plot).with_suffix("")) +  f"_estimated_beta_epoch{trainer.current_epoch}.npy", estimated_beta)

    if pl_module.mask is not None:
        sns.heatmap(
            pl_module.mask.detach().cpu().numpy(),
            annot=annot,
            center=0,
            cmap="vlag",
            square=True,
            annot_kws={"fontsize": 7},
            ax=ax[2, 1],
            cbar=False,
        )
        ax[2, 1].set_title("Mask")

    # Do not show ax[2, 1]
    ax[2, 1].axis("off")

    # Create folder for file
    Path(file_name_plot).parent.mkdir(parents=True, exist_ok=True)

    if callback:
        plt.suptitle(
            f"L1: {scale_l1}, KL: {scale_kl}, Spectral: {scale_spectral}, Lyapunov: {scale_lyapunov} | Final T: {pl_module.T.item():.2f} | Epochs: {trainer.current_epoch}",
            fontsize=20,
        )
        fig.savefig(str(Path(file_name_plot).with_suffix("")) + f"_epoch{trainer.current_epoch}.png")
    else:
        plt.suptitle(
            f"L1: {scale_l1}, KL: {scale_kl}, Spectral: {scale_spectral}, Lyapunov: {scale_lyapunov} | Final T: {pl_module.T.item():.2f}",
            fontsize=20,
        )
        fig.savefig(file_name_plot)
    plt.tight_layout()
    plt.close(fig)

    if labels is not None:
        # sns.set(font_scale=0.5)
        df = pd.DataFrame(columns=labels)
        for i in range(len(labels)):
            df[labels[i]] = estimated_beta[:, i]

        df.index = labels
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        f = sns.clustermap(df, cmap="vlag", center=0)
        f.savefig(
            str(Path(file_name_plot).with_suffix("")) + f"_clustermap_epoch{trainer.current_epoch}.png",
            bbox_inches="tight",
        )
        plt.close(fig)
        plt.close("all")

        '''y_axis = [
            "CST3",
            "PTMA",
            "LGALS3",
            "CTPS1",
            "FKBP4",
            "SINHCAF",
            "SOX4",
            "CDK6",
            "HLA-A",
            "CCND1",
            "FOS",
            "SSR2",
            "SEC11C",
            "ACSL3",
            "SAT1",
            "PET100",
            "IFNGR2",
            "PUF60",
            "MYC",
            "STAT1",
            "LAMP2",
            "B2M",
            "IFNGR1",
            "SMAD4",
            "STOM",
            "GSEC",
            "TMEM173",
            "RNASEH2A",
            "GSN",
            "CD59",
            "TPRKB",
        ]
        x_axis = [
            "CKS1B",
            "CST3",
            "B2M",
            "HLA-A",
            "GSN",
            "CDK6",
            "FKBP4",
            "SSR2",
            "DNMT1",
            "SEC11C",
            "CCND1",
            "EIF3K",
            "ACSL3",
            "SINHCAF",
            "CDKN1A",
            "TXNDC17",
            "MRPL47",
            "VDAC2",
            "SAT1",
            "TMED10",
            "RRS1",
            "MYC",
            "SOX4",
            "HLA-B",
            "CD274",
            "IRF3",
            "HASPIN",
            "GSEC",
            "RNASEH2A",
            "JAK2",
            "TMEM173",
        ]

        # Convert list string to index using labels
        x_idx = [[x for x in labels].index(x) for x in x_axis]
        y_idx = [[x for x in labels].index(x) for x in y_axis]

        ref_paper_plot = pd.DataFrame(estimated_beta[y_idx][:, x_idx], columns=x_axis, index=y_axis)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        sns.heatmap(ref_paper_plot, cmap="vlag", center=0, xticklabels=True, yticklabels=True)
        plt.savefig(
            str(Path(file_name_plot).with_suffix("")) + f"_heatmap_epoch{trainer.current_epoch}.png", bbox_inches="tight"
        )
        plt.close(fig)
        plt.close("all")'''


def plot_style(minimal=True):
    MEDIUM_SIZE = 8
    BIGGER_SIZE = 10

    plt.style.use("default")
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.family"] = "STIXGeneral"
    if not minimal:
        plt.rcParams["font.size"] = MEDIUM_SIZE
        plt.rcParams["axes.labelsize"] = MEDIUM_SIZE
        plt.rcParams["legend.fontsize"] = MEDIUM_SIZE
        plt.rcParams["axes.titlesize"] = MEDIUM_SIZE
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
        plt.rcParams["figure.titlesize"] = MEDIUM_SIZE
