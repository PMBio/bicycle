from pathlib import Path
from bicycle.model import BICYCLE
import torch
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from bicycle.dictlogger import DictLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
import seaborn as sns
import pandas as pd

# %%
SEED = 3141
pl.seed_everything(SEED)

torch.set_float32_matmul_precision("medium")

# %%
batch_size = 512
N_SAMPLES = 0.5
HARD_THRESHOLD = True
num_downstream_genes = 5
num_perturbation_genes = 10

# %%
CUDA = torch.cuda.is_available()
device = torch.device("cpu")
if CUDA:
    GPU_DEVICE = 1
    torch.cuda.set_device(GPU_DEVICE)
    device = torch.device("cuda")

DATA_PATH = Path("/data/m015k/data/bicycle")

# %%
adata = sc.read_h5ad(DATA_PATH / "NormanWeissman2019_filtered.h5ad")

# Remove all genes with 0 variance
adata = adata[:, adata.X.todense().var(axis=0) > 0]

# %% [markdown]
# def cdf(x):
#     # Calculate and plot cumulative distribution function from 1D-samples
#     x, y = sorted(x), np.arange(len(x)) / len(x)
#     return (x, y)

# %% [markdown]
# gene_set = adata.obs.perturbation.unique()[:10]

# %% [markdown]
# var_names = adata.var_names.to_list()
#
# for gene in gene_set:
#     if gene in var_names:
#         plt.figure()
#         plt.title(gene)
#         x_obs, y_obs = cdf(
#             np.asarray(
#                 adata[adata.obs["perturbation"] == "control", gene].X.todense()
#             ).squeeze()
#         )
#         plt.plot(x_obs, y_obs, label="unperturbed: %d cells" % len(x_obs))
#
#         x_obs, y_obs = cdf(
#             np.asarray(
#                 adata[adata.obs["perturbation"] == gene, gene].X.todense()
#             ).squeeze()
#         )
#         plt.plot(x_obs, y_obs, label="perturbed: %d cells" % len(x_obs))
#
#         plt.legend()
#
#         plt.show()

# %%
adata.obs.head()

# %%
# Subsample rows randomly
if N_SAMPLES:
    if N_SAMPLES < 1:
        # Subsample adata to N per cent of samples
        adata = adata[
            np.random.choice(
                adata.shape[0], int(N_SAMPLES * adata.shape[0]), replace=False
            ),
            :,
        ]
    else:
        # Subsample adata to N samples
        adata = adata[np.random.choice(adata.shape[0], N_SAMPLES, replace=False), :]

# %%
# Genes that must be included
prio_downstream_genes = ["MMP13"]
prio_perturbation_genes = ["TGFBR2"]

# Print if genes are found or not in data
for gene in prio_downstream_genes:
    if gene in adata.var.index:
        print(f"✓ Gene {gene} found in data")
    else:
        print(f"✗ Gene {gene} not found in data")
        prio_downstream_genes.remove(gene)

for gene in prio_perturbation_genes:
    if gene in adata.obs.perturbation.unique():
        print(f"✓ Gene {gene} found in data")
    else:
        print(f"✗ Gene {gene} not found in data")
        # Remove gene from prio_perturbation_genes
        prio_perturbation_genes.remove(gene)

# %% [markdown]
# results = prepare_dataset(
#     adata,
#     p_val=0.2,
#     p_test=0.2,
#     min_perturbed_cells=0,
#     use_clusters_for_holdout=True,
#     stratify_regimes=True,
# )

# %%
# Find all genes that have been perturbed at least once
perturbed_genes = adata.obs.perturbation.unique()
perturbed_genes = [
    x for x in perturbed_genes if (not "_" in x) & (x in adata.var.index)
]
perturbation_genes = list(set(perturbed_genes))
print(f"Number of single perturbations: {len(perturbation_genes)}")
print(perturbation_genes)

# %%
# CLUSTER perturbations
import math

pert_name = "perturbation"
intervened_variables = perturbation_genes

adata_perturbed = adata[adata.obs[pert_name].isin(intervened_variables)]
sc.tl.dendrogram(adata_perturbed, groupby=pert_name)
clusters = list()

# Individual leave colors
leave_colors = adata_perturbed.uns["dendrogram_" + pert_name]["dendrogram_info"][
    "leaves_color_list"
]
names = adata_perturbed.uns["dendrogram_" + pert_name]["dendrogram_info"]["ivl"]

for leave_color in set(leave_colors):
    cluster = [names[i] for i, lc in enumerate(leave_colors) if lc == leave_color]
    clusters.append(cluster)

for c in clusters:
    print(f"Lenght of cluster: {len(c)}")

# %%
# Find cluster of prio_perturbation_genes
perturbation_cluster = []
for idx, c in enumerate(clusters):
    if any([x in c for x in prio_perturbation_genes]):
        perturbation_cluster.extend(c)

# %%
# Select a subset of genes, but only those that are in the data
selected_perturbed_genes = perturbation_cluster
all_perturbation_wo_pairs = [
    x
    for x in adata.obs.perturbation.to_list()
    if (not "_" in x) & (x in adata.var.index)
]
if len(selected_perturbed_genes) > num_perturbation_genes:
    print("Warning: selected more genes than initially planned")
    if HARD_THRESHOLD:
        print("- Using hard threshold")
        selected_perturbed_genes = selected_perturbed_genes[:num_perturbation_genes]
else:
    not_included_genes = list(
        set(all_perturbation_wo_pairs) - set(selected_perturbed_genes)
    )
    selected_perturbed_genes = (
        selected_perturbed_genes
        + not_included_genes[: (num_perturbation_genes - len(selected_perturbed_genes))]
    )
selected_perturbed_genes = selected_perturbed_genes + ["control"]
print(selected_perturbed_genes)

# %% [markdown]
# selected_perturbed_genes_wo_control = [
#     x for x in selected_perturbed_genes if x != "control"
# ]
# np.corrcoef(
#     adata[:, selected_perturbed_genes_wo_control].X.todense(), rowvar=False
# )[: len(selected_perturbed_genes_wo_control), -1].shape

# %%
selected_perturbed_genes_wo_control = [
    x for x in selected_perturbed_genes if x != "control"
]

# Find genes that are highly correlated with genes in selected_perturbed_genes
x1 = adata[:, selected_perturbed_genes_wo_control].X.todense()
names = [x for x in adata.var.index if x not in selected_perturbed_genes_wo_control]
x2 = adata[:, names].X.todense()

# %%
# Subsample both dfs
n_samples = 1000
idx = np.random.choice(x1.shape[0], 1000, replace=False)
x1 = x1[idx, :]
x2 = x2[idx, :]
# Compute column wise correlation between x1 and x2
corr = np.corrcoef(x1, x2, rowvar=False)

# %%
c = np.abs(
    corr[
        len(selected_perturbed_genes_wo_control) :,
        : len(selected_perturbed_genes_wo_control),
    ]
)

# %% [markdown]
# import seaborn as sns
# sns.heatmap(c, cmap="vlag", center=0)


# %%
def top_indices(matrix, num_indices):
    flat_indices = np.argsort(matrix.ravel())[::-1]
    row_indices, _ = np.unravel_index(flat_indices, matrix.shape)
    return row_indices


top_row_indices = list(top_indices(c, num_downstream_genes))
# Selct first num_downstream_genes different genes
top_genes = []
for k in top_row_indices:
    if not names[k] in top_genes:
        top_genes.append(names[k])
    if len(top_genes) == num_downstream_genes:
        break
print("Top row indices:", top_genes)

prio_downstream_genes = list(set(prio_downstream_genes) | set(top_genes))

# %% [markdown]
# # Fing top 10 highest correlation values and return the corresponding genes
# top_n = 10
# top_n_idx = np.argsort(c, axis=0)[-top_n:, :].flatten()
# top_n_genes = [adata.var.index[i] for i in top_n_idx]
# top_n_genes

# %%
# All genes in study we care about
all_gene_in_graph = list(
    (set(selected_perturbed_genes) | set(prio_downstream_genes)) - set(["control"])
)
print(all_gene_in_graph)
assert any([x for x in selected_perturbed_genes if x not in all_gene_in_graph])

# %%
samples = adata
print(samples.shape)

# Subset obs data to only include perturbation we want to use
samples = adata[adata.obs.perturbation.isin(selected_perturbed_genes), :]
print(samples.shape)

# Optional: subset genes
samples = samples[:, samples.var.index.isin(all_gene_in_graph)]
print(f"Count matrix shape: {samples.shape}")

# %%
# Create a mapping between gene and context:
gene_to_intervention_idx = {"control": 0}
c = 1
for k in all_gene_in_graph:
    if (k in selected_perturbed_genes) & (k != "control"):
        gene_to_intervention_idx[k] = c
        c += 1
    else:
        gene_to_intervention_idx[k] = None

gene_to_intervention_idx

# %%
all_gene_in_graph_map_to_idx = {k: i for i, k in enumerate(all_gene_in_graph)}
all_gene_in_graph_map_to_idx

# %%
# Genes x regimes matrices, which indexes which genes are intervened in
# which context
n_contexts = len(selected_perturbed_genes)  # including control
gt_interv = torch.zeros((len(gene_to_intervention_idx) - 1, n_contexts), device=device)
for n, name in enumerate(all_gene_in_graph):
    if gene_to_intervention_idx.get(name):
        gt_interv[n, gene_to_intervention_idx[name]] = 1
print(f"{gt_interv.shape=}")
gt_interv

# %%
sim_regime = torch.tensor(
    [x for x in samples.obs.perturbation.map(gene_to_intervention_idx).values]
).long()
# intervened_variables = gt_interv[:, sim_regime].transpose(0, 1)
plt.hist(sim_regime.cpu().numpy(), bins=100)

# %%
samples_normalized = torch.tensor(samples.X.todense())
print(f"{samples_normalized.shape=}")

# Log x+1 transform
samples_normalized = torch.log(samples_normalized + 1)
# z-score
samples_normalized = (
    samples_normalized - samples_normalized.mean(dim=0)
) / samples_normalized.std(dim=0)

# %%
plt.hist(samples_normalized.flatten().cpu().numpy(), bins=100)

# %%
print(f"Final dataset size: {samples_normalized.shape=}")

# %%
samples_interventions = torch.tensor(
    samples.obs.perturbation.map(gene_to_intervention_idx).values
).long()
samples_interventions = samples_interventions.to(torch.int64)

train_data = torch.utils.data.TensorDataset(
    samples_normalized,
    samples_interventions,
)

# %%
# Stratified Sampling for train and val
train_idx, validation_idx = train_test_split(
    np.arange(len(samples_normalized)),
    test_size=0.25,
    random_state=SEED,
    shuffle=True,
    stratify=samples_interventions,
)

# %%
# Subset dataset for train and val
train_dataset = Subset(train_data, train_idx)
validation_dataset = Subset(train_data, validation_idx)

# Dataloader for train and val
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
    num_workers=4,
)
validation_loader = DataLoader(
    validation_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=True,
    pin_memory=True,
    num_workers=4,
)

# %%
lr = 1e-4
early_stopping = True
n_epochs = 25_000
n_plot_intervals = 1
lyapunov_penalty = True
perfect_interventions = True
n_genes = samples.shape[1]

GPU_DEVICE = 0
device = torch.device(f"cuda:{GPU_DEVICE}")
gt_interv = gt_interv.to(device)

for scale_l1 in [0]:
    for scale_spectral_loss in [10]:
        for scale_lyapunov in [1, 10]:
            print(f"Scale Lyapunov: {scale_lyapunov}")
            print(f"Scale spectral loss: {scale_spectral_loss}")
            model = BICYCLE(
                lr,
                gt_interv,
                n_genes,
                lyapunov_penalty=lyapunov_penalty,
                perfect_interventions=perfect_interventions,
                rank_w_cov_factor=2,
                inits=None,
                optimizer="adam",
                device=device,
                normalise=False,
                scale_l1=scale_l1,
                scale_lyapunov=scale_lyapunov,
                scale_spectral_loss=scale_spectral_loss,
                early_stopping=True,
                early_stopping_min_delta=0.02,
                early_stopping_patience=400,
                early_stopping_p_mode=True,
            )
            model.to(device)

            dlogger = DictLogger()
            loggers = [dlogger]

            from pytorch_lightning.callbacks import RichProgressBar

            assert str(device).startswith("cuda")
            trainer = pl.Trainer(
                max_epochs=n_epochs,
                accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                logger=loggers,
                log_every_n_steps=n_plot_intervals,
                enable_model_summary=True,
                enable_progress_bar=True,
                enable_checkpointing=False,
                check_val_every_n_epoch=10,
                devices=[GPU_DEVICE] if str(device).startswith("cuda") else 1,
                num_sanity_val_steps=0,
                callbacks=[RichProgressBar()],
            )

            # try:
            trainer.fit(model, train_loader)

            # Plot training curve
            fig, ax = plt.subplots(1, 2, figsize=(15, 5))
            df_plot = pd.DataFrame(
                {
                    "train_loss": trainer.logger.history["train_loss"].reset_index(
                        drop=True
                    ),
                    "val_loss": trainer.logger.history["val_loss"].reset_index(
                        drop=True
                    ),
                },
            ).reset_index(drop=True)
            ax[0].scatter(
                range(len(df_plot)), df_plot["train_loss"], label="train_loss"
            )
            ax[1].scatter(range(len(df_plot)), df_plot["val_loss"], label="val_loss")
            ax[0].grid(True)
            ax[1].grid(True)
            plt.show()

            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            sns.heatmap(
                model.beta.detach().cpu().numpy(),
                center=0,
                cmap="vlag",
                annot=True,
                fmt=".2f",
            )


# %% [markdown]
# import pandas as pd
# pd.DataFrame(trainer.logger.history)
