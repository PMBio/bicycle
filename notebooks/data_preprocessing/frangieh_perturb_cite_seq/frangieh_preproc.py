"""
This script reproduces the data generation of the perturb-cite-seq data used in the NODAGS-Flow paper.
"""
import warnings

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*The loaded checkpoint was .*")
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from bicycle.utils.data import create_loaders
import torch
import pytorch_lightning as pl
import os

pl.seed_everything(0)

DATA_PATH = Path("/g/stegle/ueltzhoe/bicycle/data")

raw_counts = True

#
# Notebook 1
# Uncommented, because it saves a file (l. 56) which is loaded in Notebook 2. Hence
# we only need to run it once.
#
# To download the files, follow the instructions in: https://github.com/Genentech/nodags-flows/blob/master/perturb_cite_seq/0-data-download.ipynb
#
'''
print('Reading RNA_expression...')

# Load data from DATA_PATH
data = sc.read_csv(DATA_PATH / "SCP1064/other/RNA_expression.csv.gz").transpose()

print('Sparsifying...')

# sparsify!
data_sp = sp.csr_matrix(data.X)

print('Reading metadata...')

# get covariates
covariates = pd.read_csv(DATA_PATH / "SCP1064/metadata/RNA_metadata.csv", index_col=0).iloc[1:,]
data.obs = covariates

# correct dtypes
data.obs["MOI"] = data.obs["MOI"].astype(np.int32)
data.obs["UMI_count"] = data.obs["UMI_count"].astype(np.double)

# de-normalize and round up
norm_factor = data.obs["UMI_count"].values / 1.0e6
Z = sp.diags(norm_factor).dot(np.expm1(data_sp))
print(np.greater_equal(np.abs(Z.data - np.rint(Z.data)), 0.01).any())
Z.data = np.rint(Z.data)
data.X = Z

# read guide info
guide_info = pd.read_csv(DATA_PATH / "SCP1064/documentation/all_sgRNA_assignments.txt", index_col=0)
guide_info = guide_info.replace(np.nan, "", regex=True)
data.obs["sgRNAs"] = guide_info["sgRNAs"].astype(str)

data.write_h5ad(DATA_PATH / "SCP1064/other/adata.h5ad")

# REPRODUCTION CHECKS
# ADDITIONAL CHECKS TO ENSURE DATA CONSISTENCY WITH THE PAPER
assert guide_info.shape[0] == 218331
assert all(covariates["condition"].value_counts().values == np.array([87590, 73114, 57627]))
assert covariates["MOI"].value_counts()[1] == 126966
assert covariates["MOI"].value_counts()[10] == 64
assert covariates["MOI"].value_counts()[16] == 1
assert np.allclose(covariates["MOI"].mean(), 1.3874850570922133)
assert covariates["sgRNA"].value_counts()["IFNGR2_2"] == 358
assert covariates["sgRNA"].value_counts()["NO_SITE_23"] == 296
assert covariates["sgRNA"].value_counts()["PSMA7_1"] == 2
'''
#
# Notebook 2
#
# Reproduces: https://github.com/Genentech/nodags-flows/blob/master/perturb_cite_seq/1-assignments-vs-variability.ipynb
'''
adata = sc.read_h5ad(DATA_PATH / "SCP1064/other/adata.h5ad")

sc.pp.filter_cells(adata, min_genes=0)
sc.pp.filter_genes(adata, min_cells=0)

# for val in adata.obs[["sgRNA", "sgRNAs"]].value_counts().to_frame().iterrows():
#     print(val[0], val[1])

sc.pp.filter_cells(adata, min_genes=500)
sc.pp.filter_genes(adata, min_cells=500)

adata.layers["counts"] = adata.X.copy()

sc.pp.normalize_total(adata, target_sum=1e5)
sc.pp.log1p(adata)

# check if the field MOI has some irregularities, such that a truncated guide or a wrong number
ind = []
for index, row in tqdm(adata.obs.iterrows(), total=adata.n_obs):
    flag = False
    if row["sgRNAs"] != "":
        guides = row["sgRNAs"].split(",")
        if len(guides) != row["MOI"]:
            flag = True
        if "_" not in guides[-1]:
            flag = True
    if flag:
        ind += [index]

# check gene sets and ensure matching with measurements
err = 0
ind = []
obs_genes = {}
unfound_genes = {}
targets = []
for index, row in tqdm(adata.obs.iterrows(), total=adata.n_obs):
    current_target = []
    if row["sgRNAs"] != "":
        # get all guides in cells
        sg = row["sgRNAs"].split(",")
        # get gene name by stripping guide specific info
        sg_genes = [guide.rsplit("_", maxsplit=1)[0] for guide in sg]
        for gene in sg_genes:
            if gene in adata.var.index:
                # gene is found
                current_target += [gene]
                if gene not in obs_genes:
                    obs_genes[gene] = 1
                else:
                    obs_genes[gene] += 1
            else:
                if gene not in unfound_genes:
                    unfound_genes[gene] = 1
                else:
                    unfound_genes[gene] += 1
    # end gene list
    targets += [",".join(current_target)]


# create regimes
regimes = np.unique(targets, return_inverse=True)[1]

# REPRODUCTION CHECKS
assert len(regimes) == 218027

adata.obs["targets"] = targets
adata.obs["regimes"] = regimes

# REPRODUCTION CHECKS
assert len(obs_genes.keys()) == 220

# REPRODUCTION CHECKS
reference = [
    "NGFR",
    "SERPINF1",
    "CSPG4",
    "PIK3IP1",
    "ONE_NON-GENE_SITE",
    "NO_SITE",
    "NUP50-AS1",
    "IDI2-AS1",
    "CXCR4",
    "JMJD7",
    "TYR",
    "BOLA2",
    "LRRC75A-AS1",
    "LINC00518",
    "APOD",
    "A2M",
    "LEF1-AS1",
    "SLC7A5P1",
    "SERPINA3",
    "WNT7A",
    "GAS5",
    "IRF4",
    "APOC2",
    "XAGE1A",
    "CCND2",
    "CDH19",
    "ST6GALNAC2",
    "S100B",
    "HLA-H",
    "SNHG6",
]
diff = set(reference) - set(unfound_genes.keys()) | set(unfound_genes.keys()) - set(reference)
assert len(diff) == 0

sc.pp.highly_variable_genes(adata, layer="counts", flavor="seurat_v3", n_top_genes=3000, span=0.2)

plt.scatter(adata.var["means"], adata.var["variances"], c=adata.var["highly_variable"])
plt.xlabel("means")
plt.ylabel("variances")
plt.xlim((0, 50))
plt.ylim((0, 2000))
plt.show()

presence = pd.Series(data=np.zeros_like(adata.var["means"]), index=adata.var.index)
for gene in list(obs_genes.keys()):
    presence[gene] += 1
adata.var["targeted"] = presence

plt.scatter(adata.var["means"], adata.var["variances"], c=adata.var["targeted"])
plt.xlabel("means")
plt.ylabel("variances")
plt.xlim((0, 50))
plt.ylim((0, 2000))
plt.show()

# REPRODUCTION CHECKS
assert (adata.var["highly_variable"] * adata.var["targeted"]).sum() == 100

np.unique(adata.obs.condition.values)

if raw_counts:
    adata.X = adata.layers["counts"]

# filter genes (1000)
to_keep = np.logical_or(adata.var["highly_variable_rank"] < 1500, adata.var["targeted"])
adata_gf = adata[:, to_keep].copy()

# Loading the preselected genes
# This is the renamed temp.csv in https://github.com/Genentech/nodags-flows/tree/master/perturb_cite_seq
chosen_genes = pd.read_csv(DATA_PATH / "220930_PerturbCITE_regulators.csv", index_col=0)
# ADDED BY MARTIN
chosen_genes = chosen_genes.index
# chosen_genes = chosen_genes.to_numpy().T.squeeze()

present = 0
final_genes = list()
for gene in chosen_genes:
    if gene.upper() in adata_gf.var[adata.var["targeted"] > 0].index:
        present += 1
        final_genes.append(gene.upper())

print(present)

# THIS IS COPIED FROM THE THIRD NOTEBOOK
final_genes = [
    "ACSL3",
    "ACTA2",
    "B2M",
    "CCND1",
    "CD274",
    "CD58",
    "CD59",
    "CDK4",
    "CDK6",
    "CDKN1A",
    "CKS1B",
    "CST3",
    "CTPS1",
    "DNMT1",
    "EIF3K",
    "EVA1A",
    "FKBP4",
    "FOS",
    "GSEC",
    "GSN",
    "HASPIN",
    "HLA-A",
    "HLA-B",
    "HLA-C",
    "HLA-E",
    "IFNGR1",
    "IFNGR2",
    "ILF2",
    "IRF3",
    "JAK1",
    "JAK2",
    "LAMP2",
    "LGALS3",
    "MRPL47",
    "MYC",
    "P2RX4",
    "PABPC1",
    "PAICS",
    "PET100",
    "PTMA",
    "PUF60",
    "RNASEH2A",
    "RRS1",
    "SAT1",
    "SEC11C",
    "SINHCAF",
    "SMAD4",
    "SOX4",
    "SP100",
    "SSR2",
    "STAT1",
    "STOM",
    "TGFB1",
    "TIMP2",
    "TM4SF1",
    "TMED10",
    "TMEM173",
    "TOP1MT",
    "TPRKB",
    "TXNDC17",
    "VDAC2",
]

in_final_gene = list()
for index, row in adata.var.iterrows():
    if index in final_genes:
        in_final_gene.append(True)
    else:
        in_final_gene.append(False)


adata.var["in_final"] = in_final_gene

to_keep = adata.var["in_final"]
adata_gf = adata[:, to_keep].copy()

if raw_counts:
    OUT_DIR = "ready/COUNTS/"
else:
    OUT_DIR = "ready"

# Make sure output directories exist
for condition in ['control','cocult','ifn']:
    READY_PATH =  DATA_PATH / "SCP1064" / OUT_DIR / condition

    if os.path.isdir(READY_PATH):
        print('Output directory exists: ',READY_PATH)
    else:
        os.makedirs(READY_PATH)
        print("Created folder: ", READY_PATH)

adata_gf[adata_gf.obs.condition == "Control"].copy().write_h5ad(
    DATA_PATH / "SCP1064" / OUT_DIR / "control/gene_filtered_adata.h5ad"
)
adata_gf[adata_gf.obs.condition == "Co-culture"].copy().write_h5ad(
    DATA_PATH / "SCP1064" / OUT_DIR / "cocult/gene_filtered_adata.h5ad"
)
adata_gf[adata_gf.obs.condition == "IFNγ"].copy().write_h5ad(
    DATA_PATH / "SCP1064" / OUT_DIR / "ifn/gene_filtered_adata.h5ad"
)

adata_small = adata_gf[np.random.choice(np.arange(adata.n_obs), size=10000, replace=False)].copy()

if raw_counts:
    OTHER_COUNT_DIR = DATA_PATH / "SCP1064/other/COUNTS/"
    if os.path.isdir(OTHER_COUNT_DIR):
        print('Output directory exists: ', OTHER_COUNT_DIR)
    else:
        os.makedirs(OTHER_COUNT_DIR)
        print("Created folder: ", OTHER_COUNT_DIR)
    adata_small.write_h5ad(DATA_PATH / "SCP1064/other/COUNTS/small_adata.h5ad")
else:
    adata_small.write_h5ad(DATA_PATH / "SCP1064/other/small_adata.h5ad")
'''
# adf = adata_gf[adata_gf.obs.condition == "Control"]
# gene_list = adf.var.index.values
# targets = list(set([x for x in adf.obs.targets.values]))
# tx = []
# for x in targets:
#     tx += x.split(",")

#
# Notebook 3
#
# Reproduces: https://github.com/Genentech/nodags-flows/blob/master/perturb_cite_seq/Untitled.ipynb

def generate_nodags_data(adata_path):
    adata = sc.read_h5ad(adata_path)
    data = sp.csr_matrix.toarray(adata.X)

    final_genes = adata.var.index

    datasets = []

    for gene in final_genes:
        datasets.append(data[adata.obs["targets"] == gene, :])

    intervention_sets = [[i] for i in range(61)]

    return datasets, intervention_sets

if raw_counts:
    OUT_DIR = "ready/COUNTS/"
else:
    OUT_DIR = "ready"

control_datasets, control_interventions = generate_nodags_data(
    DATA_PATH / "SCP1064" / OUT_DIR / "control/gene_filtered_adata.h5ad"
)
cocult_datasets, cocult_interventions = generate_nodags_data(
    DATA_PATH / "SCP1064" / OUT_DIR / "cocult/gene_filtered_adata.h5ad"
)
ifn_datasets, ifn_interventions = generate_nodags_data(
    DATA_PATH / "SCP1064" / OUT_DIR / "ifn/gene_filtered_adata.h5ad"
)

control_training_data, control_training_interventions = control_datasets[:-6], control_interventions[:-6]
cocult_training_data, cocult_training_interventions = cocult_datasets[:-6], cocult_interventions[:-6]
ifn_training_data, ifn_training_interventions = ifn_datasets[:-6], ifn_interventions[:-6]

control_validation_data, control_validation_interventions = control_datasets[-6:], control_interventions[-6:]
cocult_validation_data, cocult_validation_interventions = cocult_datasets[-6:], cocult_interventions[-6:]
ifn_validation_data, ifn_validation_interventions = ifn_datasets[-6:], ifn_interventions[-6:]

print('TEST INTERVENTIONS:',control_interventions[-6:], cocult_interventions[-6:], ifn_interventions[-6:])

if raw_counts:
    NODAGS_PATH = "nodags_data/COUNTS"
else:
    NODAGS_PATH = "nodags_data"

# Make sure output directories exist
for condition in ['control','cocult','ifn']:
    for split in ['training_data','validation_data','test_data']:
        READY_PATH =  DATA_PATH /  NODAGS_PATH / condition / split

        if os.path.isdir(READY_PATH):
            print('Output directory exists: ',READY_PATH)
        else:
            os.makedirs(READY_PATH)
            print("Created folder: ", READY_PATH)

# saving control data
for i, dataset in enumerate(control_training_data):
    np.save(DATA_PATH /  NODAGS_PATH / "control/training_data/dataset_{}.npy".format(i), dataset)
np.save(DATA_PATH / NODAGS_PATH / "control/training_data/intervention_sets.npy", control_training_interventions)

for i, dataset in enumerate(control_validation_data):
    np.save(DATA_PATH / NODAGS_PATH / "control/validation_data/dataset_{}.npy".format(i), dataset)
np.save(
    DATA_PATH / NODAGS_PATH / "control/validation_data/intervention_sets.npy", control_validation_interventions
)

# saving cocult data
for i, dataset in enumerate(cocult_training_data):
    np.save(DATA_PATH / NODAGS_PATH / "cocult/training_data/dataset_{}.npy".format(i), dataset)
np.save(DATA_PATH / NODAGS_PATH / "cocult/training_data/intervention_sets.npy", cocult_training_interventions)

for i, dataset in enumerate(cocult_validation_data):
    np.save(DATA_PATH / NODAGS_PATH / "cocult/validation_data/dataset_{}.npy".format(i), dataset)
np.save(
    DATA_PATH / NODAGS_PATH / "cocult/validation_data/intervention_sets.npy", cocult_validation_interventions
)

# saving ifn data
for i, dataset in enumerate(ifn_training_data):
    np.save(DATA_PATH / NODAGS_PATH / "ifn/training_data/dataset_{}.npy".format(i), dataset)
np.save(DATA_PATH / NODAGS_PATH / "ifn/training_data/intervention_sets.npy", ifn_training_interventions)

for i, dataset in enumerate(ifn_validation_data):
    np.save(DATA_PATH / NODAGS_PATH / "ifn/validation_data/dataset_{}.npy".format(i), dataset)
np.save(DATA_PATH / NODAGS_PATH / "ifn/validation_data/intervention_sets.npy", ifn_validation_interventions)

# saving control data
for i, dataset in enumerate(control_datasets):
    np.save(DATA_PATH / NODAGS_PATH / "control/dataset_{}.npy".format(i), dataset)
np.save(DATA_PATH / NODAGS_PATH / "control/intervention_sets.npy", control_interventions)

# saving cocult data
for i, dataset in enumerate(cocult_datasets):
    np.save(DATA_PATH / NODAGS_PATH / "cocult/dataset_{}.npy".format(i), dataset)
np.save(DATA_PATH / NODAGS_PATH / "cocult/intervention_sets.npy", cocult_interventions)

# saving ifn data
for i, dataset in enumerate(ifn_datasets):
    np.save(DATA_PATH / NODAGS_PATH / "ifn/dataset_{}.npy".format(i), dataset)
np.save(DATA_PATH / NODAGS_PATH / "ifn/intervention_sets.npy", ifn_interventions)

np.save(DATA_PATH / NODAGS_PATH / "control/training_data/weights.npy", np.eye(61))
np.save(DATA_PATH / NODAGS_PATH / "cocult/training_data/weights.npy", np.eye(61))
np.save(DATA_PATH / NODAGS_PATH / "ifn/training_data/weights.npy", np.eye(61))

np.save(DATA_PATH / NODAGS_PATH / "control/validation_data/weights.npy", np.eye(61))
np.save(DATA_PATH / NODAGS_PATH / "cocult/validation_data/weights.npy", np.eye(61))
np.save(DATA_PATH / NODAGS_PATH / "ifn/validation_data/weights.npy", np.eye(61))

'''data = sp.csr_matrix.toarray(adata.X)

final_genes = adata.var.index

gene_int_samples = {gene: 0 for gene in final_genes}
gene_int_samples[""] = 0


def checkTargetsinFinalGenes(targets, final_genes):
    targets_list = targets.split(",")
    ans = True
    for target in targets_list:
        if targets not in final_genes:
            ans = False

    return ans


useful_samples = 0
single_inter_cells = 0
unique_targets = list()
for index, row in adata.obs.iterrows():
    if checkTargetsinFinalGenes(row["targets"], final_genes):
        useful_samples += 1
        unique_targets.append(row["targets"])
        gene_int_samples[row["targets"]] += 1
    if row["MOI"] == 1 or row["MOI"] == 0:
        single_inter_cells += 1

assert useful_samples == 8013
assert single_inter_cells == 35428

obs_data = data[adata.obs["targets"] == "", :]
datasets = [obs_data]

for gene in final_genes:
    datasets.append(data[adata.obs["targets"] == gene, :])

intervention_sets = [[None]] + [[i] for i in range(61)]

obs_data = data[adata.obs["targets"] == "", :]
b2m_data = data[adata.obs["targets"] == "B2M", :]

if not raw_counts:
    obs_data_cent = obs_data - obs_data.mean(axis=0)
    x = np.array(
        [
            1.35076493,
            0.27123336,
            1.43438899,
            0.57417099,
            0.65185934,
            1.22373799,
            1.13910842,
            1.35342653,
            1.37653152,
            0.94962468,
            1.46981513,
            0.90265875,
            1.35275132,
            1.27338793,
            0.8543487,
            1.28981213,
            1.28209372,
            0.58883364,
            0.63089262,
            0.8985343,
            0.4505661,
            0.91311169,
            1.42362646,
            0.5947215,
            1.28714299,
            1.26883561,
            1.27623896,
            1.07184669,
            1.24687354,
            1.32525902,
            0.56226679,
            1.32019722,
            0.58940607,
            1.24761515,
            1.2619307,
            0.55745304,
            0.8130263,
            1.29347191,
            1.20980449,
            0.46099867,
            1.19850521,
            0.17897846,
            1.35106942,
            1.37135192,
            1.30646342,
            1.45254706,
            0.85252696,
            1.38077169,
            1.37243477,
            0.89969125,
            1.28038136,
            1.04200766,
            1.26619642,
            1.07230801,
            1.41598745,
            1.37603538,
            0.54119139,
            0.79979143,
            1.36777045,
            1.21083722,
            1.15708291,
        ]
    )
    assert np.allclose(b2m_data.std(axis=0), x)
    x = np.array(
        [
            1.05757233e-14,
            -5.30623690e-16,
            -1.02542300e-14,
            -1.21662939e-14,
            6.83211245e-16,
            -9.73198086e-16,
            -8.54192518e-15,
            1.73065370e-15,
            1.59726136e-14,
            -7.85147831e-17,
            -8.78787734e-15,
            -8.31780239e-16,
            -1.41553013e-16,
            5.78961624e-15,
            7.54971932e-16,
            -2.72444777e-15,
            2.90867957e-15,
            8.26806538e-16,
            -5.19106562e-16,
            -2.42375321e-15,
            -5.03361789e-16,
            -9.62857856e-16,
            1.37384819e-15,
            -7.12553336e-15,
            -1.67616474e-15,
            -8.57850459e-16,
            2.51983963e-15,
            1.19903073e-14,
            1.39545792e-15,
            -3.20036866e-15,
            -1.20663468e-16,
            2.57241936e-15,
            -7.80731269e-15,
            1.20480233e-14,
            -2.75077142e-15,
            -4.23358063e-16,
            1.09019812e-14,
            2.75982757e-15,
            -1.43016190e-15,
            -7.13800246e-15,
            1.03659113e-15,
            2.82058487e-16,
            -2.13576430e-15,
            -3.88145526e-15,
            9.54741114e-15,
            3.44728389e-15,
            9.19017479e-16,
            2.17033311e-15,
            1.81920459e-15,
            -2.56798590e-14,
            -3.72167168e-15,
            -1.14502837e-16,
            3.58613357e-16,
            1.38179260e-14,
            1.00095451e-14,
            2.01128618e-14,
            1.50108624e-16,
            -1.83741826e-17,
            7.78544952e-15,
            -1.58574518e-14,
            -5.53523307e-15,
        ]
    )
    assert np.allclose(obs_data_cent.mean(axis=0), x)

# REPRODUCTION CHECKS
assert min([x[0] for x in control_training_interventions]) == 0
assert max([x[0] for x in control_training_interventions]) == 54
assert sp.csr_matrix.toarray(adata.X).shape == (57523, 61)
assert adata.obs[adata.obs["condition"] == "Control"]["targets"].value_counts()["TSC22D3"] == 216
assert adata.obs[adata.obs["condition"] == "Control"]["targets"].value_counts()["RTP4"] == 208
assert adata.obs[adata.obs["condition"] == "Control"]["targets"].value_counts()["CGAS,UQCRH"] == 1
'''
#
# Additional code to save data directly as train loaders
#

# %%
# Create loaders for Control experiments
# Concatenate all list elements
for name, train_samples_, valid_samples_, train_interv_, valid_interv_ in [
    (
        "control",
        control_training_data,
        control_validation_data,
        control_training_interventions,
        control_validation_interventions,
    ),
    (
        "cocult",
        cocult_training_data,
        cocult_validation_data,
        cocult_training_interventions,
        cocult_validation_interventions,
    ),
    ("ifn", ifn_training_data, ifn_validation_data, ifn_training_interventions, ifn_validation_interventions),
]:

    adata = sc.read_h5ad(DATA_PATH / f"SCP1064/ready/{name}/gene_filtered_adata.h5ad")
    data = sp.csr_matrix.toarray(adata.X)
    obs_data = data[adata.obs["targets"] == "", :]
    obs_data_mu = obs_data.mean(axis=0)

    np.save(DATA_PATH / NODAGS_PATH / name / "training_data/obs_means.npy", obs_data_mu)

    for split in ["training_data","validation_data","test_data"]:
        if not os.path.isdir(DATA_PATH / NODAGS_PATH / name / split):
            os.makedirs(DATA_PATH / NODAGS_PATH / name / split)

    dataset_train = np.concatenate(train_samples_, axis=0)
    dataset_train_targets = np.concatenate(
        [np.ones(train_samples_[k].shape[0]) * i for k, i in enumerate(train_interv_)]
    )
    datset_test = np.concatenate(valid_samples_, axis=0)
    dataset_test_targets = np.concatenate(
        [np.ones(valid_samples_[k].shape[0]) * i for k, i in enumerate(valid_interv_)]
    )

    samples = np.concatenate((dataset_train, datset_test), axis=0)
    sim_regimes = np.concatenate((dataset_train_targets, dataset_test_targets), axis=0)

    if not raw_counts:
        # Remove averages from all samples
        samples = samples - obs_data_mu

    # Convert to torch
    samples = torch.from_numpy(samples).float()
    sim_regimes = torch.from_numpy(sim_regimes).long()

    train_loader, validation_loader, test_loader = create_loaders(
        samples,
        sim_regimes,
        validation_size=0.0,
        batch_size=1024,
        SEED=0,
        train_gene_ko=[x[0] for x in control_training_interventions],
        test_gene_ko=[x[0] for x in control_validation_interventions],
        num_workers=1,
        persistent_workers=False,
        # prefetch_factor=4,
    )

    # for batch in test_loader:
    #     samples, interventions, id, k = batch
    #     print(min(interventions), max(interventions))

    # Save all loaders as pickle
    torch.save(train_loader, DATA_PATH / NODAGS_PATH / name / "training_data/train_loader.pth")
    torch.save(validation_loader, DATA_PATH / NODAGS_PATH / name / "validation_data/validation_loader.pth")
    torch.save(test_loader, DATA_PATH / NODAGS_PATH / name / "test_data/test_loader.pth")
    # Save labels
    np.save(DATA_PATH / NODAGS_PATH / name / "labels.npy", adata.var.index.values)

# %%
