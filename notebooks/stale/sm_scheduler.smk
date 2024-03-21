from pathlib import Path
import os

user_dir = "/omics/groups/OE0540/internal/users/rohbeck/bicycle/"
MODEL_PATH = Path(os.path.join(user_dir, "models"))
PLOT_PATH = Path(os.path.join(user_dir, "plots"))
MODEL_PATH.mkdir(parents=True, exist_ok=True)
PLOT_PATH.mkdir(parents=True, exist_ok=True)

n_genes = 20  # Number of modelled genes
graph = "cycle-random"
graph_kwargs = {"abs_weight_low": 0.25, "abs_weight_high": 0.95, "p_success": 0.4}
graph_kwargs_str = "_".join([f"{v}" for v in graph_kwargs.values()])
# LEARNING
batch_size = 1024
USE_INITS = False
use_encoder = False
optimizer = "adam"
# DATA
# LOGO = []
# LOGO = [0, 1, 2]
# LOGO = [0, 1, 2, 3, 4, 5]
# LOGO = [0, 1, 2, 3, 4, 5, 6, 7, 8]
# LOGO = [0, 1, 2, 3, 4, 5, 6, 7, 8]
# train_gene_ko = [str(x) for x in set(range(0, n_genes)) - set(LOGO)]  # We start counting at 0
# test_gene_ko = [str(x) for x in LOGO]
x_distribution = "Normal"  # "Poisson" if "random" in graph else 

lyapunov_penalty = False
lr = [1e-3] 
scale_l1 = [1, 10]
scale_kl = [0.1]
scale_spectral = [1, 10]
scale_lyapunov = [0]
n_genes = [n_genes]
gradient_clip_val = [1]
swa = [0]
nlogo = [20, 18, 15, 10, 5, 2, 0]
seed = [0, 1, 2, 3, 4]

name_prefix = f"v3_inc_{graph}_{graph_kwargs_str}_{use_encoder}_{optimizer}_{batch_size}_{lyapunov_penalty}_{x_distribution}"
FILEDIR = f"/omics/groups/OE0540/internal/users/rohbeck/bicycle/plots/{name_prefix}_"

rule all:
    input:
        expand(FILEDIR + "{nlogo}_{seed}_{lr}_{n_genes}_{scale_l1}_{scale_kl}_{scale_spectral}_{scale_lyapunov}_{gradient_clip_val}_{swa}/last.png", nlogo=nlogo, seed=seed, lr=lr, n_genes=n_genes, scale_l1=scale_l1, scale_kl=scale_kl, scale_spectral=scale_spectral, scale_lyapunov=scale_lyapunov, gradient_clip_val=gradient_clip_val, swa=swa),

rule run_bicycle_training:
    output:
        pngs = FILEDIR + "{nlogo}_{seed}_{lr}_{n_genes}_{scale_l1}_{scale_kl}_{scale_spectral}_{scale_lyapunov}_{gradient_clip_val}_{swa}/last.png",
    resources:
        mem_mb = 6000,
        gpu = 1
    shell:
        "python notebooks/data/clicked_synthetic.py --nlogo {wildcards.nlogo} --seed {wildcards.seed} --lr {wildcards.lr} --n-genes {wildcards.n_genes} --scale-l1 {wildcards.scale_l1} --scale-kl {wildcards.scale_kl} --scale-spectral {wildcards.scale_spectral} --scale-lyapunov {wildcards.scale_lyapunov} --gradient-clip-val {wildcards.gradient_clip_val} --swa {wildcards.swa}"