from pathlib import Path
import os

MODEL_PATH = Path("/omics/groups/OE0540/internal/users/rohbeck/bicycle/models/")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

graph_type = "erdos-renyi"
edge_assignment = "random-uniform"
graph_kwargs = {
    "abs_weight_low": 0.25,
    "abs_weight_high": 0.95,
    "p_success": 0.5,
    "expected_density": 2,
    "noise_scale": 0.5,
}
batch_size = 1024
early_stopping_patience = 500
early_stopping_min_delta = 0.01
x_distribution = "Normal"
lyapunov_penalty = False
early_stopping = True

# Parameters
lr = [1e-2, 5e-3, 1e-3, 5e-4]
scale_lyapunov = [0]
n_genes = [10]
swa = [0, 250]
validation_size = [0.2]
seed = [0, 1, 2]
scale_l1 = [0.1, 1, 10]
scale_kl = [0.1, 1]
scale_spectral = [0, 1]
n_samples_control = [0, 500]
n_samples_per_perturbation = [250]
sem = ["linear-ou", "linear"]
intervention_scale = [1.0]
use_latents = [False]
nlogo = [9, 8, 6, 4, 2, 0] # Include 10 if n_samples_control != 0
rank_w_cov_factor = [10]

graph_kwargs_str = graph_kwargs["noise_scale"]

name_prefix = f"v1_{graph_type}_{graph_kwargs_str}_{early_stopping}_{early_stopping_patience}_{early_stopping_min_delta}_{x_distribution}"
filedir_incl_prefix = str(MODEL_PATH) + f"/{name_prefix}_"

rule all:
    input:
        expand(filedir_incl_prefix + "{nlogo}_{seed}_{lr}_{n_genes}_{scale_l1}_{scale_kl}_{scale_spectral}_{scale_lyapunov}_{swa}_{n_samples_control}_{n_samples_per_perturbation}_{validation_size}_{sem}_{use_latents}_{intervention_scale}_{rank_w_cov_factor}/report.yaml", nlogo=nlogo, seed=seed, lr=lr, n_genes=n_genes, scale_l1=scale_l1, scale_kl=scale_kl, scale_spectral=scale_spectral, scale_lyapunov=scale_lyapunov, swa=swa, n_samples_control=n_samples_control, n_samples_per_perturbation=n_samples_per_perturbation, validation_size=validation_size, sem=sem, use_latents=use_latents, intervention_scale=intervention_scale, rank_w_cov_factor=rank_w_cov_factor),

rule run_bicycle_training:
    output:
        pngs = filedir_incl_prefix + "{nlogo}_{seed}_{lr}_{n_genes}_{scale_l1}_{scale_kl}_{scale_spectral}_{scale_lyapunov}_{swa}_{n_samples_control}_{n_samples_per_perturbation}_{validation_size}_{sem}_{use_latents}_{intervention_scale}_{rank_w_cov_factor}/report.yaml",
    resources:
        mem_mb = 4000,
        gpu = 1
    shell:
        "python clicked_randnodags.py --nlogo {wildcards.nlogo} --seed {wildcards.seed} --lr {wildcards.lr} --n-genes {wildcards.n_genes} --scale-l1 {wildcards.scale_l1} --scale-kl {wildcards.scale_kl} --scale-spectral {wildcards.scale_spectral} --scale-lyapunov {wildcards.scale_lyapunov} --swa {wildcards.swa} --n-samples-control {wildcards.n_samples_control} --n-samples-per-perturbation {wildcards.n_samples_per_perturbation} --validation-size {wildcards.validation_size} --sem {wildcards.sem} --use-latents {wildcards.use_latents} --intervention-scale {wildcards.intervention_scale} --rank-w-cov-factor {wildcards.rank_w_cov_factor}"
