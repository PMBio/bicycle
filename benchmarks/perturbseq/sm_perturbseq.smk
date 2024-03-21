from pathlib import Path
import os

user_dir = "/omics/groups/OE0540/internal/users/rohbeck/bicycle/"
PLOT_PATH = Path(os.path.join(user_dir, "plots"))
PLOT_PATH.mkdir(parents=True, exist_ok=True)

# LEARNING
batch_size = 1024
lyapunov_penalty = True
gradient_clip_val = [1]

seed = [0]
swa = [0, 500]
lr = [1e-3] 
scale_lyapunov = [0.1, 1, 10]
scale_l1 = [0.1, 1, 10]
scale_kl = [0.1, 1, 10]
scale_spectral = [0, 1]
use_inits = [False, True]

name_prefix = f"perturbseq_{batch_size}"
FILEDIR = str(PLOT_PATH) + f"/{name_prefix}_"

rule all:
    input:
        expand(FILEDIR + "{seed}_{lr}_{scale_l1}_{scale_kl}_{scale_spectral}_{scale_lyapunov}_{gradient_clip_val}_{swa}_{use_inits}/last.png", seed=seed, lr=lr, scale_l1=scale_l1, scale_kl=scale_kl, scale_spectral=scale_spectral, scale_lyapunov=scale_lyapunov, gradient_clip_val=gradient_clip_val, swa=swa, use_inits=use_inits),

rule run_bicycle_training:
    output:
        pngs = FILEDIR + "{seed}_{lr}_{scale_l1}_{scale_kl}_{scale_spectral}_{scale_lyapunov}_{gradient_clip_val}_{swa}_{use_inits}/last.png",
    resources:
        mem_mb = 2000000,
        gpu = 1
    shell:
        "python clicked_perturbseq.py --seed {wildcards.seed} --lr {wildcards.lr} --scale-l1 {wildcards.scale_l1} --scale-kl {wildcards.scale_kl} --scale-spectral {wildcards.scale_spectral} --scale-lyapunov {wildcards.scale_lyapunov} --gradient-clip-val {wildcards.gradient_clip_val} --swa {wildcards.swa} --use-inits {wildcards.use_inits}"

# l1 : 1  0.1
# kl : 1  0.1
# spectral 1 0
# scale_lyapunov: 0.1, 1, 10
# 0.001   1      0.1  1  10   1    0  False/last.ckpt
# 0.001   0.1    1    0  1    1    0   False/last.ckpt
# 0.001   0.1    0.1  0  0.1  1    0   True/last.ckpt
# 0.001   1      1    1  10   1    0    True/last.ckpt
# 0.001   0.1    0.1  0  0.1  1    500 False/last.ckpt
# 0.001   0.1    0.1  0  1    1    0 False/last.ckpt
# 0.001   0.1    0.1  0  0.1  1    0   False/last.ckpt
# 0.001   1      1    1  1    1    500   False/last.ckpt
# 0.001   0.1    0.1  0  0.1  1    500 True/last.ckpt
# 0.001   0.1    1    0  1    1    0   True/last.ckpt
# 0.001   1      1    1  10   1     0    False/last.ckpt

# l1 = 1 -> spectral: 1 
# l1 =1  -> lyap >= 1
# kl = 0.1 -> spectral: 0