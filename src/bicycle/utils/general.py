def get_full_name(
    name_prefix,
    nlogo,
    seed,
    lr,
    n_genes,
    scale_l1,
    scale_kl,
    scale_spectral_loss,
    scale_lyapunov,
    gradient_clip_val,
    swa,
):
    return f"{name_prefix}_{nlogo}_{seed}_{lr}_{n_genes}_{scale_l1}_{scale_kl}_{scale_spectral_loss}_{scale_lyapunov}_{gradient_clip_val}_{swa}"
