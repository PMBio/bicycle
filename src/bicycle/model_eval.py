import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch import Tensor, nn
from torch.distributions.kl import kl_divergence

from bicycle.utils.training import EarlyStopperTorch, lyapunov_direct


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
        torch.nn.init.xavier_normal_(m.weight, gain=0.1)
        if m.bias is not None:
            # m.bias.data.fill_(0.01)
            m.bias.data.zero_()


class Encoder(nn.Module):
    def __init__(self, x_dim: int, z_dim: int, n_cond: int, act_fn: object = nn.GELU):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(x_dim + n_cond, 2 * z_dim),
            act_fn(),
            nn.Dropout(p=0.05),
            nn.Linear(2 * z_dim, 2 * z_dim),
            act_fn(),
            nn.Dropout(p=0.05),
            nn.Linear(2 * z_dim, 2 * z_dim),
        )

        self.net.apply(init_weights)

    def forward(self, x):
        x = self.net(x)
        mu = x[:, : self.z_dim]
        variance = torch.nn.Softplus()(x[:, self.z_dim :]) + 1e-6
        return mu, variance


class BICYCLE_EVAL(pl.LightningModule):
    def __init__(
        self,
        lr,
        gt_interv,
        n_genes,
        n_samples,
        lyapunov_penalty=True,
        perfect_interventions=True,
        rank_w_cov_factor=1,
        optimizer="adam",
        device="cuda",
        scale_l1=1.0,
        scale_spectral=1.0,
        scale_lyapunov=1.0,
        scale_kl=1.0,
        early_stopping: bool = True,
        early_stopping_min_delta: float = 0.5,
        early_stopping_patience: int = 100,
        early_stopping_p_mode: bool = True,
        x_distribution: str = None,
        init_tensors: dict = None,
        mask: Tensor = None,
        use_encoder: bool = False,
        gt_beta: Tensor = None,
        train_gene_ko: list = None,
        test_gene_ko: list = None,
        use_latents: bool = True,
        pred_gene: int = -1,
    ):
        super().__init__()

        # FIXME / TODO
        # should we decrease self.sigma_min?

        self.save_hyperparameters()

        self.init_tensors = init_tensors
        self.lr = lr
        self.gt_interv = gt_interv
        self.n_genes = n_genes
        self.early_stopping = early_stopping
        self.lyapunov_penalty = lyapunov_penalty
        self.perfect_interventions = perfect_interventions
        self.rank_w_cov_factor = rank_w_cov_factor
        self.optimizer = optimizer
        self._device = device
        self.contexts = torch.arange(gt_interv.shape[1])
        self.n_contexts = gt_interv.shape[1]
        self.n_samples = n_samples
        self.mask = mask
        self.use_encoder = use_encoder
        self.use_latents = use_latents
        self.pred_gene = pred_gene

        if self.use_latents:
            if self.use_encoder:
                self.n_conditions = torch.sum(gt_interv.sum(axis=1) > 0).item()
                self.encoder = Encoder(x_dim=self.n_genes, z_dim=self.n_genes, n_cond=self.n_conditions)

                self.gt_nonzeros = self.gt_interv[~torch.all(self.gt_interv == 0, axis=1)]
            else:
                # Cell and gene specific latent expression values
                self.z_loc = torch.nn.Parameter(torch.zeros((self.n_samples, n_genes)))
                self.z_scale = torch.nn.Parameter(torch.zeros((self.n_samples, n_genes)))
                self.sigma_min = 1e-3

        self.scale_l1 = scale_l1
        self.scale_spectral = scale_spectral
        self.scale_lyapunov = scale_lyapunov
        self.scale_kl = scale_kl
        self._normalisation_computed = False

        if x_distribution is not None:
            if x_distribution not in ["Poisson", "Normal"]:
                raise ValueError(
                    f"Unknown distribution {x_distribution}. " "Only Poisson, Normal is supported."
                )
        self.x_distribution = x_distribution

        if early_stopping:
            self.earlystopper = EarlyStopperTorch(
                mode="min",
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
                percentage=early_stopping_p_mode,
            )

        self.validation_step_outputs = []

        self.pos = nn.Softplus()

        # Variables describing *UNPERTURBED* mechanisms
        if self.mask is None:
            if "beta" in self.init_tensors:
                self.beta = torch.nn.Parameter(self.init_tensors["beta"])
            else:
                self.beta = torch.nn.Parameter(0.001 * torch.randn((n_genes, n_genes)))
        else:
            with torch.no_grad():
                self.n_entries = (self.mask > 0.5).sum()
                # print(f"No. entries in mask for beta: {self.n_entries}")
                self.beta_idx = torch.where(self.mask > 0.5)
            self.beta_val = torch.nn.Parameter(0.001 * torch.randn((self.n_entries,)))

        if not self.perfect_interventions:
            if self.mask is None:
                self.beta_p = torch.nn.Parameter(0.1 * torch.randn((n_genes, n_genes)))
            else:
                self.beta_p_val = torch.nn.Parameter(0.1 * torch.randn((self.n_entries)))
        else:
            if self.mask is None:
                self.beta_p = torch.nn.Parameter(torch.zeros((n_genes, n_genes)))
                self.beta_p.requires_grad = False
            else:
                self.beta_p_val = torch.nn.Parameter(torch.zeros((self.n_entries)))
                self.beta_p_val.requires_grad = False

        # Must be positive
        self.alpha = torch.nn.Parameter(0.001 * torch.exp(torch.randn((n_genes,))))
        self.alpha_p = torch.nn.Parameter(0.001 * torch.exp(torch.randn((n_genes,))))
        self.sigma = torch.nn.Parameter(1.0 * torch.exp(torch.randn((n_genes,))))
        self.sigma_p = torch.nn.Parameter(1.0 * torch.exp(torch.randn((n_genes,))))

        self.T = torch.tensor(1.0)  # torch.nn.Parameter(torch.tensor(1.0))

        if self.lyapunov_penalty:
            # covariance_matrix = cov_factor @ cov_factor.T + cov_diag
            # The computation for determinant and inverse of covariance matrix is
            # avoided when cov_factor.shape[1] << cov_factor.shape[0], see pytorch
            # docs
            self.w_cov_diag = torch.nn.Parameter(
                torch.exp(
                    0.1
                    * torch.rand(
                        (
                            self.n_contexts,
                            n_genes,
                        )
                    )
                )
            )
            self.w_cov_factor = torch.nn.Parameter(
                0.1 * torch.randn((self.n_contexts, n_genes, rank_w_cov_factor))
            )

        if init_tensors is not None:
            with torch.no_grad():
                print("Initializing parameters from data")
                if "alpha" in self.init_tensors:
                    self.alpha.data = self.init_tensors["alpha"]
                if "w_cov_factor" in self.init_tensors:
                    self.w_cov_factor.data = self.init_tensors["w_cov_factor"]
                if "w_cov_diag" in self.init_tensors:
                    self.w_cov_diag.data = self.init_tensors["w_cov_diag"]

        if gt_beta is not None:
            self.gt_beta = gt_beta
        if train_gene_ko is not None:
            self.train_gene_ko = train_gene_ko
        if test_gene_ko is not None:
            self.test_gene_ko = test_gene_ko

    def configure_optimizers(self):
        if self.optimizer == "adam":
            return optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == "rmsprop":
            return optim.RMSprop(self.parameters(), lr=self.lr)
        elif self.optimizer == "adamlrs":
            optmsr = optim.Adam(self.parameters(), lr=self.lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optmsr, mode="min", factor=0.2, patience=10, min_lr=5e-5
            )
            # FIXME: CHECK IF THIS IS THE CORRECT LOSS
            return {
                "optimizer": optmsr,
                "lr_scheduler": scheduler,
                "monitor": "train_loss",
            }

    def get_updated_states(self):
        iv_a = (1 - self.gt_interv).T

        if self.mask is not None:
            self.beta = torch.zeros((self.n_genes, self.n_genes), device=self.device)
            self.beta[self.beta_idx[0], self.beta_idx[1]] = self.beta_val
            self.beta_p = torch.zeros((self.n_genes, self.n_genes), device=self.device)
            self.beta_p[self.beta_idx[0], self.beta_idx[1]] = self.beta_p_val

        iv_a = iv_a.to(self.device)

        betas = iv_a[:, None, :] * self.beta.to(self.device) + (1 - iv_a)[:, None, :] * self.beta_p.to(
            self.device
        )
        alphas = (
            iv_a * self.pos(self.alpha.to(self.device))[None, :]
            + (1 - iv_a) * self.pos(self.alpha_p.to(self.device))[None, :]
        )

        B = torch.eye(self.n_genes, device=self.device)[None, :, :] - (
            1.0 - torch.eye(self.n_genes, device=self.device)
        )[None, :, :] * betas.transpose(1, 2)

        sigmas = iv_a[:, None, :] * torch.diag(self.pos(self.sigma)) + (1 - iv_a)[:, None, :] * torch.diag(
            self.pos(self.sigma_p)
        )

        return alphas, betas, B, sigmas

    def lyapunov_lhs(self, B):
        mat = B @ (
            torch.diag_embed(self.pos(self.w_cov_diag))
            + self.w_cov_factor @ self.w_cov_factor.transpose(1, 2)
        )
        return mat + mat.transpose(1, 2)

    def lyapunov_rhs(self, sigmas):
        return torch.bmm(sigmas, sigmas.transpose(1, 2))

    # def compute_normalisations(self, log_likelihood, z_kl, l1_loss, spectral_loss, loss_lyapunov):
    #     # Normalise losses
    #     if self.normalise:
    #         if (self.current_epoch == 0) & (self._normalise_llh is None) & self.training:
    #             # save normalization constants
    #             self._normalise_llh = log_likelihood.detach().abs()
    #             self._normalise_kl = z_kl.detach().abs()
    #             self._normalise_l1 = l1_loss.detach().abs()
    #             if self.scale_spectral > 0:
    #                 self._normalise_spectral_loss = spectral_loss.detach().abs()
    #             if self.scale_lyapunov > 0:
    #                 self._normalise_lyapunov = loss_lyapunov.detach().abs()
    #     else:
    #         self._normalise_llh = 1.0
    #         self._normalise_kl = 1.0
    #         self._normalise_l1 = 1.0
    #         self._normalise_spectral_loss = 1.0
    #         self._normalise_lyapunov = 1.0

    #     self._normalisation_computed = True

    def scale_losses(self, z_kl, l1_loss, loss_spectral=None, loss_lyapunov=None):
        l1_loss = self.scale_l1 * l1_loss
        z_kl = self.scale_kl * z_kl

        if loss_spectral:
            loss_spectral = self.scale_spectral * loss_spectral
        if loss_lyapunov:
            loss_lyapunov = self.scale_lyapunov * loss_lyapunov

        return z_kl, l1_loss, loss_spectral, loss_lyapunov

    def split_samples(self, samples, sim_regime, sample_idx, data_category):
        # Split all rows according to is_valid_data
        samples_train = samples[data_category == 0]
        sim_regime_train = sim_regime[data_category == 0]
        sample_idx_train = sample_idx[data_category == 0]
        samples_valid = samples[data_category == 1]
        sim_regime_valid = sim_regime[data_category == 1]
        sample_idx_valid = sample_idx[data_category == 1]
        samples_test = samples[data_category == 2]
        sim_regime_test = sim_regime[data_category == 2]
        sample_idx_test = sample_idx[data_category == 2]

        return (
            samples_train,
            sim_regime_train,
            sample_idx_train,
            samples_valid,
            sim_regime_valid,
            sample_idx_valid,
            samples_test,
            sim_regime_test,
            sample_idx_test,
        )

    def get_x_bar(self, B, alphas, sim_regime):
        # Broadcast arrays to batch_shape
        B_broadcasted = B[sim_regime]
        alphas_broadcasted = alphas[sim_regime]
        x_bar = torch.bmm(torch.linalg.inv(B_broadcasted), alphas_broadcasted[:, :, None]).squeeze()
        return x_bar

    def get_mvn_normal(self, B, alphas, sim_regime, sigmas):
        x_bar = self.get_x_bar(B, alphas, sim_regime)

        if self.lyapunov_penalty:
            return torch.distributions.LowRankMultivariateNormal(
                x_bar,
                self.w_cov_factor[sim_regime],
                self.pos(self.w_cov_diag)[sim_regime],
            )
        else:
            omegas = lyapunov_direct(
                B.double(),
                torch.bmm(sigmas, sigmas.transpose(1, 2)).double(),
            ).float()
            return torch.distributions.MultivariateNormal(x_bar, covariance_matrix=omegas[sim_regime])

    def _get_posterior_dist(self, sample_idx, samples, sim_regime):
        if self.use_encoder:
            gt_nonzeros = self.gt_nonzeros.to(self.device)
            ohes = gt_nonzeros[:, sim_regime].T
            m = torch.cat([samples, ohes], 1)
            mu, variance = self.encoder(m)
            return torch.distributions.MultivariateNormal(mu, torch.diag_embed(variance))
        else:
            z_locs = self.z_loc[sample_idx]
            z_scales = self.pos(self.z_scale[sample_idx])
            return torch.distributions.MultivariateNormal(z_locs, torch.diag_embed(z_scales + self.sigma_min))

    def compute_kl_divergence_loss(self, mvn_dist, sample_idx, samples, sim_regime):
        """Compute KL Divergence between prior and posterior distribution"""
        z_mvn = self._get_posterior_dist(sample_idx, samples, sim_regime)
        z_kl = kl_divergence(z_mvn, mvn_dist).mean()
        return z_kl

    def compute_nll_loss(self, samples, sample_idx, sim_regime, mvn=None):

        pred_mask = torch.ones(samples.shape, device=samples.device)
        if self.pred_gene >= 0:
            pred_mask[:, self.pred_gene] = 0.0

        """Compute NLL Loss."""
        if self.use_latents:
            if self.x_distribution == "Poisson":
                zs = self._get_posterior_dist(sample_idx, samples, sim_regime).rsample()

                library_size = samples.sum(axis=1).reshape(-1, 1)
                # FIXME: Figure out why softplus does not work, i.e. self.pos
                ps = torch.softmax(zs / self.T, dim=-1)

                P = torch.distributions.poisson.Poisson(rate=library_size * ps)
            elif self.x_distribution == "Normal":
                z_locs = self.z_loc[sample_idx]
                z_scales = self.pos(self.z_scale[sample_idx])
                P = torch.distributions.normal.Normal(loc=z_locs, scale=z_scales)

            return -1 * (pred_mask * P.log_prob(samples)).mean()

        else:
            return -1 * (pred_mask * mvn.log_prob(samples)).mean()

    def training_step(self, batch, batch_idx):
        kwargs = {"on_step": False, "on_epoch": True}
        prefix = "train" if self.training else "valid"

        samples, sim_regime, sample_idx, data_category = batch

        # Split all rows according to data_category
        (
            samples_train,
            sim_regime_train,
            sample_idx_train,
            samples_valid,
            sim_regime_valid,
            sample_idx_valid,
            samples_test,
            sim_regime_test,
            sample_idx_test,
        ) = self.split_samples(samples, sim_regime, sample_idx, data_category)

        alphas, _, B, sigmas = self.get_updated_states()

        # We only optimize LATENTS in the EVAL CLASS case we face valid or test data, we have to detach some parameters that must not get an update
        B_detached = B.detach()
        alphas_detached = alphas.detach()
        sigmas_detached = sigmas.detach()

        # KL Divergence & NLL Loss
        neg_log_likelihood = 0
        z_kl = 0
        if self.use_latents:
            # Test Data
            if len(samples_test) > 0:
                # Block every gradient coming from the MVN, only optimize LATENTS!!!!
                mvn_test = self.get_mvn_normal(
                    B_detached,
                    alphas_detached,
                    sim_regime_test,
                    sigmas_detached,
                )
                z_kl_test = self.compute_kl_divergence_loss(
                    mvn_test, sample_idx_test, samples_test, sim_regime_test
                )
                neg_log_likelihood_test = self.compute_nll_loss(
                    samples_test, sample_idx_test, sim_regime_test
                )
                self.log(f"{prefix}_kl_test", z_kl_test, **kwargs)
                self.log(f"{prefix}_nll_test", neg_log_likelihood_test, **kwargs)
                neg_log_likelihood += neg_log_likelihood_test
                z_kl += z_kl_test

        #
        # Combine Losses
        #

        # Rescale combined KL divergence (train, valid and test) by number of genes
        z_kl = z_kl / self.n_genes

        loss = neg_log_likelihood + z_kl

        self.log(f"{prefix}_loss", loss, **kwargs)

        return loss

    # def validation_step(self, batch, batch_idx):
    #    self.training_step(batch, batch_idx)

    def predict_step(self, batch, dataloader_idx=0):
        samples, sim_regime, sample_idx, _ = batch

        alphas, _, B, sigmas = self.get_updated_states()

        # KL Divergence & NLL Loss
        mvn_test = self.get_mvn_normal(
            B,
            alphas,
            sim_regime,
            sigmas,
        )
        if self.use_latents:
            z_kl = self.compute_kl_divergence_loss(mvn_test, sample_idx, samples, sim_regime)
        else:
            z_kl = 0

        neg_log_likelihood = self.compute_nll_loss(samples, sample_idx, sim_regime, mvn_test)

        # Rescale combined KL divergence
        z_kl = z_kl / self.n_genes

        loss = neg_log_likelihood + z_kl

        return loss

    def forward(self):
        raise NotImplementedError()

    """def on_validation_epoch_end(self):
        if self.early_stopping:
            avg_loss = torch.stack(self.validation_step_outputs).mean()
            self.log("avg_valid_loss", avg_loss)

            if self.earlystopper.step(avg_loss):
                print(f"Earlystopping due to convergence at step {self.current_epoch}")
                self.trainer.should_stop = True

            self.validation_step_outputs.clear()"""
