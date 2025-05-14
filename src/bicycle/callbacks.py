from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint

from bicycle.utils.plotting import plot_training_results


class MyLoggerCallback(pl.Callback):
    def __init__(self, dirpath: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dirpath = dirpath

    def on_fit_end(self, trainer, pl_module):
        pd.DataFrame(trainer.logger.history).to_parquet(Path(self.dirpath) / "logger.parquet")
        # Create yaml with hyperparameters
        # TODO: Maybe directly overwrite hyperparameters of model itself
        training_stats = {
            "finished": True,
            "n_epochs": trainer.current_epoch,
            "n_steps": trainer.global_step,
            "early_stopping": trainer.model.early_stopping,
        }
        if trainer.model.early_stopping:
            training_stats = training_stats | {
                "early_stopping_min_delta": trainer.model.earlystopper.min_delta,
                "early_stopping_patience": trainer.model.earlystopper.patience,
                "early_stopping_p_mode": trainer.model.earlystopper.percentage,
            }
        else:
            training_stats = training_stats | {
                "early_stopping_min_delta": None,
                "early_stopping_patience": None,
                "early_stopping_p_mode": None,
            }
        # Save to yaml
        filepath = Path(self.dirpath) / "report.yaml"
        print(f"Saving training stats to {filepath}...")
        with open(filepath, "w") as outfile:
            yaml.dump(training_stats, outfile, default_flow_style=False)


# Subclass _should_skip_saving_checkpoint
class CustomModelCheckpoint(ModelCheckpoint):
    """Custom ModelCheckpoint to save after n epochs."""

    def __init__(self, start_after: int = 1000, **kwargs):
        super().__init__(**kwargs)
        self.start_after = start_after

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Save a checkpoint at the end of the validation stage."""
        if (
            not self._should_skip_saving_checkpoint(trainer)
            and not self._should_save_on_train_epoch_end(trainer)
            and (self.start_after <= trainer.current_epoch)
        ):
            monitor_candidates = self._monitor_candidates(trainer)
            if self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
                self._save_topk_checkpoint(trainer, monitor_candidates)
            self._save_last_checkpoint(trainer, monitor_candidates)


class GenerateCallback(pl.Callback):
    """
    Class for generating a callback at the end of each training epoch of the bicycle model.
    Subclasses pytorch_lightnings Callback function.

    Notes:
    - Calls the plot_training_results function from bicycle.utils.plotting
        after each epoch that's a multiple of `plot_epoch_callback`. 
    """
    def __init__(self, file_name_plot, plot_epoch_callback=10, true_beta=None, labels=None):
        """
        Initializes the GenerateCallback function for plotting training results during training.

        Args:
            file_name_plot (str): containing the file_name for the saved plots.
                Can be given with or without the suffix.
            plot_epoch_callback (int): specifying the callback interval.
            true_beta (Optional|torch.Tensor): containing the ground truth gene adjacency matrix.
            labels (Iterable): contains the gene labels.
        """
        super().__init__()
        self.plot_epoch_callback = plot_epoch_callback
        self.true_beta = true_beta
        self.file_name_plot = file_name_plot
        self.labels = labels

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch % self.plot_epoch_callback == 0) & (trainer.current_epoch > 0):
            # Reconstruct images
            with torch.no_grad():
                pl_module.eval()
                if pl_module.mask is None:
                    if pl_module.n_factors == 0:
                        estimated_beta = pl_module.beta.detach().cpu().numpy()
                    else:
                        estimated_beta = torch.einsum(
                            "ij,jk->ik", pl_module.gene2factor, pl_module.factor2gene
                        )
                        beta_diag = torch.diagonal(estimated_beta, offset=0, dim1=-2, dim2=-1)
                        beta_diag[:] = 0
                        estimated_beta = estimated_beta.detach().cpu().numpy()
                else:
                    estimated_beta = torch.zeros(
                        (pl_module.n_genes, pl_module.n_genes), device=pl_module.device
                    )
                    estimated_beta[pl_module.beta_idx[0], pl_module.beta_idx[1]] = pl_module.beta_val
                    estimated_beta = estimated_beta.detach().cpu().numpy()
                pl_module.train()

            plot_training_results(
                trainer,
                pl_module,
                estimated_beta,
                self.true_beta,
                pl_module.scale_l1,
                pl_module.scale_kl,
                pl_module.scale_spectral,
                pl_module.scale_lyapunov,
                self.file_name_plot,
                callback=True,
                labels=self.labels,
            )
