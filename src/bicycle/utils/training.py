import numpy as np
import torch
import random
import os
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from typing import Optional, Tuple


# from: https://gist.github.com/yulkang/4a597bcc5e9ccf8c7291f8ecb776382d
def kron(a, b):
    """
    Kronecker product of matrices a and b with leading batch dimensions.
    Batch dimensions are broadcast. The number of them mush
    :type a: torch.Tensor
    :type b: torch.Tensor
    :rtype: torch.Tensor
    """
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    return res.reshape(siz0 + siz1)


def sylvester_direct(A, B, C):
    batch_size = A.shape[0]
    m = A.shape[-1]
    n = B.shape[-1]

    eyen = torch.eye(n, device=A.device)  # .to_sparse()
    eyem = torch.eye(m, device=A.device)  # .to_sparse()

    M = kron(eyen, A) + kron(B.transpose(1, 2), eyem)  # .to_sparse()
    L = C.transpose(1, 2).reshape((batch_size, -1))
    # x = M^-1 L
    x = torch.linalg.solve(M, L).reshape((batch_size, n, m)).transpose(1, 2)

    return x


def lyapunov_direct(A, C):
    B = A.transpose(1, 2).double()
    batch_size = A.shape[0]
    m = A.shape[-1]
    n = B.shape[-1]

    eyen = torch.eye(n, device=A.device)  # .to_sparse()
    eyem = torch.eye(m, device=A.device)  # .to_sparse()

    M = kron(eyen, A) + kron(B.transpose(1, 2), eyem)  # .to_sparse()
    L = C.transpose(1, 2).reshape((batch_size, -1))
    # x = M^-1 L
    x = torch.linalg.solve(M, L).reshape((batch_size, n, m)).transpose(1, 2)

    return x


def seed_everything(seed: int):
    """Seed everything for reproducibility"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Only for Apple chips
    torch.mps.manual_seed(seed)


class EarlyStopper(object):
    """Class to manage early stopping of model training."""

    # Adapted from https://gist.github.com/stefanonardo
    def __init__(self, mode="min", min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self.percentage = percentage
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if np.isinf(metrics):
            self.num_bad_epochs += 1
        elif self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if not percentage:
            if mode == "min":
                self.is_better = lambda a, best: a < best - min_delta
            if mode == "max":
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == "min":
                self.is_better = lambda a, best: a < best - (abs(best) * min_delta / 100)
            if mode == "max":
                self.is_better = lambda a, best: a > best + (abs(best) * min_delta / 100)


class EarlyStopperTorch(object):
    """Class to manage early stopping of model training."""

    # Adapted from https://gist.github.com/stefanonardo
    def __init__(self, mode="min", min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self.percentage = percentage
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if torch.isinf(metrics):
            self.num_bad_epochs += 1
        elif self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if not percentage:
            if mode == "min":
                self.is_better = lambda a, best: a < best - min_delta
            if mode == "max":
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == "min":
                self.is_better = lambda a, best: a < best - (abs(best) * min_delta / 100)
            if mode == "max":
                self.is_better = lambda a, best: a > best + (abs(best) * min_delta / 100)


class EarlyStopping_mod(EarlyStopping):
    """
    Class to manage early stopping of model training.
    Adapted from : https://github.com/Lightning-AI/pytorch-lightning/issues/12094#issuecomment-1825914097 
    """
    
    def __init__(self, threshold_mode='abs', **kwargs):
        super().__init__(**kwargs)
        self.threshold_mode = threshold_mode

    def _evaluate_stopping_criteria(self, current: torch.Tensor) -> Tuple[bool, Optional[str]]:
        should_stop = False
        reason = None
        #Catching case when self.best_score starts at Inf and the rel expression evaluates to NaN
        if self.threshold_mode == 'rel' and torch.isinf(self.best_score):
            eval_criteria = (self.best_score.to(current.device)
                                            + self.min_delta * torch.abs(self.best_score.to(current.device)))
            if torch.isnan(eval_criteria):
                #reset this val back to Inf to allow for proper adjustment of best_score 
                eval_criteria = torch.tensor(np.Inf) if self.mode == 'min' else -torch.tensor(np.Inf)
        else:
            eval_criteria = (self.best_score.to(current.device)
                                            + self.min_delta * torch.abs(self.best_score.to(current.device)))

        if self.check_finite and not torch.isfinite(current):
            should_stop = True
            reason = (
                f"Monitored metric {self.monitor} = {current} is not finite."
                f" Previous best value was {self.best_score:.3f}. Signaling Trainer to stop."
            )
        elif self.stopping_threshold is not None and self.monitor_op(current, self.stopping_threshold):
            should_stop = True
            reason = (
                "Stopping threshold reached:"
                f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.stopping_threshold}."
                " Signaling Trainer to stop."
            )
        elif self.divergence_threshold is not None and self.monitor_op(-current, -self.divergence_threshold):
            should_stop = True
            reason = (
                "Divergence threshold reached:"
                f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.divergence_threshold}."
                " Signaling Trainer to stop."
            )
        elif (self.threshold_mode == 'abs'
              and self.monitor_op(current - self.min_delta, self.best_score.to(current.device))):
            should_stop = False
            reason = self._improvement_message(current)
            self.best_score = current
            self.wait_count = 0
        elif (self.threshold_mode == 'rel'
              and self.monitor_op(current, eval_criteria)):
            should_stop = False
            reason = self._improvement_message(current)
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                should_stop = True
                reason = (
                    f"Monitored metric {self.monitor} did not improve in the last {self.wait_count} records."
                    f" Best score: {self.best_score:.3f}. Signaling Trainer to stop.\n"
                    # f"min_delat: {self.min_delta}\n"
                    # f"current: {current}\n"
                    # f"rel val: {self.best_score.to(current.device) - self.min_delta * torch.abs(self.best_score.to(current.device))}"
                )

        return should_stop, reason

    def _improvement_message(self, current: torch.Tensor) -> str:
        """Formats a log message that informs the user about an improvement in the monitored score."""
        if torch.isfinite(self.best_score):
            if self.threshold_mode == 'abs':
                msg = (
                    f"Metric {self.monitor} improved by {abs(self.best_score - current):.3f} >="
                    f" min_delta = {abs(self.min_delta)}. New best score: {current:.3f}"
                )
            else:  # self.threshold_mode == 'rel':
                msg = (
                    f"Metric {self.monitor} improved by {abs(self.best_score - current) / self.best_score:.3f} >="
                    f" min_delta = {abs(self.min_delta)}. New best score: {current:.3f}"
                ) 
        else:
            msg = f"Metric {self.monitor} improved. New best score: {current:.3f}"
        return msg