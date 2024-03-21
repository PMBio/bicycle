import numpy as np
import torch
import random
import os


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
