from argparse import Namespace

import pandas as pd
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only


class _History(dict):
    def __missing__(self, key):
        df = pd.Series(name=key)
        df.index.name = "step"
        self[key] = df
        return df


class DictLogger(Logger):
    def __init__(self, version=None):
        self._version = version
        self.experiment = None
        self.hyperparams = {}
        self.history = _History()

    @property
    def name(self):
        return "DictLogger"

    @property
    def version(self):
        return "0" if self._version is None else self._version

    @rank_zero_only
    def log_hyperparams(self, params: Namespace):
        self.hyperparams = vars(params)

    @rank_zero_only
    def log_metrics(self, metrics: dict[str, float], step: int):
        for k, v in metrics.items():
            self.history[k].loc[step] = v