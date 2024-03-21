import numpy as np
import torch


def error_metrics(W_gt, W_est):
    # Both the matrices are assumed to binary.
    W_gt = W_gt * 1.0
    W_est = W_est * 1.0

    acc = (W_gt == W_est).sum()
    tp_matrix = W_gt * W_est
    fp_matrix = (W_est - tp_matrix).sum()

    return tp_matrix.sum(), acc - tp_matrix.sum(), fp_matrix.sum(), (W_gt - tp_matrix).sum()


def compute_auprc(W_gt, W_est, n_points=50):
    # In contrast to the AUPRC from nodags, we compute the threshold below and don't require
    # it as an input.
    max_threshold = W_est.max()
    threshold_list = np.linspace(0, max_threshold, n_points)
    rec_list, pre_list = list(), list()
    for threshold in threshold_list:
        tp, tn, fp, fn = error_metrics(W_gt, torch.abs(W_est) >= threshold)
        rec = tp / (tp + fn)
        pre = tp / (tp + fp)
        rec_list.append(rec)
        pre_list.append(pre)

    rec_list.append(0)
    pre_list.append(1.0)

    area = np.trapz(pre_list[::-1], rec_list[::-1])
    # baseline = W_gt.sum() / (W_gt.shape[0] * W_gt.shape[1])

    return area


def compute_shd(W_gt, W_est, threshold=0):
    # both W_gt & W_est should be binary matrices
    W_est = torch.abs(W_est) >= threshold

    W_gt = W_gt * 1.0
    W_est = W_est * 1.0

    corr_edges = (W_gt == W_est) * W_gt  # All the correctly identified edges

    W_gt -= corr_edges
    W_est -= corr_edges

    R = (W_est.T == W_gt) * W_gt  # Reverse edges

    W_gt -= R
    W_est -= R.T

    E = W_est > W_gt  # Extra edges
    M = W_est < W_gt  # Missing edges

    return R.sum() + E.sum() + M.sum(), (R.sum(), E.sum(), M.sum())
