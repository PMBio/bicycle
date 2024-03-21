import torch 
import numpy as np 

def get_adjacency_from_func(f, threshold=0.5, full_input=False):

    W_hat = np.zeros((f.n_nodes, f.n_nodes))
    for i in range(f.n_nodes):
        W = f.functions[i][0][0].weight.cpu().detach().numpy()
        W = np.abs(W).sum(axis=0)
        ind_exc_i = np.setdiff1d(np.arange(f.n_nodes), [i])
        if full_input:
            ind_exc_i = np.setdiff1d(np.arange(f.n_nodes), [None])
        W_hat[ind_exc_i, i] = W

    return W_hat, W_hat >= threshold

def get_adj_from_single_func(f, device):
    st_basis_vec = torch.eye(f.n_nodes, device=device)
    sensitivity = f(st_basis_vec).detach().cpu().numpy()
    neg_sensitivity = f(-1*st_basis_vec).detach().cpu().numpy()

    fin_sensitivity = np.abs(sensitivity) + np.abs(neg_sensitivity)
    return fin_sensitivity

def compute_shd(W_gt, W_est):
    # both W_gt & W_est should be binary matrices
    W_gt = W_gt * 1.0
    W_est = W_est * 1.0
    
    corr_edges = (W_gt == W_est) * W_gt # All the correctly identified edges
    
    W_gt -= corr_edges
    W_est -= corr_edges
    
    R = (W_est.T == W_gt) * W_gt # Reverse edges
    
    W_gt -= R
    W_est -= R.T
    
    E = W_est > W_gt # Extra edges
    M = W_est < W_gt # Missing edges

    return R.sum() + E.sum() + M.sum(), (R.sum(), E.sum(), M.sum())

def error_metrics(W_gt, W_est):
    # Computes the accuracy, precision and recall between the estimated W and true W. 
    # Parameters:
    # 1) W_gt - Ground truth adjacency matrix.
    # 2) W_est - Estimated adjacency matrix.
    # Both the matrices are assumed to binary. 

    W_gt = W_gt * 1.0
    W_est = W_est * 1.0

    acc = (W_gt == W_est).sum()
    tp_matrix = W_gt * W_est
    fp_matrix = (W_est - tp_matrix).sum()
    
    return tp_matrix.sum(), acc - tp_matrix.sum(), fp_matrix.sum(), (W_gt - tp_matrix).sum() 

def compute_auroc(W_gt, W_est, n_points=50):
    # Computes the Area Under Precision Recall Curve (AUPR)
    # W_gt - binary matrix (ground truth)
    # W_est - Positive entry matrix (estimated)
    # n_points - (int) - number of points to compute precision and recall

    max_threshold = W_est.max()
    threshold_list = np.linspace(0, max_threshold, n_points)
    
    tpr_list, fpr_list = list(), list()
    for threshold in threshold_list:
        tp, tn, fp, fn = error_metrics(W_gt, W_est >= threshold)
        tp_rate = tp / (tp + fn)
        fp_rate = fp / (tn + fp)
        tpr_list.append(tp_rate)
        fpr_list.append(fp_rate)
    
    area = np.trapz(tpr_list[::-1], fpr_list[::-1])
    return tpr_list, fpr_list, area

def compute_auprc(W_gt, W_est, n_points=50):

    max_threshold = W_est.max()
    threshold_list = np.linspace(0, max_threshold, n_points)
    rec_list, pre_list = list(), list()
    for threshold in threshold_list:
        tp, tn, fp, fn = error_metrics(W_gt, W_est >= threshold)
        rec = tp / (tp + fn)
        pre = tp / (tp + fp)
        rec_list.append(rec)
        pre_list.append(pre)

    rec_list.append(0)
    pre_list.append(1.0)

    area = np.trapz(pre_list[::-1], rec_list[::-1])
    baseline = W_gt.sum() / (W_gt.shape[0] * W_gt.shape[1])

    return baseline, area

def samesign(a, b):
    return a * b > 0

def is_acyclic(adjacency):
    """
    Return true if adjacency is a acyclic
    :param np.ndarray adjacency: adjacency matrix
    """
    prod = np.eye(adjacency.shape[0], dtype=adjacency.dtype)
    for _ in range(1, adjacency.shape[0] + 1):
        prod = adjacency @ prod
        if np.trace(prod) != 0:
            return False
    return True

def bisect(func, low, high, T=20):
    "Find root of continuous function where f(low) and f(high) have opposite signs"
    flow = func(low)
    fhigh = func(high)
    assert not samesign(flow, fhigh)
    for i in range(T):
        midpoint = (low + high) / 2.0
        fmid = func(midpoint)
        if samesign(flow, fmid):
            low = midpoint
            flow = fmid
        else:
            high = midpoint
            fhigh = fmid
    # after all those iterations, low has one sign, and high another one. midpoint is unknown
    return high