import numpy as np
import math
from bicycle.nodags_files.nodags_utils.utils import compute_auprc

# TODO make standard_normal_logprob adaptive to nonuniform variance


def standard_normal_logprob(z, noise_scale=0.5):
    logZ = -0.5 * np.log(2 * math.pi * noise_scale**2)
    return logZ - z**2 / (2 * noise_scale**2)


def replace_submatrix(mat, ind1, ind2, mat_replace):
    for i, index in enumerate(ind1):
        mat[index, ind2] = mat_replace[i, :]
    return mat


def get_gt_covariance(B, n_nodes, intervention_set, int_scale=1.0, noise_scale=0.5):
    # B is the transpose of the weights matrix
    Cov_x = np.zeros((n_nodes, n_nodes))

    observed_set = np.setdiff1d(np.arange(n_nodes), intervention_set)

    mat_t = np.linalg.inv(np.eye(len(observed_set)) - B[observed_set, :][:, observed_set])
    cross_weights = B[observed_set, :][:, intervention_set]
    T = mat_t @ cross_weights
    C_obs = (
        mat_t @ (cross_weights @ cross_weights.T + (noise_scale**2) * np.eye(len(observed_set))) @ mat_t.T
    )

    Cov_x = replace_submatrix(
        Cov_x, intervention_set, intervention_set, (int_scale**2) * np.eye(len(intervention_set))
    )
    Cov_x = replace_submatrix(Cov_x, observed_set, intervention_set, T)
    Cov_x = replace_submatrix(Cov_x, intervention_set, observed_set, T.T)
    Cov_x = replace_submatrix(Cov_x, observed_set, observed_set, C_obs)

    return Cov_x


def get_coefficients(cov, i, u, intervention_set, observed_set):
    coefs = np.zeros(len(intervention_set) + len(observed_set) - 1)

    get_index = lambda x: x if x < u else x - 1
    for node in observed_set:
        if node != u:
            coefs[get_index(node)] = cov[i, node]

    coefs[get_index(i)] = 1
    return coefs


def parse_experiment(
    dataset,
    intervention_set,
    T,
    t,
    curr_row=0,
    use_ground_truth_cov=False,
    B=None,
    int_scale=1.0,
    noise_scale=0.5,
):
    n_nodes = dataset.shape[1]
    observed_set = np.setdiff1d(np.arange(n_nodes), intervention_set)

    # step 1 - Get the covariance matrix
    if use_ground_truth_cov:
        Cov_x = get_gt_covariance(B, n_nodes, intervention_set, int_scale, noise_scale)
    else:
        dataset_cent = dataset - dataset.mean(axis=0)
        Cov_x = (1 / dataset.shape[0]) * dataset_cent.T @ dataset_cent

    st_row = curr_row
    # step 3 - construct T and t
    for int_node in intervention_set:
        for obs_node in observed_set:
            coefs = get_coefficients(Cov_x, int_node, obs_node, intervention_set, observed_set)
            st_col = obs_node * (n_nodes - 1)
            T[st_row, st_col : st_col + n_nodes - 1] = coefs
            t[st_row] = Cov_x[int_node, obs_node]
            st_row += 1

    return T, t, st_row


def compute_n_rows(n_nodes, intervention_sets):
    n_rows = 0
    for intervention_set in intervention_sets:
        n_rows += len(intervention_set) * (n_nodes - len(intervention_set))

    return n_rows


def predict_adj_llc(
    datasets, intervention_sets, use_ground_truth_cov=False, B=None, int_scale=1.0, noise_scale=0.5
):
    n_nodes = datasets[0].shape[1]
    n_rows = compute_n_rows(n_nodes, intervention_sets)
    n_cols = n_nodes * (n_nodes - 1)

    T = np.zeros((n_rows, n_cols))
    t = np.zeros((n_rows, 1))
    st_row = 0

    lat_var = np.zeros(n_nodes)
    lat_count = np.zeros(n_nodes)

    print("Parsing experiments")
    i = 0
    for dataset, intervention_set in zip(datasets, intervention_sets):
        # if intervention_set[0] != None:
        # print("parsing experiment: {}".format(i))
        i += 1
        T, t, st_row = parse_experiment(
            dataset, intervention_set, T, t, st_row, use_ground_truth_cov, B, int_scale, noise_scale
        )

    print("Estimating the adjacency matrix")
    b_est = np.linalg.pinv(T) @ t
    B_est = np.zeros((n_nodes, n_nodes))
    for n in range(n_nodes):
        exc_n_set = np.setdiff1d(np.arange(n_nodes), n)
        B_est[exc_n_set, n] = b_est[n * (n_nodes - 1) : (n + 1) * (n_nodes - 1)].squeeze()

    print("Estimating latent variances")
    for dataset, intervention_set in zip(datasets, intervention_sets):
        observed_set = np.setdiff1d(np.arange(n_nodes), intervention_set)
        U = np.zeros((n_nodes, n_nodes))
        U[observed_set, observed_set] = 1
        dataset_cent = dataset - dataset.mean(axis=0)
        Cov_x = (1 / dataset.shape[0]) * dataset_cent.T @ dataset_cent
        for obs_node in observed_set:
            lat_obs_cov = (np.eye(n_nodes) - U @ B_est.T) @ Cov_x @ (np.eye(n_nodes) - B_est @ U)
            lat_var[obs_node] += lat_obs_cov[obs_node, obs_node]
            lat_count[obs_node] += 1

    lat_var /= lat_count

    return T, t, B_est, lat_var


class LLCClassWrapper:
    def __init__(self, use_ground_truth=False, B=None, int_scale=1.0, noise_scale=0.5, thresh_val=1e-2):
        self.use_ground_truth = use_ground_truth
        self.B = B
        self.int_scale = int_scale
        self.noise_scale = noise_scale
        self.thresh_val = thresh_val

    def train(self, datasets, intervention_sets, return_weights=False, batch_size=64):
        _, _, self.B_est, self.noise_scale = predict_adj_llc(
            datasets, intervention_sets, self.use_ground_truth, self.B, self.int_scale, self.noise_scale
        )
        if return_weights:
            return self.B_est

    def threshold(self):
        self.B_est = (np.abs(self.B_est) >= self.thresh_val) * self.B_est

    def get_auprc(self, W_gt, n_points=50):
        baseline, area = compute_auprc(W_gt, np.abs(self.B_est), n_points=n_points)
        return baseline, area

    def forwardPass(self, datasets):
        predictions = list()
        for data in datasets:
            pred = data @ self.B_est
            predictions.append(pred)

        return predictions

    def get_shd(self, W_gt):
        W_est = np.abs(self.B_est) > 0
        shd, _ = compute_shd(W_gt, W_est)
        return shd

    def computeLDG(self):
        if self.B_est.shape[0] > 20:
            print(
                "WARNING: The method might be slow - Need to implement a more efficient way to compute the gradient."
            )
        I = np.eye(self.B_est.shape[0])
        det = np.linalg.det(I - self.B_est.T)
        logdetgrad = math.log(np.abs(det))
        return logdetgrad

    def get_adjacency(self):
        return self.B_est

    def computeNLL(self, x, intervention_set):
        I = np.eye(x.shape[1])
        observed_set = np.setdiff1d(np.arange(x.shape[1]), intervention_set)
        U = np.zeros((x.shape[1], x.shape[1]))
        U[observed_set, observed_set] = 1

        e = x @ (I - self.B_est @ U)
        logpe = standard_normal_logprob(
            e[:, observed_set], noise_scale=self.noise_scale[observed_set] ** 0.5
        ).sum(axis=1)
        logdetgrad = self.computeLDG()
        logdetgrad_vec = np.ones(logpe.shape) * logdetgrad
        logpx = logpe + logdetgrad_vec
        return -1 * logpx.mean()

    def predict(self, latents, intervention_sets, x_inits):
        pred_list = list()
        n_nodes = self.B_est.shape[0]
        for latent, intervention_set, x_init in zip(latents, intervention_sets, x_inits):
            observed_set = np.setdiff1d(np.arange(self.B_est.shape[0]), intervention_set)
            U = np.zeros((n_nodes, n_nodes))
            U[observed_set, observed_set] = 1
            I = np.zeros((n_nodes, n_nodes))
            I[intervention_set, intervention_set] = 1
            data_pred = (latent @ U + x_init @ I) @ np.linalg.inv(np.eye(n_nodes) - U @ self.B_est.T).T
            pred_list.append(data_pred)

        return pred_list

    def predictLikelihood(self, datasets, intervention_sets):
        nll_list = [
            self.computeNLL(dataset, intervention_set) / self.B_est.shape[0]
            for dataset, intervention_set in zip(datasets, intervention_sets)
        ]
        return nll_list

    def predictConditionalMean(self, datasets, intervention_sets, noise_scale=0.5):
        latents = [
            np.random.randn(datasets[i].shape[0], datasets[i].shape[1]) * self.noise_scale
            for i in range(len(datasets))
        ]
        pred_list = self.predict(latents, intervention_sets, x_inits=datasets)

        return [pred.mean(axis=0) for pred in pred_list]
