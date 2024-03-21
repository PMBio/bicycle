import torch
import torch.nn as nn 
import numpy as np 
import math 
import time 


class iResBlock(nn.Module):
    """
    ----------------------------------------------------------------------------------------
    The class for a single residual map, i.e., (I -f)(x) = e. 
    ----------------------------------------------------------------------------------------
    The forward method computes the residual map and also log-det-Jacobian of the map. 

    Parameters:
    1) func - (nn.Module) - torch module for modelling the function f in (I - f).
    2) n_power_series - (int/None) - Number of terms used for computing determinent of log-det-Jac, 
                                     set it to None to use Russian roulette estimator. 
    3) neumann_grad - (bool) - If True, Neumann gradient estimator is used for Jacobian.
    4) n_dist - (string) - distribution used to sample n when using Russian roulette estimator. 
                           'geometric' - geometric distribution.
                           'poisson' - poisson distribution.
    5) lamb - (float) - parameter of poisson distribution.
    6) geom_p - (float) - parameter of geometric distribution.
    7) n_samples - (int) - number of samples to be sampled from n_dist. 
    8) grad_in_forward - (bool) - If True, it will store the gradients of Jacobian with respect to 
                                  parameters in the forward pass. 
    9) n_exact_terms - (int) - Minimum number of terms in the power series. 
    """
    def __init__(self, func, n_power_series, neumann_grad=True, n_dist='geometric', lamb=2., geom_p=0.5, n_samples=1, grad_in_forward=False, n_exact_terms=2, var=None, init_var=0.5, dag_input=False, lin_logdet=False, centered=True):
        super(iResBlock, self).__init__()
        self.f = func
        self.geom_p = nn.Parameter(torch.tensor(np.log(geom_p) - np.log(1. - geom_p)))
        self.lamb = nn.Parameter(torch.tensor(lamb))
        self.n_dist = n_dist
        self.n_power_series = n_power_series 
        self.neumann_grad = neumann_grad 
        self.grad_in_forward = grad_in_forward
        self.n_exact_terms = n_exact_terms
        self.n_samples = n_samples
        self.dag_input = dag_input
        self.lin_logdet = lin_logdet
        self.centered = centered
        if var == None:
            self.var = nn.Parameter(init_var*torch.ones(self.f.n_nodes).float())
        else:
            self.var = var
        
        if not self.centered:
            self.mu = nn.Parameter(torch.zeros(self.f.n_nodes).float())
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.mu = torch.zeros(self.f.n_nodes).float().to(device)

        if dag_input:
            self.Lambda = nn.Parameter(torch.zeros(self.f.n_nodes).float())

    def forward(self, x, intervention_set, logdet=False, neumann_grad=True, logdet_time_measure=False):
        # set intervention set to [None]
        self.neumann_grad = neumann_grad

        U = torch.zeros(x.shape[1], x.shape[1], device=x.device)
        observed_set = np.setdiff1d(np.arange(x.shape[1]), intervention_set)
        U[observed_set, observed_set] = 1
        
        if not logdet:
            y = x - self.f(x) @ U
            return y
        else:
            if self.dag_input:
                Lamb_mat = torch.diag(torch.exp(self.Lambda))
                Lamb_mat_inv = torch.diag(1/torch.exp(self.Lambda))
                x_inp = (x - self.mu) @ Lamb_mat
            else:
                x_inp = x - self.mu
            f_x, logdetgrad, cmp_time = self._logdetgrad(x_inp, U)
            if logdet_time_measure:
                return x - f_x @ U, logdetgrad, cmp_time
            else:
                if self.dag_input:
                    return (x - self.mu) - f_x @ Lamb_mat_inv @ U, logdetgrad
                else:
                    return (x - self.mu) - f_x @ U, logdetgrad 

    # TODO have to update this for when self.dag_input = True (DONE - no change needed)
    def predict_from_latent(self, latent_vec, n_iter=10, intervention_set=[None], init_provided=False, x_init=None):
        if init_provided:
            x = torch.tensor(x_init).float().to(latent_vec.device) 
        else:
            x = torch.randn(latent_vec.size(), device=latent_vec.device)
        c = torch.zeros_like(x)
        obs_set = np.setdiff1d(np.arange(x.shape[1]), intervention_set)
        U = torch.zeros(x.shape[1], x.shape[1], device=x.device)
        U[obs_set, obs_set] = 1
        if intervention_set[0] != None:
            c[:, intervention_set] = torch.tensor(x_init[:, intervention_set]).float().to(latent_vec.device)

        for _ in range(n_iter):
            x = self.f(x - self.mu) @ U + latent_vec @ U + c + self.mu
        
        return x 

    def _logdetgrad(self, x, U):
        with torch.enable_grad():
            if self.n_dist == 'geometric':
                geom_p = torch.sigmoid(self.geom_p).item()
                sample_fn = lambda m: geometric_sample(geom_p, m)
                rcdf_fn = lambda k, offset: geometric_1mcdf(geom_p, k, offset)
            elif self.n_dist == 'poisson':
                lamb = self.lamb.item()
                sample_fn = lambda m: poisson_sample(lamb, m)
                rcdf_fn = lambda k, offset: poisson_1mcdf(lamb, k, offset)
            
            if self.training:
                if self.n_power_series is None:
                    # Unbiased estimation.
                    lamb = self.lamb.item()
                    n_samples = sample_fn(self.n_samples)
                    n_power_series = max(n_samples) + self.n_exact_terms
                    coeff_fn = lambda k: 1 / rcdf_fn(k, self.n_exact_terms) * \
                        sum(n_samples >= k - self.n_exact_terms) / len(n_samples)
                else:
                    # Truncated estimation.
                    n_power_series = self.n_power_series
                    coeff_fn = lambda k: 1.

            vareps = torch.randn_like(x)

            if self.lin_logdet:
                estimator_fn = linear_logdet_estimator
            else:
                if self.training and self.neumann_grad:
                    estimator_fn = neumann_logdet_estimator
                else:
                    estimator_fn = basic_logdet_estimator

            if self.training and self.grad_in_forward:
                f_x, logdetgrad = mem_eff_wrapper(
                    estimator_fn, self.f, x, n_power_series, vareps, coeff_fn, self.training
                )
            else:
                x = x.requires_grad_(True)
                f_x = self.f(x)
                tic = time.time()
                if self.lin_logdet:
                    Weight = self.f.layer.weight
                    self_loop_mask = torch.ones_like(Weight)
                    ind = np.diag_indices(Weight.shape[0])
                    self_loop_mask[ind[0], ind[1]] = 0 
                    logdetgrad = estimator_fn(U @ self_loop_mask * Weight, x.shape[0])
                else:
                    logdetgrad = estimator_fn(f_x @ U, x, n_power_series, vareps, coeff_fn, self.training)
                toc = time.time()
                comp_time = toc - tic 

        return f_x, logdetgrad.view(-1, 1), comp_time
        
def basic_logdet_estimator(g, x, n_power_series, vareps, coeff_fn, training):
    vjp = vareps
    logdetgrad = torch.tensor(0.).to(x)
    for k in range(1, n_power_series + 1):
        vjp = torch.autograd.grad(g, x, vjp, create_graph=training, retain_graph=True)[0]
        tr = torch.sum(vjp.view(x.shape[0], -1) * vareps.view(x.shape[0], -1), 1)
        delta = -1 / k * coeff_fn(k) * tr
        logdetgrad = logdetgrad + delta
    return logdetgrad


def neumann_logdet_estimator(g, x, n_power_series, vareps, coeff_fn, training):
    vjp = vareps
    neumann_vjp = vareps
    with torch.no_grad():
        for k in range(1, n_power_series + 1):
            vjp = torch.autograd.grad(g, x, vjp, retain_graph=True)[0]
            neumann_vjp = neumann_vjp + (-1) * coeff_fn(k) * vjp
    vjp_jac = torch.autograd.grad(g, x, neumann_vjp, create_graph=training)[0]
    logdetgrad = torch.sum(vjp_jac.view(x.shape[0], -1) * vareps.view(x.shape[0], -1), 1)
    return logdetgrad

def linear_logdet_estimator(W, bs):
    n = W.shape[0]
    I = torch.eye(n, device=W.device)
    return torch.log(torch.det(I - W)) * torch.ones(bs, 1, device=W.device)



def mem_eff_wrapper(): # Function to store the gradients in the forward pass. To be implemented. 
    return 0

def geometric_sample(p, n_samples):
    return np.random.geometric(p, n_samples)

def geometric_1mcdf(p, k, offset):
    if k <= offset:
        return 1.
    else:
        k = k - offset
    """P(n >= k)"""
    return (1 - p)**max(k - 1, 0)

def poisson_sample(lamb, n_samples):
    return np.random.poisson(lamb, n_samples)

def poisson_1mcdf(lamb, k, offset):
    if k <= offset:
        return 1.
    else:
        k = k - offset
    """P(n >= k)"""
    sum = 1.
    for i in range(1, k):
        sum += lamb**i / math.factorial(i)
    return 1 - np.exp(-lamb) * sum