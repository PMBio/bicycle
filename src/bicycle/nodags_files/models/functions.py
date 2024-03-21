import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import networkx as nx

from bicycle.nodags_files.models.layers.mlpLipschitz import linearLipschitz
from bicycle.nodags_files.models.layers.masks import GumbelAdjacency, GumbelInNOut

class indMLPFunction(nn.Module):
    """
    ------------------------------------------------------------------------------
    A class for modelling each node function with an MLP
    ------------------------------------------------------------------------------
    """

    def __init__(self, 
                 n_nodes, 
                 lip_constant=0.9, 
                 n_layers=1, 
                 bias=False, 
                 hidden_sizes_provided=False, 
                 hidden_size_list=None, 
                 activation='tanh', 
                 n_iterations=2000, 
                 full_input=False, 
                 graph_given=False,
                 graph=None):
        super(indMLPFunction, self).__init__()
        self.n_nodes = n_nodes 
        self.lip_constant = lip_constant
        self.n_layers = n_layers
        self.bias_ = bias
        self.n_iterations = n_iterations
        self.full_input = full_input
        self.graph_given = graph_given
        if self.graph_given:
            self.graph_adj = nx.to_numpy_array(graph)
        else:
            self.graph_adj = None

        if hidden_sizes_provided:
            self.hidden_size_list = hidden_size_list 
        else:
            self.hidden_size_list = [[self.n_nodes] * self.n_layers] * self.n_nodes
        
        activation_dict = {'tanh': nn.Tanh(), 'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid()}
        if activation not in ['tanh', 'relu', 'sigmoid']:
            print("Invalid activation function. Defaulting to 'tanh'")
            activation = 'tanh'
        self.activation = activation_dict[activation]

        self.ind_function_lipschitz = self.lip_constant / np.sqrt(self.n_nodes)
        self._create_function_layers_()

    def forward(self, x):
        
        f_x = torch.zeros(x.size(), device=x.device)
        for i, f_i in enumerate(self.functions):
            ind_exc_i = np.setdiff1d(np.arange(self.n_nodes), np.array([i]))
            if self.full_input:
                x_input = x
            elif self.graph_given:
                Par = torch.diag(torch.tensor(self.graph_adj[:, i])).float()
                Par = Par.to(x.device)
                x_input_t = x @ Par
                x_input = x_input_t[:, ind_exc_i]
            else:
                x_input = x[:, ind_exc_i]
            f_i_x = f_i(x_input)
            f_x[:, i] = f_i_x.squeeze()

        return f_x

    def _create_function_layers_(self):

        self.functions = nn.ModuleList()
        layer_lipschitz_constant = self.ind_function_lipschitz ** (1 / (self.n_layers + 1))
        for i in range(self.n_nodes):
            in_features_list = [self.n_nodes if self.full_input else (self.n_nodes-1)] + self.hidden_size_list[i]
            out_features_list = self.hidden_size_list[i] + [1]
            function = nn.Sequential(OrderedDict([
                ('layer{}'.format(t), _non_lin_layer(in_features_list[t], out_features_list[t], self.activation, layer_lipschitz_constant, bias=self.bias_)) for t in range(self.n_layers+1)
            ]))
            self.functions.append(function)
    
    def compute_weights(self):
        for m in self.functions.modules():
            if isinstance(m, linearLipschitz):
                m.compute_weight(update=True, n_iterations=self.n_iterations)

# (done) TODO: Update the linearFunction to handle graph_given = True
class linearFunction(nn.Module):

    def __init__(self, n_nodes, lip_constant, bias=False, n_iterations=2000, full_input=False, graph_given=False, graph=None):
        super(linearFunction, self).__init__()
        self.n_nodes = n_nodes
        self.lip_constant = lip_constant
        self.bias_ = bias 
        self.n_iterations = n_iterations
        self.full_input = full_input

        self.graph_given = graph_given
        if self.graph_given:
            self.graph_adj = nx.to_numpy_array(graph)
        else:
            self.graph_adj = None

        self.layer = linearLipschitz(in_features=self.n_nodes, out_features=self.n_nodes, bias=self.bias_, lip_constant=self.lip_constant)
    
    def forward(self, x):
        if self.full_input:
            return self.layer(x)
        else:
            f_x = torch.zeros_like(x)
            for i in range(self.n_nodes):
                if self.graph_given:
                    U_i = torch.diag(torch.tensor(self.graph_adj[:, i])).float()
                    U_i = U_i.to(x.device)
                else:
                    U_i = torch.eye(self.n_nodes, device=x.device)
                    U_i[i,i] = 0

                f_i_x = self.layer(x @ U_i)[:, i]
                f_x[:, i] = f_i_x.squeeze()
            
            return f_x

def _non_lin_layer(in_f, out_f, activation, lip_constant=0.9, bias=False):
    if lip_constant == None:
        layer = nn.Linear(in_features = in_f, out_features = out_f, bias=bias)
    else:
        layer = linearLipschitz(in_features=in_f, out_features=out_f, lip_constant=lip_constant, bias=bias)
    if activation:
        return nn.Sequential(layer, activation)
    else:
        return layer

class nonlinearMLP(nn.Module):

    def __init__(self, n_nodes, lip_constant=0.9, n_layers=2, bias=False, n_iterations=2000, full_input=False, activation_fn='tanh', graph_given=False, graph=None):
        super(nonlinearMLP, self).__init__()
        self.n_nodes = n_nodes
        self.lip_constant = lip_constant
        self.bias_ = bias
        self.n_layers = n_layers
        self.n_iterations = n_iterations 
        self.full_input = full_input
        self.graph_given = graph_given
        if self.graph_given:
            self.graph_adj = nx.to_numpy_array(graph)
        else:
            self.graph_adj = None

        self.activation_type = activation_fn 
        activation_dict = {'tanh' : nn.Tanh(),
                           'relu' : nn.ReLU(),
                           'sigmoid' : nn.Sigmoid(),
                           'selu': nn.SELU(),
                           'gelu': nn.GELU(),
                           'none': None}
        self.activation = activation_dict[self.activation_type]
        self.layer_lip_const = [1.0] * self.n_layers + [self.lip_constant]
        self.layers = nn.Sequential(OrderedDict([
            ('layer{}'.format(t), _non_lin_layer(self.n_nodes, self.n_nodes, self.activation, self.layer_lip_const[t], bias=self.bias_)) for t in range(self.n_layers + 1)
        ]))
    
    def forward(self, x):
        f_x = torch.zeros_like(x)
        for i in range(self.n_nodes):
            if self.full_input:
                return self.layers(x)
            elif self.graph_given:
                Par = torch.diag(torch.tensor(self.graph_adj[:, i])).float()
                Par = Par.to(x.device)
            else:
                Par = torch.eye(self.n_nodes, device=x.device)
                Par[i,i] = 0
            f_i_x = self.layers(x @ Par)[:, i]
            f_x[:, i] = f_i_x.squeeze()
        return f_x

class factorMLPFunction(nn.Module):
    def __init__(
        self,
        n_nodes,
        n_factors,
        lip_constant=0.9,
        n_hidden=0,
        activation='tanh',
        n_iterations=2000,
        graph_given=False, 
        adj_mat=None,
        hidden_size_provided=False, 
        hidden_sizes=None,
        bias=False
    ):

        super(factorMLPFunction, self).__init__()
        self.n_nodes = n_nodes
        self.n_factors = n_factors
        self.lip_constant = lip_constant 
        self.n_hidden = n_hidden
        activation_dict = {'tanh': nn.Tanh(), 'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid()}
        if activation not in ['tanh', 'relu', 'sigmoid']:
            print("Invalid activation function. Defaulting to 'tanh'")
            activation = 'tanh'
        self.activation = activation_dict[activation]
        self.bias_ = bias 
        self.n_iterations = n_iterations 
        self.graph_given = graph_given 
        if self.graph_given:
            self.adj_mat = adj_mat 
        self.hidden_size_provided = hidden_size_provided 
        if self.hidden_size_provided:
            self.hidden_size_list = hidden_sizes
        else:
            self.hidden_size_list = [self.n_nodes] * self.n_hidden
        
        self.gumbel_inout = GumbelInNOut(self.n_nodes, self.n_factors)
        
        self.func_lip_const = self.lip_constant ** (1/2)
        self.factor_fun_lc = self.func_lip_const / np.sqrt(self.n_factors)
        self.var_fun_lc = self.func_lip_const / np.sqrt(self.n_nodes)
        self.create_functions()

    def create_functions(self):
        
        fac_layer_lip_const = self.factor_fun_lc ** (1 / (self.n_hidden + 1))
        self.factor_func = nn.ModuleList()
        for i in range(self.n_factors):
            in_features_list = [self.n_nodes] + self.hidden_size_list
            out_features_list = self.hidden_size_list + [1]
            function = nn.Sequential(OrderedDict([
                ('layer{}'.format(t), _non_lin_layer(in_features_list[t], out_features_list[t], self.activation, fac_layer_lip_const, bias=self.bias_)) for t in range(self.n_hidden+1)
            ]))
            self.factor_func.append(function)
        
        var_layer_lip_const = self.var_fun_lc ** (1 / (self.n_hidden + 1))
        self.var_func = nn.ModuleList()
        for i in range(self.n_nodes):
            in_features_list = [self.n_factors] + self.hidden_size_list
            out_features_list = self.hidden_size_list + [1]
            function = nn.Sequential(OrderedDict([
                ('layer{}'.format(t), _non_lin_layer(in_features_list[t], out_features_list[t], self.activation, var_layer_lip_const, bias=self.bias_)) for t in range(self.n_hidden + 1)
            ]))
            self.var_func.append(function)
        
    def compute_weights(self):
        for m in self.factor_func.modules():
            if isinstance(m, linearLipschitz):
                m.compute_weight(update=True, n_iterations=self.n_iterations)
        
        for m in self.var_func.modules():
            if isinstance(m, linearLipschitz):
                m.compute_weight(update=True, n_iterations=self.n_iterations)

    def forward(self, x):
        num_batch = x.size(0)

        # sample masks
        mask_node2module, mask_module2node = self.gumbel_inout(num_batch)
        mask_module2node = torch.transpose(mask_module2node, 1, 2)

        g_x = torch.zeros((num_batch, self.n_factors), device=x.device)
        for i, g_i in enumerate(self.factor_func):
            if not self.graph_given:
                g_i_x = g_i(mask_node2module[:, :, i] * x)
            
            g_x[:, i] = g_i_x.squeeze()
        
        f_x = torch.zeros_like(x)
        for i, f_i in enumerate(self.var_func):
            if not self.graph_given:
                f_i_x = f_i(mask_module2node[:, :, i] * g_x)
            
            f_x[:, i] = f_i_x.squeeze()
        
        return f_x

    def threshold(self, threshold=None):
        with torch.no_grad():
            self.gumbel_inout.freeze_threshold(threshold)
    
    def get_w_adj(self):
        return self.gumbel_inout.get_proba_features()

class gumbelSoftMLP(nn.Module):
    def __init__(
        self, 
        n_nodes, 
        lip_constant,
        n_hidden=0,
        activation='tanh',
        n_iterations=2000,
        graph_given=False, 
        graph=None,
        hidden_size_provided=False,
        hidden_sizes=None,
        bias=False
    ):

        super(gumbelSoftMLP, self).__init__()
        self.n_nodes = n_nodes
        self.lip_constant = lip_constant
        self.n_hidden = n_hidden
        activation_dict = {
            'tanh' : nn.Tanh(),
            'relu' : nn.ReLU(),
            'sigmoid' : nn.Sigmoid(),
            'selu' : nn.SELU(),
            'gelu' : nn.GELU(),
            'none' : None
        }
        self.activation = activation_dict[activation]

        self.bias_ = bias 
        self.n_iterations = n_iterations 
        self.graph_given = graph_given 
        if graph_given:
            self.graph_adj = nx.to_numpy_array(graph)
        else:
            self.graph_adj = None
            
        self.hidden_size_provided = hidden_size_provided 
        if self.hidden_size_provided:
            self.hidden_size_list = hidden_sizes
        else:
            self.hidden_size_list = [self.n_nodes] * self.n_hidden 

        self.gumbel_soft_layer = GumbelAdjacency(self.n_nodes)
        self.layer_lip_const = [1.0] * self.n_hidden + [self.lip_constant]
        self.layers = nn.Sequential(OrderedDict([
            ('layer{}'.format(t), _non_lin_layer(self.n_nodes, self.n_nodes, self.activation, self.layer_lip_const[t], bias=self.bias_)) for t in range(self.n_hidden + 1)
        ]))
    
    def forward(self, x):
        num_batch = x.size(0)
        graph_adj = self.gumbel_soft_layer(num_batch)
        f_x = torch.zeros_like(x)
        for i in range(self.n_nodes):
            if not self.graph_given:
                self_loop_mask = torch.ones_like(x)
                self_loop_mask[:, i] = 0
                f_i_x = self.layers(self_loop_mask * graph_adj[:, :, i] * x)[:, i]
            
            else:
                Par = torch.diag(torch.tensor(self.graph_adj[:, i])).float()
                Par = Par.to(x.device)
                f_i_x = self.layers(x @ Par)[:, i]
            
            f_x[:, i] = f_i_x.squeeze()
        return f_x 

    def get_w_adj(self):
        return self.gumbel_soft_layer.get_proba()

        


