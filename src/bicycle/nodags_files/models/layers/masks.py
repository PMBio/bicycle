import numpy as np 
import torch 
from tqdm import tqdm 

from bicycle.nodags_files.models.utils.gumbel import gumbel_sigmoid, gumbel_softmax 

class GumbelAdjacency(torch.nn.Module):
    """
    Probabilistic mask used for DAG learning.
    Can sample a matrix and backpropagate using the
    Gumbel straigth-through estimator.
    :param int num_vars: number of variables
    """

    def __init__(self, num_rows, num_cols=None):
        super(GumbelAdjacency, self).__init__()
        if num_cols is None:
            # square matrix
            self.num_vars = (num_rows, num_rows)
        else:
            self.num_vars = (num_rows, num_cols)
        self.log_alpha = torch.nn.Parameter(torch.zeros(self.num_vars))
        self.tau = 1
        self.reset_parameters()

    def forward(self, bs):
        adj = gumbel_sigmoid(self.log_alpha, bs, tau=self.tau, hard=True)
        return adj

    def get_proba(self):
        """Returns probability of getting one"""
        return torch.sigmoid(self.log_alpha / self.tau)

    def reset_parameters(self):
        torch.nn.init.constant_(self.log_alpha, 0)

class GumbelInNOut(torch.nn.Module):
    """
    Random matrix M used for encoding egdes between modules and genes.
    Category:
    - 0 means no edge
    - 1 means node2module edge
    - 2 means module2node edge
    Can sample a matrix and backpropagate using the
    Gumbel straigth-through estimator.
    :param int num_vars: number of variables
    """

    def __init__(self, num_nodes, num_modules):
        super(GumbelInNOut, self).__init__()
        self.num_vars = (num_nodes, num_modules)
        self.log_alpha = torch.nn.Parameter(torch.zeros(num_nodes, num_modules, 3))
        self.register_buffer(
            "freeze_node2module",
            torch.zeros((num_nodes, num_modules)),
        )
        self.register_buffer(
            "freeze_module2node",
            torch.zeros((num_nodes, num_modules)),
        )
        self.tau = 1
        self.drawhard = True
        self.deterministic = False
        self.reset_parameters()

    def forward(self, bs):
        if not self.deterministic:
            design = gumbel_softmax(
                self.log_alpha, bs, tau=self.tau, hard=self.drawhard
            )
            node2module = design[:, :, :, 0]
            module2node = design[:, :, :, 1]
        else:
            node2module = self.freeze_node2module.unsqueeze(0)
            module2node = self.freeze_module2node.unsqueeze(0)
        return node2module, module2node

    def freeze_threshold(self, threshold):
        """Returns probability of being assigned into a bucket"""
        design = torch.softmax(self.log_alpha / self.tau, -1)
        node2module = design[:, :, 0]
        module2node = design[:, :, 1]
        max_in_out = torch.maximum(node2module, module2node)
        # zero for low confidence
        mask_keep = max_in_out >= threshold
        # track argmax
        self.freeze_node2module = (node2module == max_in_out) * mask_keep
        self.freeze_module2node = (module2node == max_in_out) * mask_keep
        self.deterministic = True
        print("Freeze threshold:" + str(self.freeze_module2node.device))

    def get_proba_modules(self):
        """Returns probability of being assigned into a bucket"""
        design = torch.softmax(self.log_alpha / self.tau, -1)
        node2module = design[:, :, 0]
        module2node = design[:, :, 1]
        mat = module2node.T @ node2module
        # above is correct except for diagonal values (individual values in the matrix product are corr.)
        mask_modules = torch.ones(self.num_vars[1], self.num_vars[1]) - torch.eye(
            self.num_vars[1]
        )
        return mat * mask_modules.type_as(mat)

    def get_proba_features(self, threshold=None):
        """Returns probability of being assigned into a bucket"""
        design = torch.softmax(self.log_alpha / self.tau, -1)
        node2module = design[:, :, 0]
        module2node = design[:, :, 1]
        if not threshold:
            # return a differentiable tensor
            mat = node2module @ module2node.T
            # above is correct except for diagonal values (individual values in the matrix product are corr.)
            mask_nodes = torch.ones(self.num_vars[0], self.num_vars[0]) - torch.eye(
                self.num_vars[0]
            )
            return mat * mask_nodes.type_as(mat)
        else:
            # here return a matrix without grad
            # we're thresholding here according to the edge direction confidence
            max_in_out = torch.maximum(design[:, :, 0], design[:, :, 1])
            # zero for low confidence
            mask_keep = design[:, :, 0] + design[:, :, 1] >= threshold
            # track argmax
            node2module = (design[:, :, 0] == max_in_out) * mask_keep
            module2node = (design[:, :, 1] == max_in_out) * mask_keep
            # that product below has no self cycles
            return (
                node2module.type_as(self.log_alpha)
                @ module2node.type_as(self.log_alpha).T
            )

    def get_proba_(self):
        design = torch.softmax(self.log_alpha / self.tau, -1)
        node2module = design[:, :, 0]
        module2node = design[:, :, 1]
        return node2module, module2node

    def reset_parameters(self):
        torch.nn.init.constant_(self.log_alpha, 1)