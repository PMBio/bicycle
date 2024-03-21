import math
import torch
import torch.nn as nn 
import torch.nn.init as init
import torch.nn.functional as F

class linearLipschitz(nn.Module):
    """
    ---------------------------------------------------------------------------------------------------------------
    Linear Lipschitz Layer - Linear layer with a fixed Lipschitz constant
    ---------------------------------------------------------------------------------------------------------------
    Parameters:
    1) in_features - (int) - Dimension of the input features.
    2) out_features - (int) - Dimension of the output features.
    3) lip_constant - (float) - Maximum Lipschitz constant of the layer.
    4) bias - (bool) - If True, then a bias term is added to the list of model parameters. 
    5) n_iterations - (int) - Number of power iterations used for computing the spectral norm of the layer weights.

    Run compute_weight() every time the weights are updated to ensure that the Lipschitz constraint is always satisfied.   
    """

    def __init__(self, in_features, out_features, lip_constant=0.9, bias=True, n_iterations=None):
        super(linearLipschitz, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lip_constant = lip_constant
        self.n_iterations = n_iterations

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        # creating the normal random vectors and storing it in the buffer. This way these vectors won't be a part
        # of the parameter list of the layer and hence won't be updated during backprop. 
        h, w = self.weight.shape
        self.register_buffer('scale', torch.tensor(0.))
        self.register_buffer('u', F.normalize(self.weight.new_empty(h).normal_(0, 1), dim=0))
        self.register_buffer('v', F.normalize(self.weight.new_empty(w).normal_(0, 1), dim=0))
        self.compute_weight(True, 200, init=True)

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(10))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def compute_weight(self, update=True, n_iterations=None, init=False):
        n_iterations = self.n_iterations if n_iterations is None else n_iterations
        if n_iterations is None:
            n_iterations = 2000

        u = self.u
        v = self.v 
        weight = self.weight 
        if update:
            with torch.no_grad(): # doing this since we don't the following computations towards computing the gradients with respect to the parameters.
                itrs_used = 0
                for _ in range(n_iterations):

                    # The spectral norm is obtain by repeatedly performing `u^T W v`, 
                    # where `u` and `v` are the first left and right singular vectors (power iteration)                     
                    v = F.normalize(torch.mv(weight.t(), u), dim=0, out=v)
                    u = F.normalize(torch.mv(weight, v), dim=0, out=u)
                    itrs_used = itrs_used + 1
                
                if itrs_used > 0:
                    u = u.clone()
                    v = v.clone()
            
        sigma = torch.dot(u, torch.mv(weight, v))
        with torch.no_grad():
            self.scale.copy_(sigma)
        # soft normalization: normalize the weights when the spectral norm of the weights mat is greater than the desired Lipschitz constant. 
        if init:
            factor = sigma / self.lip_constant 
        else:
            factor = torch.max(torch.ones(1).to(weight.device), sigma / self.lip_constant)
        weight = weight / factor 
        self.weight.data = weight

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, weight={self.weight}, bias={self.bias}, lip_constant={self.lip_constant}, n_iters={self.n_iterations}'


        
