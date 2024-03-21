import torch

def gumbel_sigmoid(log_alpha, bs, tau=1, hard=True):
    shape = tuple([bs] + list(log_alpha.size()))
    uniform = torch.distributions.Uniform(0, 1).sample(shape).type_as(log_alpha)
    logistic_noise = torch.log(uniform) - torch.log(1 - uniform)
    y_soft = torch.sigmoid((log_alpha + logistic_noise) / tau)

    if hard:
        y_hard = (y_soft > 0.5).type_as(y_soft)
        # straight through logistic
        y = y_hard.detach() - y_soft.detach() + y_soft
    else:
        y = y_soft

    return y

def gumbel_softmax(log_alpha, bs, tau=1, hard=True):
    shape = tuple([bs] + list(log_alpha.size()))
    gumbels = (
        -torch.empty(
            shape, memory_format=torch.legacy_contiguous_format, device=log_alpha.device
        )
        .exponential_()
        .log()
    )  # ~Gumbel(0,1)
    gumbels = (log_alpha + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(-1)

    if hard:
        # Straight through.
        index = y_soft.max(-1, keepdim=True)[1]
        y_hard = torch.zeros_like(gumbels).scatter_(-1, index, 1.0)
        ret = y_hard.detach() - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret
