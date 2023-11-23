import torch
import torch.nn as nn

"""
    Input:
        - net: model to be pruned
        - u: coefficient that determines the pruning threshold
    Output:
        None (in-place modification on the model)
"""

def CLP(net, u):
    params = net.state_dict()
    modules = list(net.named_modules())
    for module_ix, (name, m) in enumerate(modules):
        next_mod = modules[module_ix+1] if module_ix <= len(modules)-2 else None
        if isinstance(next_mod, nn.BatchNorm2d):
            conv = m
            continue
        if isinstance(m, nn.BatchNorm2d):
            std = m.running_var.sqrt()
        elif isinstance(m, nn.Conv2d):
            conv = m
        else:
            continue
        weight = m.weight

        channel_lips = []
        for idx in range(weight.shape[0]):
            # Combining weights of convolutions and BN
            w = conv.weight[idx].reshape(conv.weight.shape[1], -1)
            if isinstance(m, nn.BatchNorm2d):
                w = w * (weight[idx]/std[idx]).abs()
            channel_lips.append(torch.svd(w.cpu())[1].max())
        channel_lips = torch.Tensor(channel_lips)

        index = torch.where(channel_lips>channel_lips.mean() + u*channel_lips.std())[0]
        print(f'module number: {module_ix}, length of index: {len(index)}')

        params[name+'.weight'][index] = 0
        if name+'.bias' in params:
            params[name+'.bias'][index] = 0

    net.load_state_dict(params)