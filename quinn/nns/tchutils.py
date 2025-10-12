#!/usr/bin/env python
"""Various useful PyTorch related utilities."""

import copy
import torch


torch.set_default_dtype(torch.double)

def tch(arr, device='cpu', rgrad=False):
    """Convert a numpy array to torch Tensor.

    Args:
        arr (np.ndarray): A numpy array of any size.
        device (str, optional): It represents where tensors are allocated. Default to cpu.
        rgrad (bool, optional): Whether to require gradient tracking or not.

    Returns:
        torch.Tensor: Torch tensor of the same size as the input numpy array.
    """

    # return torch.from_numpy(arr.astype(np.double)).to(device)
    # return torch.from_numpy(arr).double()
    return torch.tensor(arr, requires_grad=rgrad, device=device)


def npy(arr):
    """Convert a torch tensor to numpy array. 

    Args:
        arr (torch.Tensor): Torch tensor of any size.

    Returns:
        np.ndarray: Numpy array of the same size as the input torch tensor.
    """
    # return data.detach().numpy()
    return arr.cpu().data.numpy()

def print_nnparams(nnmodel, names_only=False):
    """Print parameter names of a PyTorch NN module and optionally, values.

    Args:
        nnmodel (torch.nn.Module): The torch NN module.
        names_only (bool, optional): Print names only. Default is False.
    """
    assert(isinstance(nnmodel, torch.nn.Module))

    for name, param in nnmodel.named_parameters():
        if names_only:
            print(f"{name}, shape {npy(param.data).shape}")
        else:
            print(name, param.data)

def flatten_params(parameters):
    """Flattens all parameters into an array.

    Args:
        parameters (torch.nn.Parameters): Description

    Returns:
        (torch.Tensor, list[tuple]): A tuple of the flattened (1d) torch tensor and a list of pairs that correspond to start/end indices of the flattened parameters.
    """
    l = [torch.flatten(p) for p in parameters]

    indices = []
    s = 0
    for p in l:
        size = p.shape[0]
        indices.append((s, s+size))
        s += size
    flat = torch.cat(l).view(-1, 1)
    return flat, indices


def recover_flattened(flat_params, indices, model):
    """Fills the values of corresponding parameters given the flattened form.

    Args:
        flat_params (np.ndarray): A flattened form of parameters.
        indices (list[tuple]): A list of pairs that correspond to start/end indices of the flattened parameters.
        model (torch.nn.Module): The underlying PyTorch NN module.

    Returns:
        list[torch.Tensor]: List of recovered parameters, reshaped and ordered to match the model.
    """
    l = [flat_params[s:e] for (s, e) in indices]
    for i, p in enumerate(model.parameters()):
        l[i] = l[i].view(*p.shape)
    return l


