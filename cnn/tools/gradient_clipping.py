#!/usr/bin/python
# -*- coding: utf-8 -*-
# Created: 2020-05-26 11:18:53


import torch

def _add_noise(grad_in, max_norm, var_gamma, device):
    gaussian_noise = torch.normal(0,
                                var_gamma * max_norm,
                                grad_in.shape,
                                device=device,
                                )
    return grad_in + gaussian_noise * max_norm


def _conv_op(grad_in, args_in):
    grad = grad_in.detach()
    grad_ = grad.reshape((grad.shape[0], grad.shape[1], -1))
    max_coeff = torch.norm(grad_, dim=-1) / args_in[0]
    max_coeff[max_coeff < 1.0] = torch.tensor(1.0)
    grad_ = torch.div(grad_, max_coeff.unsqueeze(-1))
    grad_ = grad_.reshape(grad.shape)
    return _add_noise(grad_, *args_in)



def _linear_op(grad_in, args_in):
    grad = grad_in.detach()
    max_coeff = torch.norm(grad, dim=-1) / args_in[0]
    max_coeff[max_coeff < 1.0] = torch.tensor(1.0)
    grad = torch.div(grad, max_coeff.unsqueeze(-1))
    return _add_noise(grad, *args_in)


# def _bn_op(grad_in, args_in):
#     grad = grad_in.detach()
#     grad_ = grad.squeeze()
#     max_coeff = torch.norm(grad_, dim=-1) / args_in[0]
#     max_coeff[max_coeff < 1.0] = torch.tensor(1.0)
#     grad_ = torch.div(grad_, max_coeff.unsqueeze(-1))
#     grad_ = _add_noise(grad_, *args_in)
#     return grad_.unsqueeze(-1).unsqueeze(-1)

def _bn_op(grad_in, args_in):
    grad_ = grad_in.detach()
    max_coeff = torch.norm(grad_, dim=-1) / args_in[0]
    max_coeff[max_coeff < 1.0] = torch.tensor(1.0)
    grad_ = torch.div(grad_, max_coeff)
    grad_ = _add_noise(grad_, *args_in)
    return grad_


def clipping_dispatcher(named_param_list, max_norm, var_gamma, device, logger):
    args = (max_norm, var_gamma, device)
    with torch.no_grad():
        for name, param in named_param_list:
            if 'cells' in name:
                if len(param.shape) < 2:
                    continue
                if param.shape[-1] == param.shape[-2]:
                    if param.shape[-1] > 1:
                        # conv layer
                        param.grad.data = _conv_op(param.grad, args)
                    # elif param.shape[-1] == 1:
                    #     # BN layer
                    #     # print('bn: ', name, 'shape:', param.shape)
                    #     param.grad.data = _bn_op(param.grad, args)
            elif 'aux_head' in name:
                if len(param.shape) < 2:
                    if 'weight' in name:
                        # BN layer
                        param.grad.data = _bn_op(param.grad, args)
                else:
                    if param.shape[-1] == param.shape[-2]:
                        if param.shape[-1] > 1:
                            # conv layer
                            param.grad.data = _conv_op(param.grad, args)
            elif 'linear' in name:
                param.grad.data = _linear_op(param.grad, args)
            elif 'stem.0' in name:
                # stem conv layer
                param.grad.data = _conv_op(param.grad, args)
            elif 'stem.1.weight' in name:
                # BN layer
                param.grad.data = _bn_op(param.grad, args)
            elif 'stem.1.bias' in name:
                continue
            else:
                print('Unrecognized parameter: ', name, 'shape:', param.shape)

