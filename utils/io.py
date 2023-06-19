import torch
import torch.nn as nn


def get_state_dict_on_cpu(obj):
    cpu_device = torch.device('cpu')
    state_dict = obj.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(cpu_device)
    return state_dict


def save_ckpt(ckpt_name, models, optimizers, n_iter):
    ckpt_dict = {'n_epoch': n_iter}
    for prefix, model in models:
        ckpt_dict[prefix] = get_state_dict_on_cpu(model)

    for prefix, optimizer in optimizers:
        ckpt_dict[prefix] = optimizer.state_dict()
    torch.save(ckpt_dict, ckpt_name)


# def load_ckpt(ckpt_name, models, device,optimizers=None):
#     ckpt_dict = torch.load(ckpt_name)
#
#     from collections import OrderedDict
#
#
#     for prefix, model in models:
#         assert isinstance(model, nn.Module)
#         new_state_dict = OrderedDict()
#         for k, v in ckpt_dict[prefix].items():
#             name = k[7:]  # remove `module.`
#             new_state_dict[name] = v
#         # load params
#         model.load_state_dict(new_state_dict,strict=False)
#     if optimizers is not None:
#         for prefix, optimizer in optimizers:
#             for k, v in ckpt_dict[prefix].items():
#                 name = k[7:]  # remove `module.`
#                 new_state_dict[name] = v
#             optimizer.load_state_dict(new_state_dict,map_location=device)
#     return ckpt_dict['n_iter']
#

def load_ckpt(ckpt_name, models, device,optimizers=None):

    for prefix, model in models:
        assert isinstance(model, nn.Module)
        model.load_state_dict(ckpt_dict[prefix], strict=False)
    if optimizers is not None:
        for prefix, optimizer in optimizers:
            optimizer.load_state_dict(ckpt_dict[prefix],map_location=device)
    return ckpt_dict['n_epoch']
