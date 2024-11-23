import logging
import os
import random
from datetime import datetime

import numpy as np
import torch
from torch import Tensor


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S'
    )
    lg.setLevel(level)
    if tofile:
        os.makedirs(root, exist_ok=True)
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


def load(name, net: torch.nn.Module):
    state_dicts = torch.load(name, map_location='cpu', weights_only=True)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)


def clamp(input: Tensor, min=None, max=None):
    ndim = input.ndimension()
    if min is None:
        pass
    elif isinstance(min, (float, int)):
        input = torch.clamp(input, min=min)
    elif isinstance(min, torch.Tensor):
        if min.ndimension() == ndim - 1 and min.shape == input.shape[1:]:
            input = torch.max(input, min.view(1, *min.shape))
        else:
            assert min.shape == input.shape
            input = torch.max(input, min)
    else:
        raise ValueError("min can only be None | float | torch.Tensor")

    if max is None:
        pass
    elif isinstance(max, (float, int)):
        input = torch.clamp(input, max=max)
    elif isinstance(max, torch.Tensor):
        if max.ndimension() == ndim - 1 and max.shape == input.shape[1:]:
            input = torch.min(input, max.view(1, *max.shape))
        else:
            assert max.shape == input.shape
            input = torch.min(input, max)
    else:
        raise ValueError("max can only be None | float | torch.Tensor")
    return input


def decoded_message_acc_rate(messages: Tensor, decoded_messages: Tensor):
    acc_rate = torch.sum(messages.ge(0) == decoded_messages.ge(0)) / messages.numel()
    return acc_rate.item() * 100


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')
