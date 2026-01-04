"""
Common Utilities

Provides general helper functions for the drone navigation system.
"""

import sys
import torch
import numpy as np
import random
import os
from datetime import datetime


def solve_argv_conflict(args_list):
    """
    If there's overlap between args_list and commandline input, use commandline input.
    """
    arguments_to_be_removed = []
    arguments_size = []

    for argv in sys.argv[1:]:
        if argv.startswith('-'):
            size_count = 1
            for i, args in enumerate(args_list):
                if args == argv:
                    arguments_to_be_removed.append(args)
                    for more_args in args_list[i+1:]:
                        if not more_args.startswith('-'):
                            size_count += 1
                        else:
                            break
                    arguments_size.append(size_count)
                    break

    for args, size in zip(arguments_to_be_removed, arguments_size):
        args_index = args_list.index(args)
        for _ in range(size):
            args_list.pop(args_index)


def print_error(*message):
    """Print error message in red."""
    print('\033[91m', 'ERROR ', *message, '\033[0m')
    raise RuntimeError


def print_ok(*message):
    """Print success message in green."""
    print('\033[92m', *message, '\033[0m')


def print_warning(*message):
    """Print warning message in yellow."""
    print('\033[93m', *message, '\033[0m')


def print_info(*message):
    """Print info message in cyan."""
    print('\033[96m', *message, '\033[0m')


def get_time_stamp():
    """Get current timestamp as formatted string."""
    now = datetime.now()
    year = now.strftime('%Y')
    month = now.strftime('%m')
    day = now.strftime('%d')
    hour = now.strftime('%H')
    minute = now.strftime('%M')
    second = now.strftime('%S')
    return '{}-{}-{}-{}-{}-{}'.format(month, day, year, hour, minute, second)


def seeding(seed=0, torch_deterministic=False):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed
        torch_deterministic: Whether to use deterministic algorithms
    """
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed
