from functools import reduce, partial
import math
import operator
import os
import sys
from typing import Iterable, List, Optional, Set, Tuple, Union, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR, _LRScheduler, ReduceLROnPlateau




def feed_forward(input_size: int, hidden_size: int, dropout: float=0.2, num_layers: int=1, activation="relu") -> nn.Module:
    if num_layers == 0:
        return lambda x: x
    """Builds a 2-layer feed-forward network."""
    activation_fn = {
                        "relu": torch.nn.ReLU(),
                        "leaky_relu": torch.nn.LeakyReLU(),
                        "hardtanh": torch.nn.Hardtanh(),
                        "sigmoid": torch.nn.Sigmoid(),
                        "tanh": torch.nn.Tanh(),
                        "log_sigmoid": torch.nn.LogSigmoid(),
                        "softplus": torch.nn.Softplus(),
                        "identity": torch.nn.Identity(),
                    }
    layers = [nn.Linear(input_size, hidden_size),nn.Dropout(dropout),activation_fn[activation]]
    for i in range(num_layers-1):
        layers += [nn.Linear(hidden_size, hidden_size), nn.Dropout(dropout), activation_fn[activation]]
    return nn.Sequential(*layers)


def makedirs(path: str, isfile: bool = False) -> None:
    """
    Creates a directory given a path to either a directory or file.

    If a directory is provided, creates that directory.
    If a file is provided (i.e. isfile == True), creates the parent directory for that file.

    :param path: Path to a directory or file.
    :param isfile: Whether the provided path is a directory or file.
    """
    if isfile:
        path = os.path.dirname(path)
    if path != '':
        os.makedirs(path, exist_ok=True)


def load_weights(model: nn.Module, checkpoint_path: str) -> None:
    """
    Loads weights into a model.

    Ignores missing keys in the loaded state dict.

    :param model: The model into which weights will be loaded.
    :param checkpoint_path: Path to a .pt file containing weights to load.
    """
    model_state_dict = model.state_dict()
    loaded_state_dict = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model_state_dict.update(loaded_state_dict)
    model.load_state_dict(model_state_dict)


def save_model(model: nn.Module, save_path: str, remove_embeddings: bool = True) -> None:
    """
    Saves the weights of a model.

    :param model: The model to save.
    :param save_path: Path to .pt file where model will be saved.
    :param remove_embeddings: Whether to remove the embedding layer before saving.
    """
    makedirs(save_path, isfile=True)
    state = model.state_dict()
    # if remove_embeddings:
    #     del state['embedding.weight']
    torch.save(state, save_path)


def plot_alignment(alignment: torch.FloatTensor,
                   title: Optional[str] = None,
                   yticklabels: Optional[List[str]] = None,
                   xticklabels: Optional[List[str]] = None,
                   save_path: Optional[str] = None,
                   dpi: int = 150,
                   size: int = 10,
                   cmap=plt.cm.Blues):
    """
    Plots an alignment matrix.

    :param alignment: Alignment matrix (n x m).
    :param title: Plot title.
    :param yticklabels: Labels for the y axis ticks.
    :param xticklabels: Labels for the x axis ticks.
    :param save_path: Where to save the plot. If None, displays the plot.
    :param dpi: Dots per inch (quality).
    :param size: Font size.
    :param cmap: Colormap.
    """
    fig, ax = plt.subplots(dpi=dpi)
    ax.imshow(alignment, interpolation='nearest', cmap=cmap)

    ax.set(
        yticks=np.arange(alignment.size(0)),
        xticks=np.arange(alignment.size(1)),
        yticklabels=yticklabels or np.arange(alignment.size(0)) + 1,
        xticklabels=xticklabels or np.arange(alignment.size(1)) + 1,
        title=title
    )

    thresh = alignment.max() / 2
    for i in range(alignment.size(0)):
        for j in range(alignment.size(1)):
            ax.text(j, i, format(alignment[i, j], '.2f'),
                    ha='center', va='center',
                    color='white' if alignment[i, j] > thresh else 'black',
                    size=size)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def prod(iterable: Iterable) -> float:
    """Returns the product of the elements of an iterable."""
    return reduce(operator.mul, iterable, 1)


def find_unique_name(name: str) -> str:
    """Finds a unique name by appending numbers, ex. "name" -> "name_(1)"."""
    i = 1
    original_name = name

    while os.path.exists(name):
        name = f'{original_name}_({i})'
        i += 1

    return name


def create_exp_name(flags_and_values: List[str],
                    flag_skip_set: Optional[Set[str]] = None,
                    skip_paths: bool = False) -> str:
    """
    Creates an experiment name based on the command line arguments (besides dataset and paths).

    Example: "--dataset multinews --sinkhorn" --> "dataset=multinews_sinkhorn"

    :param flags_and_values: The command line flags and their values.
    :param flag_skip_set: A set of flags to skip.
    :param skip_paths: Whether to skip paths (i.e. any flag with a value with a "/" in it).
    :return: An experiment name created based on the command line arguments.
    """
    # Remove "-" from flags
    flag_skip_set = flag_skip_set or set()
    flag_skip_set = {flag.lstrip('-') for flag in flag_skip_set}

    # Extract flags and values, skipping where necessary
    args = {}
    current_flag = None
    for flag_or_value in flags_and_values:
        if flag_or_value.startswith('-'):
            flag = flag_or_value.lstrip('-')
            current_flag = flag if flag not in flag_skip_set else None
            if current_flag is not None:
                args[current_flag] = []
        elif current_flag is not None:
            args[current_flag].append(flag_or_value)

    # Handle paths
    if skip_paths:
        for key, values in list(args.items()):
            if any('/' in value for value in values):
                del args[key]

    # Handle boolean flags
    for key, values in args.items():
        if len(values) == 0:
            values.append('True')

    exp_name = '_'.join(f'{key}={"_".join(values)}' for key, values in args.items())

    if exp_name == '':
        exp_name = 'default'

    return exp_name


class Logger:
    def __init__(self, pipe, log_path: str):
        self.pipe = pipe
        self.log = open(log_path, 'w')

    def write(self, message: str):
        self.pipe.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.pipe.flush()
        self.log.flush()


def pad_tensors(tensor_list: List[torch.FloatTensor],
                padding: float = 0.0) -> torch.FloatTensor:
    """
    Pads a lists of tensors, each with the same number of dimensions.

    Extracts the dtype and device from the first tensor in the list.

    :param tensor_list: A list of FloatTensors to pad.
    :param padding: Padding value to use.
    :return: A FloatTensor containing the padded tensors in tensor_list.
    """
    shape_list = [tensor.shape for tensor in tensor_list]
    shape_max = torch.LongTensor(shape_list).max(dim=0)[0]

    tensor_batch = padding * torch.ones(len(tensor_list), *shape_max,
                                        dtype=tensor_list[0].dtype, device=tensor_list[0].device)

    for i, (tensor, shape) in enumerate(zip(tensor_list, shape_list)):
        tensor_slice = [i, *[slice(size) for size in shape]]
        tensor_batch[tensor_slice] = tensor

    return tensor_batch


def unpad_tensors(tensors: Union[torch.FloatTensor, List[torch.FloatTensor]],
                  shapes: List[Tuple[int, ...]]) -> List[torch.FloatTensor]:
    """
    Removes padding.

    :param tensors: List of tensors with padding.
    :param shapes: List of shapes without padding.
    :return: The same tensors but without padding.
    """
    assert len(tensors) == len(shapes)

    for i, (tensor, shape) in enumerate(zip(tensors, shapes)):
        if tensor.shape != shape:
            tensor_slice = [slice(size) for size in shape]
            tensors[i] = tensor[tensor_slice]

    return tensors



class NoamLR(_LRScheduler):
    """Learning rate schedule described in
    https://arxiv.org/pdf/1706.03762.pdf.

    Args:
        optimizer (Optimizer): Wrapped optimizer.

        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups

    """

    def __init__(self, optimizer, warmup_steps, model_size, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.model_size = model_size
        super(NoamLR, self).__init__(optimizer) #, last_epoch)


    def get_lr(self):
        step = self.last_epoch + 2
        lr = self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps**(-1.5))
        lr *= 2 #scaling factor
        # print(f'learning rate {lr} at step {step}')
        return [lr for group in self.optimizer.param_groups]


class Node:
    def __init__(self, text, node_type, level, parent_node):
        self.text = text
        self.node_type = node_type
        self.level = level
        self.children = []
        self.parent_node = parent_node

def build_tree(tree: Dict[str, Any], 
               level: int = 0, 
               parent_node: Node = None) -> Node:
    """Build the parse tree into a node tree data structure and return the root."""

    text = tree['word']
    node_type = tree['nodeType']
    node = Node(text, node_type, level, parent_node)

    if 'children' not in tree:
        return node

    for sub_tree in tree['children']:
        sub_node = build_tree(sub_tree, level + 1, node)
        node.children.append(sub_node)

    return node

def print_dfs_tree(node):
    print('Text: {:<50} | Type: {:<4} | Level: {}'.format(node.text, node.node_type, node.level))

    if not node.children:
        return

    for sub_node in node.children:
        print_dfs_tree(sub_node, encoded_target)