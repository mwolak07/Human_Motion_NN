from typing import List, Dict, Tuple, Optional, Any, Callable
from torch.utils.data import Dataset, Subset
import numpy as np
import torch
import json


def joint_groups_to_marker_labels(joint_groups_file: str, joint_groups: List[str], prefix: Optional[str] = None) \
        -> List[str]:
    """
    Reads the joint groups from the file, and outputs the list of marker labels making up those joint groups.
    This "flattens" the markers across the joint groups.

    For consistency, we sort the list of joint groups.

    Args:
        joint_groups_file: Path to the file specifying which markers are in which joint groups.
        joint_groups: List of which joint groups we want to include in the target markers.
        prefix: Optional prefix to add to each marker label.

    Returns:
        List of marker labels.
    """
    # Prepare the output list.
    marker_labels = []
    # Read the joint groups dict from the file.
    with open(joint_groups_file, 'r') as f:
        joint_map = json.load(f)
    # If joint_groups is None, make it the keys of joint_groups_map.
    joint_groups = list(joint_map.keys()) if joint_groups is None else joint_groups
    # Sort the joint group names.
    joint_groups.sort()
    # Build the list of marker labels.
    for group in joint_groups:
        group_marker_labels = joint_map[group]
        # Add the prefix if desired.
        if prefix is not None:
            group_marker_labels = [prefix + marker_label for marker_label in group_marker_labels]
        # Add the marker labels for this group to the output.
        marker_labels += group_marker_labels
    # Return our output.
    return marker_labels


def load_joint_group_names(joint_groups_file: str) -> List[str]:
    """
    Loads the joint group names from the json file, which is just the keys of the loaded map.

    Args:
        joint_groups_file: Path to the file specifying which marker labels are in which joints.

    Returns:
        A list of tje joint group names, in sorted order.
    """
    # Read the joint groups dict from the file.
    with open(joint_groups_file, 'r') as f:
        joint_map = json.load(f)
    # Return the keys as a sorted list.
    group_names = list(joint_map.keys())
    group_names.sort()
    return group_names


def data_split(dataset: Dataset, test_split: Optional[float] = None, val_split: Optional[float] = None,
               split_map: Optional[Dict[str, List[int]]] = None) \
        -> Tuple[Dataset, Dataset, Dataset, Dict[str, List[int]]]:
    """
    Splits the given dataset into training, validation, and test sets. Also returns a map of the indices it split the
    dataset along.
    Can either do a random split according to proportions, or go along specified indices (same format as is returned).

    Proportions:
        - Say test_split is 20% and val_split is 20%.
        - First, 20% of the raw dataset is split off for testing.
        - Next, 20% of the remaining dataset is split off for validation.
        - Finally, the rest of the dataset is used for training.
    Specified:
        - split_map must have 'train', 'val', and 'test' keys mapped to lists of indices.

    Args:
        dataset: The raw dataset to be split up.
        test_split: The proportion of the dataset to use for testing.
        val_split: The proportion of the training dataset used for validation.
        split_map: The mapping of indices for train, validation, and test splits.

    Returns:
        The training dataset.
        The validation dataset.
        The testing dataset.
        The split_map used to perform the split (always returned).

    Raises:
        ValueError: Neither the test/val split nor the split_map are specified.
                    Both the test/val split and split_map are specified.
    """
    # Check that the values are valid.
    if test_split is None and val_split is None and split_map is None:
        raise ValueError('Either test/val split proportions or split_map must be specified.')
    if test_split is not None and val_split is not None and split_map is not None:
        raise ValueError('Test/val split proportions or split_map cannot both be specified.')
    # Generate a split_map if proportions are specified.
    if split_map is None:
        split_map = _generate_split_map(len(dataset), test_split, val_split)
    # Split the dataset using the split_map.
    train_set = Subset(dataset, split_map['train'])
    val_set = Subset(dataset, split_map['val'])
    test_set = Subset(dataset, split_map['test'])
    # Return our results.
    return train_set, val_set, test_set, split_map


def _generate_split_map(n: int, test_split: float, val_split: float) -> Dict[str, List[int]]:
    """
    Generates the split_map used to perform the splits in data_split using the given dataset size and proportions.

    Args:
        n: The size of the dataset.
        test_split: The proportion of the dataset to use for testing.
        val_split: The proportion of the training dataset used for validation.

    Returns:
        The mapping of indices for train, validation, and test splits.
    """
    # Initialize the split_map.
    split_map = {'train': [], 'val': [], 'test': []}
    # Translate the proportions into numbers.
    n_test = round(n * test_split)
    n_val = round((n - n_test) * val_split)
    # Shuffle the indices and split them among each set.
    indices = np.arange(0, n)
    np.random.shuffle(indices)
    # Assign each set's indices in the split_map.
    split_1 = n_test
    split_2 = n_test + n_val
    split_map['test'] = indices[: split_1]
    split_map['val'] = indices[split_1: split_2]
    split_map['train'] = indices[split_2:]
    # DEBUG, check size.  # TODO: Remove this after testing.
    if len(split_map['test']) + len(split_map['val']) + len(split_map['train']) != len(indices):
        print(f'PROBLEM WITH SPLIT MAP AHHHH!')
    # Return our result
    return split_map


def txt_list_append(file_path: str, value: Any) -> None:
    """
    Appends the given value to the list stored in the given text file.

    Args:
        file_path: Path to the text file.
        value: Value to append.
    """
    with open(file_path, 'a+') as f:
        f.write(f'({value})\n')


def txt_list_read(file_path: str, type_func: Optional[Callable[[str], Any]] = None) -> List[Any]:
    """
    Reads the list stored in the given text file.

    Args:
        file_path: Path to the text file.
        type_func: Function to convert the strings stored in the txt file to the desired type.

    Returns:
        The list of values stored in the file.
    """
    with open(file_path, 'r') as f:
        out_list = f.read().split('\n')[:-1]
    if type_func is not None:
        out_list = [type_func(value) for value in out_list]
    return out_list


def show_grads(model) -> None:
    """
    Prints the gradients of the parameters of the given model. Useful for debugging.

    Args:
        model: The model to view the gradients of.
    """
    for name, param in model.named_parameters():
        print(f'{name}\n{param.grad}')


def dataloader_isnan(dataloader) -> bool:
    """
    Checks if the dataloader contains any NaN values.

    Args:
        dataloader: The dataloader to be checked.

    Returns:
        True if there were nans, False if there were not.
    """
    output = False
    for batch, (x, label) in enumerate(dataloader):
        # Manually check x.
        for sequence in range(x.shape[0]):
            for frame in range(x[sequence].shape[0]):
                for i in range(x[sequence][frame].shape[0]):
                    value = x[sequence][frame][i]
                    if torch.isnan(value):
                        print(f'NaN found in x at batch: {batch}, sequence: {sequence}, frame: {frame}, value: {value}')
                        output = True
        # Manually check label.
        for sequence in range(label.shape[0]):
            if torch.isnan(label[sequence]):
                print(f'NaN found in label at batch: {batch}, sequence: {sequence}')
                output = True
    return output
