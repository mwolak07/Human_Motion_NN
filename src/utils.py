from typing import List, Optional
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
    return list(joint_map.keys()).sort()
