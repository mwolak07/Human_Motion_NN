from typing import Tuple, List, Optional
import json


def joint_groups_to_target_markers(joint_groups_file: str, sub_1_tag: str, sub_2_tag: str,
                                   joint_groups: Optional[List[str]] = None) -> Tuple[List[str], List[str], List[str]]:
    """

    Args:
        joint_groups_file: Path to the file specifying which markers are in which joint groups.
        sub_1_tag: Preceding "tag" for subject 1's markers.
        sub_2_tag: Preceding "tag" for subject 2's markers.
        joint_groups: List of which joint groups we want to include in the target markers. If this is None, we load all
                      the joint groups from the json file.

    Returns:
        List of marker labels for subject 1.
        List of marker labels for subject 2.
        List of marker labels for the object.
    """
    # Prepare the output lists.
    sub_1_markers = []
    sub_2_markers = []
    # Read the joint groups dict from the file.
    with open(joint_groups_file, 'r') as f:
        joint_groups_map = json.load(f)
    # If joint_groups is None, make it the keys of joint_groups_map.
    joint_groups = list(joint_groups_map.keys()) if joint_groups is None else joint_groups
    # Get the list of subject and object markers from the joint groups.
    sub_markers = []
    object_markers = []
    for group in joint_groups:
        if group in ['object']:
            object_markers += joint_groups_map[group]
        else:
            sub_markers += joint_groups_map[group]
    # Expand the subject markers into those for subject 1 and subject 2.
    for marker in sub_markers:
        sub_1_markers.append(sub_1_tag + marker)
        sub_2_markers.append(sub_2_tag + marker)
    # Return our result.
    return sub_1_markers, sub_2_markers, object_markers
