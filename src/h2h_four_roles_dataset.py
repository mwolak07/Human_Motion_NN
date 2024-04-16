from typing import override, Tuple, List
from torch import Tensor

from h2h_subjects_dataset import H2HSubjectsDataset, test


class H2HFourRolesDataset(H2HSubjectsDataset):
    """
    Represents a dataset of H2H mocap sequences split between subject 1 and subject 2 labeled with each subject's role
    as initiator giver, initiator receiver, follower giver, or follower receiver.

    Notes:
    - We split the markers for subject 1 and subject 2 into separate sequences.
    - Labels are 0 for IG and 1 for IR, 2 for FG, and 3 for FR.
    - We have a sequence length, and use a sliding window approach to expand one handover motion into multiple samples.

    Attributes:
        object_group: (class attribute) The name of the joint group for the object.
        sub_1_tag: (class attribute) The preceding "tag" for subject 1's markers.
        sub_2_tag: (class attribute) The preceding "tag" for subject 2's markers.
        session_files: The list of session file paths to build the dataset from.
        sequence_length: The maximum length of a sequence of frames.
        sample_dtype: The data type to convert all the samples to.
        label_dtype: The data type to convert all the labels to.
        _sub_1_target_markers: The list of target marker labels for subject 1.
        _sub_2_target_markers: The list of target marker labels for subject 2.
        _object_target_markers: The list of target marker labels for the object.
        _samples: The tensor of input samples, shape (N, L, M * 3).
        _labels: The tensor of labels for each sample, shape (N, 2).
    """
    @override
    def _load_sequences(self) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Takes each session and loads the mocap sequences from each trial, with subject 1 and subject 2's markers split
        into separate sequences. Labels each sequence according to which each subject's role.

        Notes:
            - Each sequence is a tensor of shape (L, M * 3), where L is length and M is number of markers.
            - Each sequence has a label, 0 = IG and 1 = IR, 2 = FG, and 3 = FR.

        Returns:
            Sequences, a tensor of shape (N, L, M * 3), with two sequences per trail for all the sessions.
            Labels, a tensor of shape (N, 2), with a label for each sequence.
        """
        # Prepare the outputs.
        sequences = []
        labels = []
        # Iterate through all the session files.
        for session_file in self.session_files:
            # Iterating though each trial in the session.
            session_data = self._load_session_data(session_file)
            for trial in session_data.trials():
                mocap_data = session_data[trial]['mocap']
                role_data = session_data[trial]['role']
                # Append subject 1's mocap data.
                sequences.append(self._parse_markers(mocap_data, self._sub_1_target_markers))
                labels.append(self._parse_label(role_data, 1))
                # Append subject 2's mocap data.
                sequences.append(self._parse_markers(mocap_data, self._sub_2_target_markers))
                labels.append(self._parse_label(role_data, 2))
        # Reshape the outputs into tensors.
        return sequences, labels

    @staticmethod
    def _parse_label(role_data: str, subject: int) -> int:
        """
        Gets the label for the given subject based on the trial role. 0 = IG and 1 = IR, 2 = FG, and 3 = FR.
        Dataset roles are:
        - Sub1 initiator giver
        - Sub1 initiator receiver
        - Sub2 initiator giver
        - Sub2 initiator receiver

        Args:
            role_data: The string representing the role data for the trial.
            subject: The subject we are getting the label for.

        Returns:
            0 for IG and 1 for IR, 2 for FG, and 3 for FR.

        Raises:
            ValueError: The subject isn't one of (1, 2).
                        The role isn't one of ('Sub1_IG', 'Sub1_IR', 'Sub2_IG', 'Sub2_IR').
        """
        if subject == 1:
            if role_data == 'Sub1_IG':
                return 0
            if role_data == 'Sub1_IR':
                return 1
            if role_data == 'Sub2_IR':
                return 2
            if role_data == 'Sub2_IG':
                return 3
            else:
                raise ValueError(f'The role {role_data} is not one of "Sub1_IG", "Sub1_IR", "Sub2_IG", "Sub2_IR".')
        if subject == 2:
            if role_data == 'Sub2_IG':
                return 0
            if role_data == 'Sub2_IR':
                return 1
            if role_data == 'Sub1_IR':
                return 2
            if role_data == 'Sub1_IG':
                return 3
            else:
                raise ValueError(f'The role {role_data} is not one of "Sub1_IG", "Sub1_IR", "Sub2_IG", "Sub2_IR".')
        else:
            raise ValueError(f'The subject {subject} is not one of (1, 2).')


def main() -> None:
    """
    Main function that runs when this file is invoked, to test this implementation.
    """
    short_test()
    long_test()


def short_test() -> None:
    """
    Short test for H2HSubjectsDataset.
    """
    session_files = ['E:/Datasets/CS 4440 Final Project/mat_files_full/test_data.mat']
    sequence_length = 5
    joint_groups_file = 'joint_groups.json'
    joint_groups = ['spine', 'left arm', 'right arm', 'left leg', 'right leg']
    dataset = H2HFourRolesDataset(session_files, sequence_length, joint_groups_file, joint_groups)
    test(dataset)


def long_test() -> None:
    """
    Short test for H2HSubjectsDataset.
    """
    session_files = ['E:/Datasets/CS 4440 Final Project/mat_files_full/test_data.mat',
                     'E:/Datasets/CS 4440 Final Project/mat_files_full/26_M4_F5_cropped_data_v2.mat']
    sequence_length = 10
    joint_groups_file = 'joint_groups.json'
    joint_groups = ['spine', 'left arm', 'right arm', 'left leg', 'right leg']
    dataset = H2HFourRolesDataset(session_files, sequence_length, joint_groups_file, joint_groups)
    test(dataset)


if __name__ == '__main__':
    main()
