from typing import List, Tuple, Dict, Optional, ClassVar
from torch.utils.data import Dataset
from skspatial.objects import Point
from torch import Tensor
import torch

from utils import joint_groups_to_marker_labels, load_joint_group_names
from h2h_session_data import H2HSessionData


# TODO: Add joint group maps that can be shared with the model, to ensure correct unpacking of samples.
# TODO: Get a serializable format to save this to and later load from on demand (TFRecord is a candidate).


class H2HSubjectsDataset(Dataset):
    """
    Represents a dataset of H2H mocap sequences split between subject 1 and subject 2 and labeled with which subject
    they belong to.

    We split the markers for subject 1 and subject 2 into separate sequences, and label them with 0 for subject 1
    and 1 for subject 2. We have a max length for each sequence of frames, and use a sliding window approach to expand
    one handover motion into multiple sequences.

    Attributes:
        object_group: (class attribute) The name of the joint group for the object.
        sub_1_tag: (class attribute) The preceding "tag" for subject 1's markers.
        sub_2_tag: (class attribute) The preceding "tag" for subject 2's markers.
        session_files: The list of session file paths to build the dataset from.
        sequence_length: The maximum length of a sequence of frames.
        _sub_1_target_markers: The list of target marker labels for subject 1.
        _sub_2_target_markers: The list of target marker labels for subject 2.
        _object_target_markers: The list of target marker labels for the object.
        _samples: The tensor of input samples, shape (N, (L, M, 3).
        _labels: The tensor of labels for each sample, shape (N, 1).
    """
    object_group: ClassVar[str] = 'object'
    sub_1_tag = H2HSessionData.sub_1_tag
    sub_2_tag = H2HSessionData.sub_2_tag
    session_files: List[str]
    sequence_length: int
    _sub_1_target_markers: List[str]
    _sub_2_target_markers: List[str]
    _object_target_markers: List[str]
    _samples: Tensor
    _labels: Tensor

    def __init__(self, session_files: List[str], sequence_length: int, joint_groups_file: str,
                 joint_groups: Optional[List[str]] = None):
        """
        Creates a new Dataset object using the session files. Loads everything into memory, which may not be optimal
        for very large data sets.

        Args:
            session_files: The list of Matlab 7.3 files to load data from.
            sequence_length: The maximum length of our sequences of frames.
            joint_groups_file: Path to the file specifying which markers are in which joint groups.
            joint_groups: List of which joint groups we want to include in the target markers.
        """
        self.session_files = session_files
        self.sequence_length = sequence_length
        # Set the joint groups to everything in the file if it is None.
        if joint_groups is None:
            joint_groups = load_joint_group_names(joint_groups_file)
        # Pop the object group from the list for the subjects if needed.
        subject_joint_groups = joint_groups.copy()
        if self.object_group in joint_groups:
            subject_joint_groups.remove(self.object_group)
        # Get the list of target markers for each subject.
        self._sub_1_target_markers = joint_groups_to_marker_labels(joint_groups_file, subject_joint_groups,
                                                                   self.sub_1_tag)
        self._sub_2_target_markers = joint_groups_to_marker_labels(joint_groups_file, subject_joint_groups,
                                                                   self.sub_2_tag)
        # Determine if the object is part of the loaded joint_groups, and get the object target markers if so.
        if self.object_group in load_joint_group_names(joint_groups_file):
            self._object_target_markers = joint_groups_to_marker_labels(joint_groups_file, [self.object_group])
        else:
            self._object_target_markers = []
        # Get a list of input sequences for each trial, "flattening" all the sessions.
        sequences, labels = self._load_sequences()
        # Split the trail-length sequences with a sliding window according to our max sequence length,
        # and re-assign the labels to match the new shape.
        self._split_sequences(sequences, labels)

    def _load_sequences(self) -> Tuple[Tensor, Tensor]:
        """
        Takes the trials from each Session and loads the marker data in as separate sequences, and splits each
        subject's markers into its own sequence. Formats each sequence as a tensor of shape (L, M, 3). The labels are
        which subject each sequence belongs to, with subject 1 having a label of 0 and subject 2 a label of 1.

        Returns:
            Sequences, a tensor of shape (N, (L, M, 3), with two sequences per trail for all the sessions.
            Labels, a tensor of shape (N, 1), with a label for each sequence.
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
                # Append subject 1's mocap data.
                sequences.append(self._parse_markers(mocap_data, self._sub_1_target_markers))
                labels.append(0)
                # Append subject 2's mocap data.
                sequences.append(self._parse_markers(mocap_data, self._sub_2_target_markers))
                labels.append(1)
        # Reshape the outputs into tensors.
        return torch.tensor(sequences, dtype=torch.float64), torch.tensor(labels, dtype=torch.float64)

    @staticmethod
    def _load_session_data(session_file: str) -> H2HSessionData:
        """
        Loads the session data from the session file into an H2HSessionData object. Also crops each trial to exclude
        nan markers, and cops the end of each trial to the time of handover.

        Args:
            session_file: Path to the session data file.

        Returns:
            H2HSessionData object.
        """
        session_data = H2HSessionData(session_file)
        session_data.load()
        session_data.crop_nan()
        session_data.crop_to_handover()
        return session_data

    @staticmethod
    def _parse_markers(mocap_data: Dict[str, List[Tuple[float, Point]]], target_markers: List[str]) -> Tensor:
        """
        Parses the marker data corresponding to the given target markers (in order) from the given trial data.
        Reshapes it into a tensor of shape (L, M, 3), by putting frames first, and markers second.
        The markers are ordered according to the sorted joints, then the sorted order of the marker names.

        Args:
            mocap_data: The dict of marker positions for each marker label throughout the trial.
            target_markers: The list of target markers.

        Returns:
            A tensor of shape (L, M, 3), which represents a sequence the length of the trial with the markers data
            for each joint in sorted order.
        """
        # Set the output to a tensor of shape (L, M, 3).
        L = len(mocap_data[target_markers[0]])
        M = len(target_markers)
        output = torch.zeros((L, M, 3))
        # Parse the output from the mocap data according to the target markers.
        for frame in range(L):
            for m, marker in enumerate(target_markers):
                # Set each cell of the output to the mocap data point, stripping the timestamp.
                output[frame][m] = torch.tensor(mocap_data[marker][frame][1], dtype=torch.float64)
        return output

    def _split_sequences(self, sequences: Tensor, sequence_labels: Tensor):
        """
        Splits the sequences corresponding to entire trials into ones of length self.sequence_length, and using a
        sliding window method to generate multiple sequences for each trial. Inflates the labels to correspond to
        these changes.

        Args:
            sequences: A tensor of shape (N, (L, M, 3), with two sequences per trail for all the sessions.
            sequence_labels: A tensor of shape (N, 1), with a label for each sequence.

        Modifies the private attributes:
            - self._samples
            - self._labels
        """
        # Prepare the outputs.
        samples = []
        labels = []
        # Iterate through each sequence.
        for i in sequences.shape[0]:
            sequence = sequences[i, :, :, :]
            label = sequence_labels[i, :]
            # Can't apply sliding window if sequence is too small.
            if self.sequence_length <= sequence.shape[0]:
                samples.append(sequence)
            else:
                # Apply a sliding window to the sequence.
                for j in range(0, sequence.shape[0] - self.sequence_length, 1):
                    samples.append(sequence[j: j + self.sequence_length, :, :])
                    labels.append(label)
        # Reshape the outputs into tensors.
        return torch.tensor(samples, dtype=torch.float64), torch.tensor(labels, dtype=torch.float64)

    def __len__(self) -> int:
        """
        Gets the length of this Dataset. This is the total number of sequences (N).

        Returns:
            The number of sequences in the dataset.
        """
        return self._samples.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Returns the sample and label at index idx out of N.

        Args:
            idx: The index of the sample we want to retrieve

        Returns:
            Sample, the (L, J, 3) shaped Tensor representing a motion capture sequence.
            Label, the (1) shaped Tensor representing the label for the sample.
        """
        return self._samples[idx], self._labels[idx]
