from typing import List, Tuple, Dict, Optional, ClassVar
from torch.utils.data import Dataset
from skspatial.objects import Point
from torch import Tensor
import torch

from utils import joint_groups_to_marker_labels, load_joint_group_names
from h2h_session_data import H2HSessionData


# TODO: Add masking with padding tokens for small sequences.
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
        sample_dtype: The data type to convert all the samples to.
        label_dtype: The data type to convert all the labels to.
        _sub_1_target_markers: The list of target marker labels for subject 1.
        _sub_2_target_markers: The list of target marker labels for subject 2.
        _object_target_markers: The list of target marker labels for the object.
        _samples: The tensor of input samples, shape (N, L, M * 3).
        _labels: The tensor of labels for each sample, shape (N, 2).
    """
    object_group: ClassVar[str] = 'object'
    sub_1_tag = H2HSessionData.sub_1_tag
    sub_2_tag = H2HSessionData.sub_2_tag
    session_files: List[str]
    sequence_length: int
    sample_dtype: torch.dtype
    label_dtype: torch.dtype
    _sub_1_target_markers: List[str]
    _sub_2_target_markers: List[str]
    _object_target_markers: List[str]
    _samples: Tensor
    _labels: Tensor

    def __init__(self, session_files: List[str], sequence_length: int, joint_groups_file: str,
                 joint_groups: Optional[List[str]] = None,
                 sample_dtype: Optional[torch.dtype] = torch.float32,
                 label_dtype: Optional[torch.dtype] = torch.long):
        """
        Creates a new Dataset object using the session files. Loads everything into memory, which may not be optimal
        for very large data sets.

        Args:
            session_files: The list of Matlab 7.3 files to load data from.
            sequence_length: The maximum length of our sequences of frames.
            sample_dtype: The data type to convert all the samples to.
            label_dtype: The data type to convert all the labels to.
            joint_groups_file: Path to the file specifying which markers are in which joint groups.
            joint_groups: List of which joint groups we want to include in the target markers.
        """
        self.session_files = session_files
        self.sequence_length = sequence_length
        self.sample_dtype = sample_dtype
        self.label_dtype = label_dtype
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
        self._samples, self._labels = self._split_sequences(sequences, labels)

    def _load_sequences(self) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Takes the trials from each Session and loads the marker data in as separate sequences, and splits each
        subject's markers into its own sequence. Formats each sequence as a tensor of shape (L, M * 3).
        The labels are which subject each sequence belongs to, with 0 = subject 1 and 1 = subject 2.

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
                # Append subject 1's mocap data.
                sequences.append(self._parse_markers(mocap_data, self._sub_1_target_markers))
                labels.append(0)
                # Append subject 2's mocap data.
                sequences.append(self._parse_markers(mocap_data, self._sub_2_target_markers))
                labels.append(1)
        # Reshape the outputs into tensors.
        return sequences, labels

    def _load_session_data(self, session_file: str) -> H2HSessionData:
        """
        Loads the session data from the session file into an H2HSessionData object. Also crops each trial to exclude
        nan markers, and cops the end of each trial to the time of handover.

        Args:
            session_file: Path to the session data file.

        Returns:
            H2HSessionData object.
        """
        target_markers = self._sub_1_target_markers + self._sub_2_target_markers + self._object_target_markers
        session_data = H2HSessionData(session_file=session_file, target_markers=target_markers)
        session_data.load()
        session_data.crop_nan()
        session_data.drop_nan()
        return session_data

    def _parse_markers(self, mocap_data: Dict[str, List[Tuple[float, Point]]], target_markers: List[str]) -> Tensor:
        """
        Parses the marker data corresponding to the given target markers (in order) from the given trial data.
        Reshapes it into a tensor of shape (L, M * 3), by putting frames first, and markers second.
        The markers are ordered according to the sorted joints, then the sorted order of the marker names.
        The 3D points for each marker are flattened into one dimension M * 3.

        Args:
            mocap_data: The dict of marker positions for each marker label throughout the trial.
            target_markers: The list of target markers.

        Returns:
            A tensor of shape (L, M * 3), which represents a sequence the length of the trial with the markers data
            for each joint in sorted order.
        """
        # Set the output to a tensor of shape (L, M * 3).
        L = len(mocap_data[target_markers[0]])
        M = len(target_markers)
        output = torch.zeros((L, M * 3), dtype=self.sample_dtype)
        # Parse the output from the mocap data according to the target markers.
        for frame in range(L):
            for i, marker in enumerate(target_markers):
                # Concatenate the point for this marker to the data dimension for this frame.
                point = mocap_data[marker][frame][1]
                j = i * 3
                output[frame][j: j + 3] = torch.tensor(point, dtype=self.sample_dtype)
        return output

    def _split_sequences(self, sequences: Tensor, sequence_labels: Tensor):
        """
        Splits the sequences corresponding to entire trials into ones of length self.sequence_length, and using a
        sliding window method to generate multiple sequences for each trial. Inflates the labels to correspond to
        these changes.

        Args:
            sequences: A tensor of shape (N, L, M * 3), with two sequences per trail for all the sessions.
            sequence_labels: A tensor of shape (N, 2), with a label for each sequence.

        Modifies the private attributes:
            - self._samples
            - self._labels
        """
        # Prepare the outputs.
        samples = []
        labels = []
        # Iterate through each sequence.
        for i in range(len(sequences)):
            sequence = sequences[i]
            label = sequence_labels[i]
            # Can't apply sliding window if sequence is too small.
            if sequence.shape[0] <= self.sequence_length:
                samples.append(sequence)
                labels.append(label)
            else:
                # Apply a sliding window to the sequence.
                for j in range(0, sequence.shape[0] - self.sequence_length):
                    samples.append(sequence[j: j + self.sequence_length])
                    labels.append(label)
        # Convert the outputs into tensors.
        samples_tensor = torch.zeros((len(samples), *samples[0].shape), dtype=self.sample_dtype)
        labels_tensor = torch.zeros((len(labels),), dtype=self.label_dtype)
        for i in range(len(samples)):
            samples_tensor[i] = samples[i]
            labels_tensor[i] = labels[i]
        return samples_tensor, labels_tensor

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


def short_test():
    """
    Short test for H2HSubjectsDataset.
    """
    session_files = ['E:/Datasets/CS 4440 Final Project/mat_files_full/test_data.mat']
    sequence_length = 5
    test(session_files, sequence_length)


def long_test():
    """
    Short test for H2HSubjectsDataset.
    """
    session_files = ['E:/Datasets/CS 4440 Final Project/mat_files_full/test_data.mat',
                     'E:/Datasets/CS 4440 Final Project/mat_files_full/26_M4_F5_cropped_data_v2.mat']
    sequence_length = 10
    test(session_files, sequence_length)


def test(session_files, sequence_length):
    """
    Runs a test for the specified sessions and sequence length.

    Args:
        session_files: The list of session files to load data from.
        sequence_length: The length of each sequence in the session.
    """
    joint_groups_file = 'joint_groups.json'
    joint_groups = ['spine', 'left arm', 'right arm', 'left leg', 'right leg']
    dataset = H2HSubjectsDataset(session_files, sequence_length, joint_groups_file, joint_groups)
    sample, label = dataset[0]
    sample_shape, label_shape = sample.shape, label.shape
    for i in range(len(dataset)):
        sample, label = dataset[i]
        if sample.shape != sample_shape or label.shape != label.shape:
            print(f'Found sample {i} with different shape!')
    print(f'Num samples: {len(dataset)}')
    print(f'sample shape, label shape: {sample_shape}, {label_shape}')
    print(f'sample type, label type: {sample.dtype}, {label.dtype}')
    is_nan = h2h_subjects_dataset_isnan(dataset)
    print(f'is_nan: {is_nan}')



def h2h_subjects_dataset_isnan(dataset: H2HSubjectsDataset) -> bool:
    """
    Checks if an H2HSubjectsDataset object contains any NaN values.

    Args:
        dataset: The dataloader to be checked.

    Returns:
        True if there were nans, False if there were not.
    """
    output = False
    for sample in range(len(dataset)):
        x, label = dataset[sample]
        # Manually check x.
        for frame in range(x.shape[0]):
            for i in range(x[frame].shape[0]):
                value = x[frame][i]
                if torch.isnan(value):
                    print(f'NaN found in x, sample: {sample}, frame: {frame}, i: {i}')
                    output = True
        # Manually check label.
        if torch.isnan(label):
            print(f'NaN found in label: sample: {sample}')
    # Return False when we have checked everything.
    return output


if __name__ == '__main__':
    short_test()
    long_test()
