from typing import List, Tuple, Optional
from torch.utils.data import Dataset
from torch import Tensor

from h2h_session_data import H2HSessionData


# TODO: Add joint group maps that can be shared with the model, to ensure correct unpacking of samples.
# TODO: Get a serializable format to save this to and later load from on demand (TFRecord is a candidate).


class H2HSubjectsDataset(Dataset):
    """Represents a dataset of H2H mocap sequences split between subject 1 and subject 2 and labeled with which subject
    they belong to.

    We split the markers for subject 1 and subject 2 into separate sequences, and label them with 0 for subject 1
    and 1 for subject 2. We have a max length for each sequence of frames, and use a sliding window approach to expand
    one handover motion into multiple sequences.

    Attributes:
        session_files: The list of session file paths to build the dataset from.
        sequence_length: The maximum length of a sequence of frames.
        target_markers: The list of markers we want to include in the dataset. None means all will be included.
        _samples: The tensor of input samples, shape (N, L, J, 3).
        _labels: The tensor of labels for each sample, shape (N, 1).
    """
    session_files: List[str]
    sequence_length: int
    target_markers: List[str]
    _samples: Tensor
    _labels: Tensor

    def __init__(self, session_files: List[str], sequence_length: int, target_markers: Optional[List[str]] = None):
        """Creates a new Dataset object using the session files. Loads everything into memory, which may not be optimal
        for very large data sets.

        Args:
            session_files: The list of Matlab 7.3 files to load data from.
            sequence_length: The maximum length of our sequences of frames.
            target_markers: The list of markers we want to include in the dataset. None means all will be included.
        """
        self.session_files = session_files
        self.sequence_length = sequence_length
        self.target_markers = target_markers
        # Get a list of input sequences for each trial, "flattening" all of the sessions.
        sequences, labels = self._load_sequences()
        # Split the trail-length sequences with a sliding window according to our max sequence length,
        # and re-assign the labels to match the new shape.
        self._split_sequences(sequences, labels)

    def _load_sequences(self) -> Tuple[Tensor, Tensor]:
        """Takes the trials from each Session and loads the marker data in as separate sequences, and splits each
        subject's markers into its own sequence. Formats each sequence as a tensor of shape (L, J, 3). The labels are
        which subject each sequence belongs to, with subject 1 having a label of 0 and subject 2 a label of 1.

        Returns:
            Sequences, a tensor of shape (N, L, J, 3), with two sequences per trail for all of the sessions.
            Labels, a tensor of shape (N, 1), with a label for each sequence.
        """
        pass

    def _split_sequences(self, sequences: Tensor, labels: Tensor):
        """Splits the sequences corresponding to entire trials into ones of length self.sequence_length, and using a
        sliding window method to generate multiple sequences for each trial. Inflates the labels to correspond to
        these changes.

        Args:
            sequences: A tensor of shape (N, L, J, 3), with two sequences per trail for all of the sessions.
            labels: A tensor of shape (N, 1), with a label for each sequence.

        Modifies the private attributes:
            - self._samples
            - self._labels
        """
        pass

    def __len__(self) -> int:
        """Gets the length of this Dataset. This is the total number of sequences (N).

        Returns:
            The number of sequences in the dataset.
        """
        return self._samples.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Returns the sample and label at index idx out of N.

        Args:
            idx: The index of the sample we want to retrieve

        Returns:
            Sample, the (L, J, 3) shaped Tensor representing a motion capture sequence.
            Label, the (1) shaped Tensor representing the label for the sample.
        """
        return self._samples[idx], self._labels[idx]
