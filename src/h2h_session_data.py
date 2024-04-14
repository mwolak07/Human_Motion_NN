from typing import ClassVar, List, Dict, Tuple, Optional, Iterator, Set, Any
from collections.abc import Sequence
from skspatial.objects import Point
import numpy as np
import mat73


# TODO: More intelligent dropping of NaN values.
# TODO: Integrate a more flexible reader that can read only specific trials.
# TODO: Load in ALL data from the matlab file into this object. Including GAZE.
# TODO: Replace lists with numpy fixed memory allocation.


class H2HSessionData(Sequence):
    """Represents a fast and convenient Python interface for the H2H session data, stored in Matlab 7.3 files.
    Each H2HSessionData object corresponds to one session, and contains the data gathered for each trial, except
    for images and video. (Currently, we only include mocap).
    Handover time is also provided, computed from the velocity profile of the follower wrist marker.

    The object is accessed as a sequence of trials. Note the original files index starting with trial 1, and we
    continue that here - there is no trial 0.

    Each element of the sequence looks as follows:
    H2HSessionData[trial] = {
        'mocap': {marker: [(timestamp, point)]},
        'role': str,
        'handover': float
    }

    The mocap will only have the markers in the target_markers dict, if it was provided. All of them will be there
    otherwise.

    The object also supports the len() function, and can be used as an iterator in loops.

    Attributes:
        mocap_fps: (class attribute) The framerate the mocap data was captured at.
        wrist_marker_name: (class attribute) The name of the wrist marker, this is always loaded to compute handover.
        sub_1_tag: (class attribute) The preceding "tag" for subject 1's markers.
        sub_2_tag: (class attribute) The preceding "tag" for subject 2's markers.
        session_file: The path to the Matlab 7.3 file containing the session's data.
        target_markers: The list of markers to include (same for each subject). (If None, include all).
        _trials: A list of trials in this session (1, 2, 3, etc.).
        _mocap_data: A dictionary of the mocap data for the current session.
            Structured: {trial: {subject: {marker: [(timestamp, point)]}}}.
        _wrist_data: A dictionary of the wrist marker data for the current session.
            Structured: {trial: {subject: [(timestamp, point)]}}.
        _role_data: A dictionary of the role data for the current session.
            Structured: {trial: role}.
        _handover_data: A dictionary of the computed handover data for the current session.
            Structured: {trial: timestamp}.
    """
    mocap_fps: ClassVar[float] = 100.0
    wrist_marker_name: ClassVar[str] = 'RUSP'
    sub_1_tag: ClassVar[str] = 'Sub1'
    sub_2_tag: ClassVar[str] = 'Sub2'
    _trials: List[int]
    _mocap_data: Dict[int, Dict[int, Dict[str, List[Tuple[float, Point]]]]]
    _wrist_data: Dict[int, Dict[int, List[Tuple[float, Point]]]]
    _role_data: Dict[int, str]
    _handover_data: Dict[int, float]

    def __init__(self, session_file: str, target_markers: Optional[List[str]] = None):
        """Initializes the object with the given session file and target markers.
        Does not read any data until .load() is called.

        Args:
            session_file: The path to the Matlab 7.3 file containing the session's data.
            target_markers: The list of markers to include (same for each subject). (If None, include all). The names
                            should have the subject tags stripped. For example, 'Sub1_FH' becomes 'FH'.
        """
        # Initialize public attributes.
        self.session_file = session_file
        self.target_markers = target_markers
        # Initialize private attributes.
        self._trials = None
        self._mocap_data = None
        self._wrist_data = None
        self._role_data = None
        self._handover_data = None

    def __len__(self) -> int:
        """Gets the number of trials in this session.

        Returns:
            The number of trials in this session.
        """
        return len(self._trials)

    def __getitem__(self, k: int) -> Dict[str, Any]:
        """Gets the data for trial k. Note the original files index starts with trial one, and this function follows
        the same paradigm, so there is no trial 0.

        Args:
            k: The trial we want to access, starting with 1.

        Returns:
            A dict with the data for each trial. Refer to the class documentation for the format.

        Raises:
            KeyError when a non-integer key, or one not in self._trials is entered.
        """
        # We check if k is an integer that is in self._trials.
        if not isinstance(k, int) or k not in self._trials:
            raise KeyError()
        # Construct each element of the trial data, accounting for missing data.
        mocap_data = self._getitem_data_copy(self._mocap_data, k)
        role_data = self._getitem_data_copy(self._role_data, k)
        handover_data = self._getitem_data_copy(self._handover_data, k)
        # Return a dict with all of the data.
        return {
            'mocap': mocap_data,
            'role': role_data,
            'handover': handover_data
        }

    @staticmethod
    def _getitem_data_copy(data: Optional[Any], k: int) -> Optional[Any]:
        """Gets the copy of the item at trail k for the given private attribute.
        Returns None for the item if the attribute is None.

        Args:
            data: An attribute of self, that might be None.
            k: The trial we want to get the data for.

        Returns:
            The data for the given trial for the given attribute, or None.
        """
        return data[k].copy() if data is not None else None

    def __iter__(self) -> Iterator[int]:
        """Gets an iterator over the __getitem__ keys, which are the trial numbers, in order.

        Returns:
            An iterator over the trial numbers.
        """
        return iter(self._trials)

    def load(self) -> None:
        """Loads the data from the session file to this object. Parses the data into the output format we want.

        Modifies the private attributes:
            - self._mocap_data
            - self._wrist_data
            - self._role_data
            - self._handover_data
        """
        # Read the session file.
        session_data = mat73.loadmat(self.session_file)
        # Parse out the data for the valid trials.
        self._trials = self._parse_trials(session_data)
        # Parse out the data for the target markers.
        self._mocap_data = self._parse_mocap(session_data)
        # Parse the wrist data for the wrist markers.
        self._wrist_data = self._parse_wrist(session_data)
        # Parse out the role data.
        self._role_data = self._parse_role(session_data)
        # Calculate the handover time using the wrist and role data.
        self._handover_data = self._get_handover()

    def _parse_trials(self, session_data: Dict[str, Any]) -> List[int]:
        """Parses the number of each trial from the data dict loaded from the session file.

        We derive this from the role data. It is important to take into account missing trials - some have an empty
        role, and these should be skipped.

        Args:
            session_data: The data dict loaded from the session file.

        Returns:
            The list of trials in this session.
        """
        # Get the role list.
        role_list = session_data['cropped_data']['role']
        # For each trial with a valid role, add it to the internal list of trials in this session.
        self._trials = []
        for i, role in enumerate(role_list):
            if role is not None:
                # Make sure to offset i by 1.
                self._trials.append(i + 1)

    def _parse_mocap(self, session_data: Dict[str, Any]) -> Dict[int, Dict[int, Dict[str, List[Tuple[float, Point]]]]]:
        """Parses the mocap data from the data dict loaded from the session file. We store the object as subject -1.

        Args:
            session_data: The data dict loaded from the session file.

        Returns:
            The mocap data, represented as: {trial: {subject: {marker: [(timestamp, Point)]}}}.
        """
        # Get a list of markers for each subject, if there are any.
        if self.target_markers is not None:
            sub_1_markers = [self.sub_1_tag + marker for marker in self.target_markers]
            sub_2_markers = [self.sub_2_tag + marker for marker in self.target_markers]
        # Get the list of mocap data for each trial.
        mocap_list = session_data['cropped_data']['cropped_mocap']
        # Parse the mocap data for each trial.
        self._mocap_data = {}
        for trial in self._trials:
            # self._trials uses trial number, trial index is trial - 1.
            i = trial - 1
            # Get the corresponding lists of marker names and marker points.
            label_list = mocap_list[i]['Labels']
            point_list = mocap_list[i]['Loc']
            # Parse all of the markers.
            if self.target_markers is None:
                self.target_markers = self._parse_all_markers(label_list)
            # Get the mocap data for each target marker for each subject.
            self._mocap_data[trial] = {}
            for j in range(len(label_list)):
                label = label_list[j]
                # Parse the markers for subject 1.
                if self.sub_1_tag in label:
                     =
                elif self.sub_2_tag in label:

                else:

            # Process the target markers.
            # Get the index of each target marker in the marker list based on the label list.
            sub_1_i = label_list.index(sub_1_wrist_marker)
            sub_2_i = label_list.index(sub_2_wrist_marker)
            # Parse the mocap data for the wrist marker for each subject.
            point_list = mocap_list[i]['Loc']
            self._wrist_data[trial] = {
                1: self._parse_mocap_frames(point_list[sub_1_i], False),
                2: self._parse_mocap_frames(point_list[sub_2_i], False)
            }

    def _get_target_labels(self, label_list: List[str]) -> Set[str]:
        """Parses the names of all of the markers from """

    def _parse_mocap_frames(self, mocap_frames: np.ndarray, drop_nan: bool) -> List[Tuple[float, Point]]:
        """Parses the numpy array of mocap frames for one marker.

        Args:
            - mocap_frames: A shape (n_frames, 3) numpy array of mocap points in each frame for this trial and marker.
            - drop_nan: If True, we drop any frames with NaN values in them.

        Returns:
            A list of pairs of (timestamp, Point).
        """
        point_list = []
        n_frames = 0
        for mocap_frame in mocap_frames:
            # If drop_nan is enabled, skip any frames with NaN values.
            if drop_nan and np.any(np.isnan(mocap_frame)):
                continue
            else:
                point_list.append(Point(mocap_frame))
                n_frames += 1
        timestamp_list = self._get_timestamps(n_frames, self.mocap_fps)
        return zip(timestamp_list, point_list)

    def _parse_wrist(self, session_data: Dict[str, Any]) -> Dict[int, Dict[int, List[Tuple[float, Point]]]]:
        """Parses the wrist data from the data dict loaded from the session file.

        Args:
            session_data: The data dict loaded from the session file.

        Returns:
            The mocap data, represented as: {trial: {subject: [(timestamp, Point)]}}.
        """
        # Get the names of each subject's wrist marker.
        sub_1_wrist_marker = self.sub_1_tag + self.wrist_marker_name
        sub_2_wrist_marker = self.sub_2_tag + self.wrist_marker_name
        # Get the list of mocap data for each trial.
        mocap_list = session_data['cropped_data']['cropped_mocap']
        # Get the wrist data for each trial.
        self._wrist_data = {}
        for trial in self._trials:
            # self._trials uses trial number, trial index is trial - 1.
            i = trial - 1
            # Get the index of each wrist marker in the marker list based on the label list.
            label_list = mocap_list[i]['Labels']
            sub_1_i = label_list.index(sub_1_wrist_marker)
            sub_2_i = label_list.index(sub_2_wrist_marker)
            # Parse the mocap data for the wrist marker for each subject.
            point_list = mocap_list[i]['Loc']
            self._wrist_data[trial] = {
                1: self._parse_mocap_frames(point_list[sub_1_i], False),
                2: self._parse_mocap_frames(point_list[sub_2_i], False)
            }

    def _parse_role(self, session_data: Dict[str, Any]) -> Dict[int, str]:
        """Parses the role data from the data dict loaded from the session file.

        Args:
            session_data: The data dict loaded from the session file.

        Returns:
            The mocap data, represented as: {trial: role}.
        """
        # Get the role list.
        role_list = session_data['cropped_data']['role']
        # For each trial, record the role.
        self._role_data = {}
        # For each trial with a valid role, add it to the internal list of trials in this session.
        for trial in self._trials:
            # self._trials uses trial number, trial index is trial - 1.
            i = trial - 1
            self._role_data[trial] = role_list[i]
            # DEBUG. Should not go off.
            if self._role_data[trial] is None:
                print(f'ALARM!!! INVALID TRIAL!!! {trial}')

    @staticmethod
    def _get_timestamps(n_frames: int, fps: float) -> List[float]:
        """Generates a list of timestamps based on the framerate.

        Args:
            n_frames: The number of frames in the series.
            fps: The framerate to generate timestamps evenly for.

        Returns:
            A list of timestamps for each frame.
        """
        frame_time = 1 / fps
        max_time = n_frames * frame_time
        return np.arange(0, max_time, frame_time, dtype=float).tolist()


    def _get_handover(self) -> Dict[int, float]:
        """Gets the handover data from the already loaded wrist and role data. Does this by looking at the butterworth
        filtered velocity curve for the wrist marker of the follower.

        Returns:
            The handover data, represented as: {trial: timestamp}.
        """
        pass

    def crop_to_handover(self) -> None:
        """Crops all of the trials so that they end at object handover.

        Modifies the private attributes:
            - self._mocap_data
            - self._wrist_data
            - self._role_data
            - self._handover_data
        """
        pass


def test():
    session_data = H2HSessionData('E:/Datasets/CS 4440 Final Project/mat_files_full/26_M4_F5_cropped_data_v2.mat')
    session_data.load()


if __name__ == '__main__':
    test()
