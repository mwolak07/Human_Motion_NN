from typing import ClassVar, List, Dict, Tuple, Optional, Set, Any
from collections.abc import Sequence
from skspatial.objects import Point
from scipy import signal
import numpy as np
import mat73


# TODO: Load in ALL data from the matlab file into this object. Including GAZE (feature).
# TODO: Integrate a more flexible reader that can read only specific trials (performance).
# TODO: Replace lists with numpy fixed memory allocation (performance).


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
            Structured: {trial: {marker: [(timestamp, point)]}}.
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
    session_file: str
    target_markers: Set[str]
    _trials: List[int]
    _mocap_data: Dict[int, Dict[str, List[Tuple[float, Point]]]]
    _wrist_data: Dict[int, Dict[int, List[Tuple[float, Point]]]]
    _role_data: Dict[int, str]
    _handover_data: Dict[int, float]

    def __init__(self, session_file: str, target_markers: Optional[List[str]] = None):
        """Initializes the object with the given session file and target markers.
        Does not read any data until .load() is called.

        Args:
            session_file: The path to the Matlab 7.3 file containing the session's data.
            target_markers: The list of markers to include. (If None, include all). The names should have the subject
                            tags included, like: 'Sub1_FH'.
        """
        # Initialize public attributes.
        self.session_file = session_file
        self.target_markers = set(target_markers) if target_markers is not None else None
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
        # Return a dict with all the data.
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

    def _parse_mocap(self, session_data: Dict[str, Any]) -> Dict[int, Dict[str, List[Tuple[float, Point]]]]:
        """Parses the mocap data from the data dict loaded from the session file.

        Args:
            session_data: The data dict loaded from the session file.

        Returns:
            The mocap data, represented as: {trial: {marker: [(timestamp, Point)]}}.
        """
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
            # Get the list of target labels. If self._target_markers is None, we take all the labels.
            target_labels = self.target_markers if self.target_markers is not None else label_list
            # Get the mocap data for each target marker for each subject.
            self._mocap_data[trial] = {}
            for j in range(len(label_list)):
                label = label_list[j]
                if label in target_labels:
                    self._mocap_data[label] = self._parse_mocap_frames(point_list[j])

    def _parse_mocap_frames(self, mocap_frames: np.ndarray) -> List[Tuple[float, Point]]:
        """Parses the numpy array of mocap frames for one marker.

        Args:
            - mocap_frames: A shape (n_frames, 3) numpy array of mocap points in each frame for this trial and marker.

        Returns:
            A list of pairs of (timestamp, Point).
        """
        point_list = []
        for mocap_frame in mocap_frames:
            point_list.append(Point(mocap_frame))
        timestamp_list = self._get_timestamps(mocap_frames.shape[0], self.mocap_fps)
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
                1: self._parse_mocap_frames(point_list[sub_1_i]),
                2: self._parse_mocap_frames(point_list[sub_2_i])
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
        # Defining thresholds for picking peaks and minimums.
        peak_threshold = 0.2
        min_threshold = 0.1
        output = {}
        for trial in self._trials:
            try:
                follower = 2 if self._role_data[trial] in ['Sub1_IG', 'Sub1_IR'] else 1
                follower_velocity = self._get_follower_velocity(trial, follower)
                # Finding the local maxima using the peak_threshold.
                peaks_i, _ = signal.find_peaks([elem[1] for elem in follower_velocity])
                maxs_i = [i for i in peaks_i if follower_velocity[i][1] > peak_threshold]
                # Finding the local minima by inverting the graph and using min_threshold.
                follower_velocity_inv = [-1 * elem[1] for elem in follower_velocity]
                peaks_i, _ = signal.find_peaks(follower_velocity_inv)
                # Getting the candidate mins, that appear between the maxes we found.
                candidate_mins_i = [i for i in peaks_i if maxs_i[0] < i < maxs_i[-1]]
                # Filtering out the mins that do not have values below our min threshold.
                mins_i = [i for i in candidate_mins_i if follower_velocity[i][1] < min_threshold]
                # If the list is not empty after filtering with the min threshold, we use the end of the list.
                if len(mins_i) != 0:
                    contact_pt = mins_i[-1]
                # If the list is empty, we revert to the candidate mins list and use the end of that list.
                else:
                    contact_pt = candidate_mins_i[-1]
                # Our final output is the timestamp at our contact point.
                output[trial] = follower_velocity[contact_pt][0]
            # When there's an issue finding handover, we remove the trial.
            except Exception as e:
                print(f'Problem finding handover in trial {trial}: {e}')
                self._remove_trial(trial)
        return output

    def _get_follower_velocity(self, trial: int, follower: int) -> List[Tuple[float, float]]:
        """
        Gets a list of the instantaneous velocity at each point for the given follower subject.
        This does not include the velocity for the first point, since there is no prior point. Velocity is defined as
        the distance between the consecutive coordinates over the frame time.

        Also applies a low-pass butterworth filter with order 4 and cutoff frequency 10 / (fps / 2).

        Args:
            trial: The trial to get the follower velocity for.
            follower: 1 if subject 1 is the follower, 2 if subject 2 is the follower.

        Returns:
            A list of (timestamp, instantaneous velocity), not including the first point.
        """
        # Getting the points with the appropriate trial and marker.
        points = self._wrist_data[trial][follower]
        # Filtering out points where we have nan values, as this messes with the butterworth filter.
        points = [point for point in points if not np.any(np.isnan(point))]
        # Getting the velocities from the points.
        velocities = []
        for i in range(1, len(points)):
            d = np.linalg.norm(points[i][1] - points[i - 1][1])
            dt = points[i][0] - points[i - 1][0]
            v = d / dt if dt != 0 else 0
            velocities.append(v)
        # Setting up butterworth filter params.
        filt_order = 4
        cutoff_freq = 10
        filt_cutoff = cutoff_freq / (self.mocap_fps / 2)  # TODO: When we add gaze, this needs to go to global fps.
        filt_type = 'low'
        # Applying the butterworth filter.
        result = signal.butter(N=filt_order, Wn=filt_cutoff, btype=filt_type, output='ba')
        velocities = signal.filtfilt(result[0], result[1], velocities)
        # Adding the timestamps back in.
        output = [(points[i + 1][0], velocities[i]) for i in range(len(velocities))]
        return output

    def _remove_trial(self, trial: int) -> None:
        """Removes the given trial from our internal data.

        Args:
            trial: The trial number to be removed.

        Modifies the private attributes:
            - self._trials
            - self._mocap_data
            - self._wrist_data
            - self._role_data
            - self._handover_data
        """
        self._trials.remove(trial)
        self._mocap_data.pop(trial)
        self._wrist_data.pop(trial)
        self._role_data.pop(trial)
        self._handover_data.pop(trial)

    def crop_nan(self) -> None:
        """Crops all the nan values out of each trial. This is done by taking the smallest crop window over all
        the markers in self._mocap_data and applying it to the rest of the per-frame data.

        Modifies the private attributes:
            - self._mocap_data
            - self._wrist_data
        """
        # Iterate over each trial.
        for trial in self._trials:
            # Get the start and end crop for each marker.
            start_crops = []
            end_crops = []
            for marker in self._mocap_data[trial]:
                start_crop, end_crop = self._get_nan_crop(self._mocap_data[trial][marker])
                start_crops.append(start_crop)
                end_crops.append(end_crop)
            # Choose the latest start and earliest end.
            start_crop = max(start_crops)
            end_crop = min(end_crops)
            # If the crops eliminate all the frames, throw out the trial.
            if end_crop <= start_crop:
                self._remove_trial(trial)
            # Otherwise, crop the internal data.
            else:
                self._crop_mocap_frames(trial, start_crop, end_crop)
                self._crop_wrist_frames(trial, start_crop, end_crop)

    @staticmethod
    def _get_nan_crop(mocap_frames: np.ndarray) -> Tuple[int, int]:
        """Gets the start and end indexes to crop out all the NaN values in the mocap_frames.

        Args:
            - mocap_frames: A shape (n_frames, 3) numpy array of mocap points in each frame for this trial and marker.

        Returns:
            A pair of indexes (start, end)
        """
        # Start crop is the first index where the values are not Nan.
        start_crop = 0
        for i, frame in enumerate(mocap_frames):
            if not np.any(np.isnan(frame)):
                start_crop = i
                break
        # End crop is the first index where the values are not Nan, in the reversed list of frames.
        end_crop = mocap_frames.shape[0]
        for i, frame in enumerate(np.flip(mocap_frames)):
            if not np.any(np.isnan(frame)):
                end_crop = i + 1  # +1 because end crop is not inclusive.
                break
        return start_crop, end_crop

    def _crop_mocap_frames(self, trial: int, start_crop: int, end_crop: int) -> None:
        """Crops the frames for every marker in self._mocap_data for the given trial.

        Args:
            trial: The trial to crop the frames for.
            start_crop: The start index of the crop (inclusive).
            end_crop: The end index of the crop (exclusive).

        Modifies the private attributes:
            - self._mocap_data
        """
        for marker in self._mocap_data[trial]:
            self._mocap_data[trial][marker] = self._mocap_data[trial][marker][start_crop: end_crop]

    def _crop_wrist_frames(self, trial: int, start_crop: int, end_crop: int) -> None:
        """Crops the frames for every subject in self._wrist_data for the given trial.

        Args:
            trial: The trial to crop the frames for.
            start_crop: The start index of the crop (inclusive).
            end_crop: The end index of the crop (exclusive).

        Modifies the private attributes:
            - self._mocap_data
        """
        for subject in self._wrist_data[trial]:
            self._mocap_data[trial][subject] = self._mocap_data[trial][subject][start_crop: end_crop]

    def crop_to_handover(self) -> None:
        """Crops all the trials so that they end at object handover.

        If handover is after the end of the trial, we leave the trial alone.

        Modifies the private attributes:
            - self._mocap_data
            - self._wrist_data
        """
        # Iterate over each trial.
        for trial in self._trials:
            # Get the handover time from the internal list.
            handover_time = self._handover_data[trial]
            # Get the timestamp list from the wrist data.
            timestamps = [frame[0] for frame in self._wrist_data[trial][1]]
            # Handover is past the end, leave the trial alone.
            if handover_time > timestamps[-1]:
                continue
            # Handover is within the timestamps, perform crop.
            else:
                # Get the index of the timestamp.
                end_crop = timestamps.index(handover_time)
                self._crop_mocap_frames(trial, 0, end_crop)
                self._crop_wrist_frames(trial, 0, end_crop)


def test():
    session_data = H2HSessionData('E:/Datasets/CS 4440 Final Project/mat_files_full/26_M4_F5_cropped_data_v2.mat')
    session_data.load()


if __name__ == '__main__':
    test()
