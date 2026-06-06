import sys
lib_path = [r'C:\Users\ikahbasi\OneDrive\Applications\GitHub\SeisRoutine',
            r'C:\Users\ikahb\OneDrive\Applications\GitHub\SeisRoutine']
for path in lib_path:
    sys.path.append(path)
##########################################################################
import SeisRoutine.waveform as srw
from obspy import read
import logging


def fill_missing_channels(
        available_channels,
        reference_channels={0: "E", 1: "N", 2: "Z"},
        priority_channels={0: 1, 1: 1, 2: 1},
        ):
    """
    Generate a complete channel mapping by replacing missing channels with
    the most appropriate available channel.

    This function assumes a reference set of channels (e.g., E, N, Z) and
    checks which channels are available. If one or more reference channels
    are missing, the function inserts replacement channel indices into the
    output list so that the returned list has the same length and ordering
    as the reference channel definition.

    Replacement logic:
        - If the vertical channel ("Z") is missing, the available channel
          with the highest priority is used as the replacement.
        - Otherwise, the first available channel is used as the replacement.
        - Missing channels receive the selected replacement channel index
          at their corresponding positions.

    Parameters
    ----------
    available_channels : list[int]
        Indices of channels that are available in the input data.

    reference_channels : dict[int, str], optional
        Mapping of channel indices to channel names. The keys define the
        expected channel positions. Default is:

            {0: "E", 1: "N", 2: "Z"}

    priority_channels : dict[int, float], optional
        Priority score for each channel index. Higher values indicate a
        more suitable channel for replacement when the vertical channel
        is missing. Default is:

            {0: 1, 1: 1, 2: 1}

    Returns
    -------
    list[int]
        A list of channel indices with the same structure as the reference
        channel layout. Missing channels are replaced according to the
        selection rules described above.

    Notes
    -----
    The function modifies `priority_channels` internally by setting the
    priority of missing channels to zero. If the original dictionary must
    remain unchanged, pass a copy of it.

    Examples
    --------
    >>> fill_missing_channels([1, 2])
    [1, 1, 2]

    >>> fill_missing_channels([0, 1])
    [0, 1, 0]

    >>> fill_missing_channels(
    ...     available_channels=[0, 2],
    ...     priority_channels={0: 1, 1: 0.9, 2: 1}
    ... )
    [0, 0, 2]

    >>> lst = [[1, 2],
    ...        [0, 1],
    ...        [0, 2],
    ...       ]
    >>> for available_channels in lst:
    >>>     target = fill_missing_channels(
    ...                 available_channels=available_channels,
    ...                 reference_channels={0: "E", 1: "N", 2: "Z"},
    ...                 priority_channels={0: 1, 1: 0.9, 2: 1})
    >>>     print(target)
    [1, 1, 2]
    [0, 1, 0]
    [0, 0, 2]
    """
    available_channels = sorted(available_channels)
    defect_channels = set(reference_channels.keys()) - set(available_channels)
    output = available_channels
    if available_channels==[] or defect_channels=={}:
        pass
    else:
        for channel in defect_channels:
            priority_channels[channel] = 0
        if next((k for k, v in reference_channels.items() if v == "Z"),
                None) in defect_channels:
            replacement_channel = max(priority_channels, key=priority_channels.get)
        else:
            replacement_channel = available_channels[0]
        ###
        for channel in defect_channels:
            output.insert(channel, replacement_channel)
    return output


class obspy_stream_reader:
    def __init__(self, root, pattern_path):
        self.root = root
        self.pattern_path = pattern_path
        self.stream = None
        self.stats  = None

    def _read(self, time):
        pattern = self.pattern_path.format(time=time)
        path = f'{self.root}/{pattern}'
        logging.info(f'Reading Data: {path}')
        self.stream = read(path)
        self._soft_preprocessing()
        self.stations = list({tr.stats.station for tr in self.stream})
    
    def _soft_preprocessing(self):
        srw.waveform.uni_sps(self.stream, sps=None)
        self.stream.merge(-1)
        self.stream.detrend('constant')
        self.stream.merge()
        self.stream = self.stream.split()
        # self.stream.merge(method=1, fill_value=0)
        # self.stream.filter('bandpass', freqmin=0.5, freqmax=49, zerophase=True)

    def sps_check(self, sps=100):
        # print('Available sps:', {tr.stats.sampling_rate for tr in self.stream})
        assert all(tr.stats.sampling_rate==sps for tr in self.stream)
    
    def get_data_related_to_pick(self, pick):
        if self.stream is None:
            self._read(time=pick.time)
        if not pick.waveform_id.station_code in self.stations:
            self._read(time=pick.time)
        if not pick.time.julday == self.stream[0].stats.starttime.julday:
            self._read(time=pick.time)
        target_stream = self.stream.select(station=pick.waveform_id.station_code)
        return target_stream
