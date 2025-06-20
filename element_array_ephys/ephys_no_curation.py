import importlib
import inspect
import pathlib
from datetime import timedelta, datetime, timezone

import datajoint as dj
import numpy as np
import pandas as pd
from element_interface.utils import dict_to_uuid, find_full_path, find_root_directory
from scipy import signal
import intanrhdreader

from . import ephys_report, probe
from .readers import kilosort, openephys, spikeglx

logger = dj.logger

schema = dj.schema()

_linking_module = None


def activate(
    ephys_schema_name: str,
    probe_schema_name: str = None,
    *,
    create_schema: bool = True,
    create_tables: bool = True,
    linking_module: str = None,
):
    """Activates the `ephys` and `probe` schemas.

    Args:
        ephys_schema_name (str): A string containing the name of the ephys schema.
        probe_schema_name (str): A string containing the name of the probe schema.
        create_schema (bool): If True, schema will be created in the database.
        create_tables (bool): If True, tables related to the schema will be created in the database.
        linking_module (str): A string containing the module name or module containing the required dependencies to activate the schema.

    Dependencies:
    Upstream tables:
        culture.Experiment: A parent table to EphysSession.

    Functions:
        get_ephys_root_data_dir(): Returns absolute path for root data director(y/ies) with all electrophysiological recording sessions, as a list of string(s).
        get_organoid_directory(session_key: dict): Returns path to electrophysiology data for the a particular session as a list of strings.
        get_processed_data_dir(): Optional. Returns absolute path for processed data. Defaults to root directory.

    """

    if isinstance(linking_module, str):
        linking_module = importlib.import_module(linking_module)
    assert inspect.ismodule(
        linking_module
    ), "The argument 'dependency' must be a module's name or a module"

    global _linking_module
    _linking_module = linking_module

    # activate
    probe.activate(
        probe_schema_name, create_schema=create_schema, create_tables=create_tables
    )
    schema.activate(
        ephys_schema_name,
        create_schema=create_schema,
        create_tables=create_tables,
        add_objects=_linking_module.__dict__,
    )
    ephys_report.activate(f"{ephys_schema_name}_report", ephys_schema_name)


# -------------- Functions required by the elements-ephys  ---------------


def get_ephys_root_data_dir() -> list:
    """Fetches absolute data path to ephys data directories.

    The absolute path here is used as a reference for all downstream relative paths used in DataJoint.

    Returns:
        A list of the absolute path(s) to ephys data directories.
    """
    root_directories = _linking_module.get_ephys_root_data_dir()
    if isinstance(root_directories, (str, pathlib.Path)):
        root_directories = [root_directories]

    if hasattr(_linking_module, "get_processed_root_data_dir"):
        root_directories.append(_linking_module.get_processed_root_data_dir())

    return root_directories


def get_organoid_directory(session_key: dict) -> str:
    """Retrieve the session directory with Neuropixels for the given session.

    Args:
        session_key (dict): A dictionary mapping subject to an entry in the subject table, and session_datetime corresponding to a session in the database.

    Returns:
        A string for the path to the session directory.
    """
    return _linking_module.get_organoid_directory(session_key)


def get_processed_root_data_dir() -> str:
    """Retrieve the root directory for all processed data.

    Returns:
        A string for the full path to the root directory for processed data.
    """

    if hasattr(_linking_module, "get_processed_root_data_dir"):
        return _linking_module.get_processed_root_data_dir()
    else:
        return get_ephys_root_data_dir()[0]


# ----------------------------- Table declarations ----------------------


@schema
class AcquisitionSoftware(dj.Lookup):
    """Name of software used for recording electrophysiological data.

    Attributes:
        acq_software ( varchar(24) ): Acquisition software, e.g,. SpikeGLX, OpenEphys
    """

    definition = """  # Name of software used for recording of neuropixels probes - SpikeGLX or Open Ephys
    acq_software: varchar(24)    
    """
    contents = zip(["SpikeGLX", "Open Ephys", "Intan"])


@schema
class Port(dj.Lookup):
    definition = """  # Port ID of the Intan acquisition system
    port_id     : char(2)
    """
    contents = zip(["A", "B", "C", "D"])


@schema
class EphysRawFile(dj.Manual):
    definition = """ # Catalog of all raw ephys files
    file_path         : varchar(512) # path to the file relative to the root directory
    ---
    -> AcquisitionSoftware
    file_time         : datetime #  date and time of the file acquisition
    parent_folder     : varchar(128) #  parent folder containing the file
    filename_prefix   : varchar(64)  #  filename prefix, if any, excluding the datetime information
    """


@schema
class EphysSession(dj.Manual):
    definition = """ # User defined ephys session for downstream analysis.
    -> culture.Experiment
    insertion_number            : tinyint unsigned
    start_time                  : datetime
    end_time                    : datetime
    ---
    session_type                : enum("lfp", "spike_sorting", "both", "test")
    """


@schema
class EphysSessionProbe(dj.Manual):
    """User defined probe for each ephys session.
    Attributes:
        EphysSession (foreign key): EphysSession primary key.
        probe.Probe (foreign key): probe.Probe primary key.
        probe.ElectrodeConfig (foreign key): probe.ElectrodeConfig primary key.
    """

    definition = """
    -> EphysSession
    ---
    -> probe.Probe 
    -> Port  # port ID where the probe was connected to.
    used_electrodes=null     : longblob  # list of electrode IDs used in this session (if null, all electrodes are used)
    """


@schema
class EphysSessionInfo(dj.Imported):
    definition = """ # Store header information from the first session file.
    -> EphysSession
    ---
    session_info: longblob  # Session header info from intan .rhd file. Get this from the first session file.
    """

    def make(self, key):
        query = (
            EphysRawFile
            & f"file_time BETWEEN '{key['start_time']}' AND '{key['end_time']}'"
        )
        if not query:
            raise FileNotFoundError(
                f"No EphysRawFile found BETWEEN '{key['start_time']}' AND '{key['end_time']}'"
            )

        first_file = query.fetch("file_path", order_by="file_time", limit=1)[0]
        first_file = find_full_path(get_ephys_root_data_dir(), first_file)

        # Read file header
        with open(first_file, "rb") as f:
            try:
                header = intanrhdreader.read_header(f)
            except OSError:
                raise OSError(f"Error occurred when reading file {first_file}")
            else:
                del header["spike_triggers"], header["aux_input_channels"]

        logger.info(f"Populating ephys.EphysSessionInfo for <{key}>")

        self.insert(
            [
                {
                    **key,
                    "session_info": header,
                }
            ]
        )


@schema
class LFP(dj.Imported):
    definition = """ # Store pre-processed LFP traces per electrode. Only the LFPs collected from a pre-defined recording session.
    -> EphysSession
    ---
    lfp_sampling_rate    : float # Down-sampled sampling rate (Hz).
    execution_duration   : float # execution duration in hours
    """

    class Trace(dj.Part):
        definition = """
        -> master
        -> probe.ElectrodeConfig.Electrode
        ---
        lfp              : blob@datajoint-blob # uV
        """

    @property
    def key_source(self):
        return (
            EphysSession
            & EphysSessionInfo
            & EphysSessionProbe
            & 'session_type IN ("lfp", "both")'
        )

    TARGET_SAMPLING_RATE = 2500  # Hz
    POWERLINE_NOISE_FREQ = 60  # Hz
    MAX_DURATION_MINUTES = 30  # Minutes

    def make_fetch(self, key):
        # Check if the trace duration is within the expected range
        duration = (key["end_time"] - key["start_time"]).total_seconds() / 60  # minutes
        assert (
            duration <= self.MAX_DURATION_MINUTES
        ), f"LFP session duration {duration} min > max session duration {self.MAX_DURATION_MINUTES} min"

        # Fetch the raw data files for the given ephys session
        query = (
            EphysRawFile
            & f"file_time BETWEEN '{key['start_time']}' AND '{key['end_time']}'"
        )
        if not query:
            logger.info(f"No raw data file found. Skipping LFP for <{key}>")
            return

        logger.info(f"Populating ephys.LFP for <{key}>")

        # Fetch the probe information for the given ephys session
        probe_info = (EphysSessionProbe & key).fetch1()
        probe_type = (probe.Probe & {"probe": probe_info["probe"]}).fetch1("probe_type")
        electrode_query = probe.ElectrodeConfig.Electrode & (
            probe.ElectrodeConfig & {"probe_type": probe_type}
        )

        # Fetch the electrode configuration for the given probe
        # Filter for used electrodes. If probe_info["used_electrodes"] is None, it means all electrodes were used.
        if probe_info["used_electrodes"]:
            electrode_query &= f"electrode IN {tuple(probe_info['used_electrodes'])}"

        lfp_indices = np.array(electrode_query.fetch("channel_idx"), dtype=int)

        electrode_df = electrode_query.fetch(format="frame").reset_index()

        file_paths = query.fetch("file_path", order_by="file_time")

        return (
            file_paths,
            lfp_indices,
            probe_info,
            electrode_df,
            duration,
        )

    def make_compute(
        self,
        key,
        file_paths,
        lfp_indices,
        probe_info,
        electrode_df,
        duration,
    ):
        """Compute broadband LFP signals for each electrode.

        Args:
            key (dict): EphysSession primary key.

        Raises:
            ValueError: If the trace duration is not within the expected range.
            OSError: If there is an error when loading the file.

        Logic:
            - Fetch the probe information for the given ephys session.
            - Fetch the electrode configuration for the given probe.
            - Fetch the raw data files for the given ephys session.
            - Check for missing files or short trace durations in min
            - Design notch filter to remove powerline noise that contaminates the LFP
            - Downsample the signal with `decimate` and apply an anti-aliasing FIR filter
        """
        execution_time = datetime.now(timezone.utc)

        header = {}
        lfp_concat = []
        # Iterate over the raw data files for the given ephys session to load the data
        for file_relpath in file_paths:
            file = find_full_path(get_ephys_root_data_dir(), file_relpath)
            try:
                data = intanrhdreader.load_file(file)
            except OSError:
                raise OSError(f"OS error occurred when loading file {file.name}")

            if not header:
                header = data.pop("header")
                lfp_sampling_rate = header["sample_rate"]
                powerline_noise_freq = (
                    header["notch_filter_frequency"] or self.POWERLINE_NOISE_FREQ
                )  # in Hz

                # Calculate downsampling factor
                true_ratio = lfp_sampling_rate / self.TARGET_SAMPLING_RATE
                downsample_factor = int(np.round(true_ratio))

                # Check if the ratio is within 1% of an integer (1% tolerance)
                if not np.isclose(true_ratio, downsample_factor, rtol=0.01, atol=1e-8):
                    raise ValueError(
                        f"Downsampling factor {true_ratio} is too far from an integer. Check LFP sampling rates."
                    )

                # Get LFP indices (row index of the LFP matrix to be used)
                port_indices = np.array(
                    [
                        ind
                        for ind, ch in enumerate(data["amplifier_channels"])
                        if ch["port_prefix"] == probe_info["port_id"]
                    ]
                )
                lfp_indices = np.sort(port_indices[lfp_indices])

                # Get LFP channels
                channels = np.array(
                    [
                        ch["native_channel_name"]
                        for ch in data["amplifier_channels"]
                        if ch["port_prefix"]
                    ]
                )[lfp_indices]

                # Get channel to electrode mapping
                channel_to_electrode_map = dict(
                    zip(electrode_df["channel_idx"], electrode_df["electrode"])
                )

                channel_to_electrode_map = {
                    f'{probe_info["port_id"]}-{int(channel):03d}': electrode
                    for channel, electrode in channel_to_electrode_map.items()
                }

            lfps = data.pop("amplifier_data")[lfp_indices]
            lfp_concat.append(lfps)

        full_lfp = np.hstack(lfp_concat)

        # Check if the trace duration is within the expected range
        trace_duration = full_lfp.shape[1] / lfp_sampling_rate / 60  # in min
        if abs(trace_duration - duration) > 0.5:
            raise ValueError(
                f"Trace duration mismatch: expected {duration}, got {trace_duration} min"
            )

        # Design notch filter
        notch_b, notch_a = signal.iirnotch(
            w0=powerline_noise_freq, Q=30, fs=lfp_sampling_rate
        )

        all_lfps = []
        for ch_idx, raw_lfp in zip(channels, full_lfp):
            # Apply notch filter
            lfp = signal.filtfilt(notch_b, notch_a, raw_lfp)

            # Downsample the signal with `decimate`
            lfp = signal.decimate(lfp, downsample_factor, ftype="fir", zero_phase=True)
            all_lfps.append(lfp)
            
        execution_duration = ((
                    datetime.now(timezone.utc) - execution_time
                ).total_seconds()
                / 3600)
        return (
            all_lfps,
            channels,
            electrode_df,
            channel_to_electrode_map,
            execution_duration,
        )

    def make_insert(
        self,
        key,
        all_lfps,
        channels,
        electrode_df,
        channel_to_electrode_map,
        execution_duration,
    ):
        self.insert1(
            {
                **key,
                "lfp_sampling_rate": self.TARGET_SAMPLING_RATE,
                "execution_duration": execution_duration,
            }
        )

        for ch_idx, lfp in zip(channels, all_lfps):
            self.Trace.insert1(
                {
                    **key,
                    "electrode_config_hash": electrode_df["electrode_config_hash"][0],
                    "probe_type": electrode_df["probe_type"][0],
                    "electrode": channel_to_electrode_map[ch_idx],
                    "lfp": lfp,
                }
            )


# ------------ Clustering --------------


@schema
class ClusteringMethod(dj.Lookup):
    """Kilosort clustering method.

    Attributes:
        clustering_method (foreign key, varchar(20) ): Kilosort clustering method.
        clustering_methods_desc (varchar(1000) ): Additional description of the clustering method.
    """

    definition = """
    # Method for clustering
    clustering_method: varchar(20)
    ---
    clustering_method_desc: varchar(1000)
    """

    contents = [
        ("kilosort2", "kilosort2 clustering method"),
        ("kilosort2.5", "kilosort2.5 clustering method"),
        ("kilosort3", "kilosort3 clustering method"),
    ]


@schema
class ClusteringParamSet(dj.Lookup):
    """Parameters to be used in clustering procedure for spike sorting.

    Attributes:
        paramset_idx (foreign key): Unique ID for the clustering parameter set.
        ClusteringMethod (dict): ClusteringMethod primary key.
        paramset_desc (varchar(128) ): Description of the clustering parameter set.
        param_set_hash (uuid): UUID hash for the parameter set.
        params (longblob): Set of clustering parameters.
    """

    definition = """
    # Parameter set to be used in a clustering procedure
    paramset_idx:  smallint
    ---
    -> ClusteringMethod    
    paramset_desc: varchar(128)
    param_set_hash: uuid
    unique index (param_set_hash)
    params: longblob  # dictionary of all applicable parameters
    """

    @classmethod
    def insert_new_params(
        cls,
        clustering_method: str,
        paramset_desc: str,
        params: dict,
        paramset_idx: int = None,
    ):
        """Inserts new parameters into the ClusteringParamSet table.

        Args:
            clustering_method (str): name of the clustering method.
            paramset_desc (str): description of the parameter set
            params (dict): clustering parameters
            paramset_idx (int, optional): Unique parameter set ID. Defaults to None.
        """
        if paramset_idx is None:
            paramset_idx = (
                dj.U().aggr(cls, n="max(paramset_idx)").fetch1("n") or 0
            ) + 1

        param_dict = {
            "clustering_method": clustering_method,
            "paramset_idx": paramset_idx,
            "paramset_desc": paramset_desc,
            "params": params,
            "param_set_hash": dict_to_uuid(
                {**params, "clustering_method": clustering_method}
            ),
        }
        param_query = cls & {"param_set_hash": param_dict["param_set_hash"]}

        if param_query:  # If the specified param-set already exists
            existing_paramset_idx = param_query.fetch1("paramset_idx")
            if (
                existing_paramset_idx == paramset_idx
            ):  # If the existing set has the same paramset_idx: job done
                return
            else:  # If not same name: human error, trying to add the same paramset with different name
                raise dj.DataJointError(
                    f"The specified param-set already exists"
                    f" - with paramset_idx: {existing_paramset_idx}"
                )
        else:
            if {"paramset_idx": paramset_idx} in cls.proj():
                raise dj.DataJointError(
                    f"The specified paramset_idx {paramset_idx} already exists,"
                    f" please pick a different one."
                )
            cls.insert1(param_dict)


@schema
class ClusterQualityLabel(dj.Lookup):
    """Quality label for each spike sorted cluster.

    Attributes:
        cluster_quality_label (foreign key, varchar(100) ): Cluster quality type.
        cluster_quality_description (varchar(4000) ): Description of the cluster quality type.
    """

    definition = """
    # Quality
    cluster_quality_label:  varchar(100)  # cluster quality type - e.g. 'good', 'MUA', 'noise', etc.
    ---
    cluster_quality_description:  varchar(4000)
    """
    contents = [
        ("good", "single unit"),
        ("ok", "probably a single unit, but could be contaminated"),
        ("mua", "multi-unit activity"),
        ("noise", "bad unit"),
        ("n.a.", "not available"),
    ]


@schema
class ClusteringTask(dj.Manual):
    """A clustering task to spike sort electrophysiology datasets.

    Attributes:
        EphysSession (foreign key): EphysSession primary key.
        ClusteringParamSet (foreign key): ClusteringParamSet primary key.
        clustering_outdir_dir (varchar (255) ): Relative path to output clustering results.
    """

    definition = """
    # Manual table for defining a clustering task ready to be run
    -> EphysSession
    -> ClusteringParamSet
    ---
    clustering_output_dir='': varchar(255)  #  clustering output directory relative to the clustering root data directory
    """

    @property
    def key_source(self):
        return EphysSession & 'session_type IN ("spike_sorting", "both")'

    @classmethod
    def infer_output_dir(cls, key, relative=False, mkdir=False) -> pathlib.Path:
        """Infer output directory if it is not provided.

        Args:
            key (dict): ClusteringTask primary key.

        Returns:
            Expected clustering_output_dir based on the following convention:
                processed_dir / subject_dir / {clustering_method}_{paramset_idx}
                e.g.: sub4/sess1/kilosort2_0
        """
        processed_dir = pathlib.Path(get_processed_root_data_dir())
        exp_dir = find_full_path(get_ephys_root_data_dir(), get_organoid_directory(key))

        session_time = "_".join(
            [
                key["start_time"].strftime("%Y%m%d%H%M"),
                key["end_time"].strftime("%Y%m%d%H%M"),
            ]
        )

        session_dir = exp_dir / session_time / key["organoid_id"]
        root_dir = find_root_directory(get_ephys_root_data_dir(), exp_dir)

        method = (
            (ClusteringParamSet * ClusteringMethod & key)
            .fetch1("clustering_method")
            .replace(".", "-")
        )

        output_dir = (
            processed_dir
            / session_dir.relative_to(root_dir)
            / f'{method}_{key["paramset_idx"]}'
        )

        if mkdir:
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"{output_dir} created!")

        return output_dir.relative_to(processed_dir) if relative else output_dir

    @classmethod
    def auto_generate_entries(cls, ephys_recording_key: dict, paramset_idx: int = 0):
        """Autogenerate entries based on a particular ephys recording.

        Args:
            ephys_recording_key (dict): EphysSession primary key.
            paramset_idx (int, optional): Parameter index to use for clustering task. Defaults to 0.
        """
        key = {**ephys_recording_key, "paramset_idx": paramset_idx}

        processed_dir = get_processed_root_data_dir()
        output_dir = ClusteringTask.infer_output_dir(key, relative=False, mkdir=True)

        cls.insert1(
            {
                **key,
                "clustering_output_dir": output_dir.relative_to(
                    processed_dir
                ).as_posix(),
            }
        )


@schema
class Clustering(dj.Imported):
    """A processing table to handle each clustering task.

    Attributes:
        ClusteringTask (foreign key): ClusteringTask primary key.
        clustering_time (datetime): Time when clustering results are generated.
        package_version (varchar(16) ): Package version used for a clustering analysis.
    """

    definition = """
    # Clustering Procedure
    -> ClusteringTask
    ---
    clustering_time: datetime  # time of generation of this set of clustering results 
    package_version='': varchar(16)
    """

    def make(self, key):
        """This will be implemented via `ephys_sorter` schema with `si_spike_sorting` tables."""
        pass


@schema
class CuratedClustering(dj.Imported):
    """Clustering results after curation.

    Attributes:
        Clustering (foreign key): Clustering primary key.
    """

    definition = """
    # Clustering results of the spike sorting step.
    -> Clustering    
    """

    class Unit(dj.Part):
        """Single unit properties after clustering and curation.

        Attributes:
            CuratedClustering (foreign key): CuratedClustering primary key.
            unit (int): Unique integer identifying a single unit.
            probe.ElectrodeConfig.Electrode (foreign key): probe.ElectrodeConfig.Electrode primary key.
            ClusteringQualityLabel (foreign key): CLusteringQualityLabel primary key.
            spike_count (int): Number of spikes in this recording for this unit.
            spike_times (longblob): Spike times of this unit, relative to start time of EphysRecording.
            spike_sites (longblob): Array of electrode associated with each spike.
            spike_depths (longblob): Array of depths associated with each spike, relative to each spike.
        """

        definition = """   
        # Properties of a given unit from a round of clustering (and curation)
        -> master
        unit: int
        ---
        -> probe.ElectrodeConfig.Electrode  # electrode with highest waveform amplitude for this unit
        -> ClusterQualityLabel
        spike_count: int         # how many spikes in this recording for this unit
        spike_times: longblob    # (s) spike times of this unit, relative to the start of the EphysRecording
        spike_sites : longblob   # array of electrode associated with each spike
        spike_depths=null : longblob  # (um) array of depths associated with each spike, relative to the (0, 0) of the probe    
        """

    def make(self, key):
        """Automated population of Unit information."""
        clustering_method, output_dir = (
            ClusteringTask * ClusteringParamSet & key
        ).fetch1("clustering_method", "clustering_output_dir")
        output_dir = find_full_path(get_ephys_root_data_dir(), output_dir)

        # Get electrode & channel info
        probe_info = (probe.Probe * EphysSessionProbe & key).fetch1()

        electrode_config_key = (probe.ElectrodeConfig & probe_info).fetch1("KEY")

        electrode_query = (
            probe.ProbeType.Electrode * probe.ElectrodeConfig.Electrode
            & electrode_config_key
        )

        channel2electrode_map: dict[int, dict] = {
            chn.pop("channel_idx"): chn for chn in electrode_query.fetch(as_dict=True)
        }  # e.g., {0: {'organoid_id': 'O09',

        # Get sorter method and create output directory.
        sorter_name = clustering_method.replace(".", "_")
        si_sorting_analyzer_dir = output_dir / sorter_name / "sorting_analyzer"

        if si_sorting_analyzer_dir.exists():  # Read from spikeinterface outputs
            import spikeinterface as si

            sorting_analyzer = si.load_sorting_analyzer(folder=si_sorting_analyzer_dir)
            si_sorting = sorting_analyzer.sorting

            # Find representative channel for each unit
            unit_peak_channel: dict[int, np.ndarray] = (
                si.ChannelSparsity.from_best_channels(
                    sorting_analyzer, 1, peak_sign="both"
                ).unit_id_to_channel_indices
            )
            unit_peak_channel: dict[int, int] = {
                u: chn[0] for u, chn in unit_peak_channel.items()
            }

            spike_count_dict: dict[int, int] = si_sorting.count_num_spikes_per_unit()
            # {unit: spike_count}

            # update channel2electrode_map to match with probe's channel index
            channel2electrode_map = {
                idx: channel2electrode_map[int(chn_idx)]
                for idx, chn_idx in enumerate(sorting_analyzer.get_probe().contact_ids)
            }

            # Get unit id to quality label mapping
            cluster_quality_label_map = {
                int(unit_id): (
                    si_sorting.get_unit_property(unit_id, "KSLabel")
                    if "KSLabel" in si_sorting.get_property_keys()
                    else "n.a."
                )
                for unit_id in si_sorting.unit_ids
            }

            spike_locations = sorting_analyzer.get_extension("spike_locations")
            extremum_channel_inds = si.template_tools.get_template_extremum_channel(
                sorting_analyzer, outputs="index"
            )
            spikes_df = pd.DataFrame(
                sorting_analyzer.sorting.to_spike_vector(
                    extremum_channel_inds=extremum_channel_inds
                )
            )

            units = []
            for unit_idx, unit_id in enumerate(si_sorting.unit_ids):
                unit_id = int(unit_id)
                unit_spikes_df = spikes_df[spikes_df.unit_index == unit_idx]
                spike_sites = np.array(
                    [
                        channel2electrode_map[chn_idx]["electrode"]
                        for chn_idx in unit_spikes_df.channel_index
                    ]
                )
                unit_spikes_loc = spike_locations.get_data()[unit_spikes_df.index]
                _, spike_depths = zip(*unit_spikes_loc)  # x-coordinates, y-coordinates
                spike_times = si_sorting.get_unit_spike_train(
                    unit_id, return_times=True
                )

                assert len(spike_times) == len(spike_sites) == len(spike_depths)

                units.append(
                    {
                        **key,
                        **channel2electrode_map[unit_peak_channel[unit_id]],
                        "unit": unit_id,
                        "cluster_quality_label": cluster_quality_label_map[unit_id],
                        "spike_times": spike_times,
                        "spike_count": spike_count_dict[unit_id],
                        "spike_sites": spike_sites,
                        "spike_depths": spike_depths,
                    }
                )

        else:  # read from kilosort outputs
            raise NotImplementedError

        self.insert1(key)
        self.Unit.insert(units, ignore_extra_fields=True)


@schema
class WaveformSet(dj.Imported):
    """A set of spike waveforms for units out of a given CuratedClustering.

    Attributes:
        CuratedClustering (foreign key): CuratedClustering primary key.
    """

    definition = """
    # A set of spike waveforms for units out of a given CuratedClustering
    -> CuratedClustering
    """

    class PeakWaveform(dj.Part):
        """Mean waveform across spikes for a given unit.

        Attributes:
            WaveformSet (foreign key): WaveformSet primary key.
            CuratedClustering.Unit (foreign key): CuratedClustering.Unit primary key.
            peak_electrode_waveform (longblob): Mean waveform for a given unit at its representative electrode.
        """

        definition = """
        # Mean waveform across spikes for a given unit at its representative electrode
        -> master
        -> CuratedClustering.Unit
        ---
        peak_electrode_waveform: longblob  # (uV) mean waveform for a given unit at its representative electrode
        """

    class Waveform(dj.Part):
        """Spike waveforms for a given unit.

        Attributes:
            WaveformSet (foreign key): WaveformSet primary key.
            CuratedClustering.Unit (foreign key): CuratedClustering.Unit primary key.
            probe.ElectrodeConfig.Electrode (foreign key): probe.ElectrodeConfig.Electrode primary key.
            waveform_mean (longblob): mean waveform across spikes of the unit in microvolts.
            waveforms (longblob): waveforms of a sampling of spikes at the given electrode and unit.
        """

        definition = """
        # Spike waveforms and their mean across spikes for the given unit
        -> master
        -> CuratedClustering.Unit
        -> probe.ElectrodeConfig.Electrode  
        --- 
        waveform_mean: longblob   # (uV) mean waveform across spikes of the given unit
        waveforms=null: longblob  # (uV) (spike x sample) waveforms of a sampling of spikes at the given electrode for the given unit
        """

    def make(self, key):
        """Populates waveform tables."""
        clustering_method, output_dir = (
            ClusteringTask * ClusteringParamSet & key
        ).fetch1("clustering_method", "clustering_output_dir")
        output_dir = find_full_path(get_ephys_root_data_dir(), output_dir)
        sorter_name = clustering_method.replace(".", "_")

        # Get electrode & channel info
        probe_info = (probe.Probe * EphysSessionProbe & key).fetch1()
        electrode_config_key = (probe.ElectrodeConfig & probe_info).fetch1("KEY")
        electrode_query = (
            probe.ProbeType.Electrode * probe.ElectrodeConfig.Electrode
            & electrode_config_key
        )

        channel2electrode_map: dict[int, dict] = {
            chn.pop("channel_idx"): chn for chn in electrode_query.fetch(as_dict=True)
        }  # e.g., {0: {'organoid_id': 'O09',

        si_sorting_analyzer_dir = output_dir / sorter_name / "sorting_analyzer"
        if si_sorting_analyzer_dir.exists():  # read from spikeinterface outputs
            import spikeinterface as si

            sorting_analyzer = si.load_sorting_analyzer(folder=si_sorting_analyzer_dir)

            # Find representative channel for each unit
            unit_peak_channel: dict[int, np.ndarray] = (
                si.ChannelSparsity.from_best_channels(
                    sorting_analyzer, 1, peak_sign="both"
                ).unit_id_to_channel_indices
            )  # {unit: peak_channel_index}
            unit_peak_channel = {u: chn[0] for u, chn in unit_peak_channel.items()}

            # update channel2electrode_map to match with probe's channel index
            channel2electrode_map = {
                idx: channel2electrode_map[int(chn_idx)]
                for idx, chn_idx in enumerate(sorting_analyzer.get_probe().contact_ids)
            }

            templates = sorting_analyzer.get_extension("templates")

            def yield_unit_waveforms():
                for unit in (CuratedClustering.Unit & key).fetch(
                    "KEY", order_by="unit"
                ):
                    # Get mean waveform for this unit from all channels - (sample x channel)
                    unit_waveforms = templates.get_unit_template(
                        unit_id=unit["unit"], operator="average"
                    )
                    unit_peak_waveform = {
                        **unit,
                        "peak_electrode_waveform": unit_waveforms[
                            :, unit_peak_channel[unit["unit"]]
                        ],
                    }

                    unit_electrode_waveforms = [
                        {
                            **unit,
                            **channel2electrode_map[chn_idx],
                            "waveform_mean": unit_waveforms[:, chn_idx],
                        }
                        for chn_idx in channel2electrode_map
                    ]

                    yield unit_peak_waveform, unit_electrode_waveforms

        else:  # read from kilosort outputs (ecephys pipeline)
            raise NotImplementedError

        # insert waveform on a per-unit basis to mitigate potential memory issue
        self.insert1(key)
        for unit_peak_waveform, unit_electrode_waveforms in yield_unit_waveforms():
            if unit_peak_waveform:
                self.PeakWaveform.insert1(unit_peak_waveform, ignore_extra_fields=True)
            if unit_electrode_waveforms:
                self.Waveform.insert(unit_electrode_waveforms, ignore_extra_fields=True)


@schema
class QualityMetrics(dj.Imported):
    """Clustering and waveform quality metrics.

    Attributes:
        CuratedClustering (foreign key): CuratedClustering primary key.
    """

    definition = """
    # Clusters and waveforms metrics
    -> CuratedClustering    
    """

    class Cluster(dj.Part):
        """Cluster metrics for a unit.

        Attributes:
            QualityMetrics (foreign key): QualityMetrics primary key.
            CuratedClustering.Unit (foreign key): CuratedClustering.Unit primary key.
            firing_rate (float): Firing rate of the unit.
            snr (float): Signal-to-noise ratio for a unit.
            presence_ratio (float): Fraction of time where spikes are present.
            isi_violation (float): rate of ISI violation as a fraction of overall rate.
            number_violation (int): Total ISI violations.
            amplitude_cutoff (float): Estimate of miss rate based on amplitude histogram.
            isolation_distance (float): Distance to nearest cluster.
            l_ratio (float): Amount of empty space between a cluster and other spikes in dataset.
            d_prime (float): Classification accuracy based on LDA.
            nn_hit_rate (float): Fraction of neighbors for target cluster that are also in target cluster.
            nn_miss_rate (float): Fraction of neighbors outside target cluster that are in the target cluster.
            silhouette_core (float): Maximum change in spike depth throughout recording.
            cumulative_drift (float): Cumulative change in spike depth throughout recording.
            contamination_rate (float): Frequency of spikes in the refractory period.
        """

        definition = """   
        # Cluster metrics for a particular unit
        -> master
        -> CuratedClustering.Unit
        ---
        firing_rate=null: float # (Hz) firing rate for a unit 
        snr=null: float  # signal-to-noise ratio for a unit
        presence_ratio=null: float  # fraction of time in which spikes are present
        isi_violation=null: float   # rate of ISI violation as a fraction of overall rate
        number_violation=null: int  # total number of ISI violations
        amplitude_cutoff=null: float  # estimate of miss rate based on amplitude histogram
        isolation_distance=null: float  # distance to nearest cluster in Mahalanobis space
        l_ratio=null: float  # 
        d_prime=null: float  # Classification accuracy based on LDA
        nn_hit_rate=null: float  # Fraction of neighbors for target cluster that are also in target cluster
        nn_miss_rate=null: float # Fraction of neighbors outside target cluster that are in target cluster
        silhouette_score=null: float  # Standard metric for cluster overlap
        max_drift=null: float  # Maximum change in spike depth throughout recording
        cumulative_drift=null: float  # Cumulative change in spike depth throughout recording 
        contamination_rate=null: float # 
        """

    class Waveform(dj.Part):
        """Waveform metrics for a particular unit.

        Attributes:
            QualityMetrics (foreign key): QualityMetrics primary key.
            CuratedClustering.Unit (foreign key): CuratedClustering.Unit primary key.
            amplitude (float): Absolute difference between waveform peak and trough in microvolts.
            duration (float): Time between waveform peak and trough in milliseconds.
            halfwidth (float): Spike width at half max amplitude.
            pt_ratio (float): Absolute amplitude of peak divided by absolute amplitude of trough relative to 0.
            repolarization_slope (float): Slope of the regression line fit to first 30 microseconds from trough to peak.
            recovery_slope (float): Slope of the regression line fit to first 30 microseconds from peak to tail.
            spread (float): The range with amplitude over 12-percent of maximum amplitude along the probe.
            velocity_above (float): inverse velocity of waveform propagation from soma to the top of the probe.
            velocity_below (float): inverse velocity of waveform propagation from soma toward the bottom of the probe.
        """

        definition = """   
        # Waveform metrics for a particular unit
        -> master
        -> CuratedClustering.Unit
        ---
        amplitude=null: float  # (uV) absolute difference between waveform peak and trough
        duration=null: float  # (ms) time between waveform peak and trough
        halfwidth=null: float  # (ms) spike width at half max amplitude
        pt_ratio=null: float  # absolute amplitude of peak divided by absolute amplitude of trough relative to 0
        repolarization_slope=null: float  # the repolarization slope was defined by fitting a regression line to the first 30us from trough to peak
        recovery_slope=null: float  # the recovery slope was defined by fitting a regression line to the first 30us from peak to tail
        spread=null: float  # (um) the range with amplitude above 12-percent of the maximum amplitude along the probe
        velocity_above=null: float  # (s/m) inverse velocity of waveform propagation from the soma toward the top of the probe
        velocity_below=null: float  # (s/m) inverse velocity of waveform propagation from the soma toward the bottom of the probe
        """

    def make(self, key):
        """Populates tables with quality metrics data."""
        # Load metrics.csv
        clustering_method, output_dir = (
            ClusteringTask * ClusteringParamSet & key
        ).fetch1("clustering_method", "clustering_output_dir")
        output_dir = find_full_path(get_ephys_root_data_dir(), output_dir)
        sorter_name = clustering_method.replace(".", "_")

        si_sorting_analyzer_dir = output_dir / sorter_name / "sorting_analyzer"
        if si_sorting_analyzer_dir.exists():  # read from spikeinterface outputs
            import spikeinterface as si

            sorting_analyzer = si.load_sorting_analyzer(folder=si_sorting_analyzer_dir)
            qc_metrics = sorting_analyzer.get_extension("quality_metrics").get_data()
            template_metrics = sorting_analyzer.get_extension(
                "template_metrics"
            ).get_data()
            metrics_df = pd.concat([qc_metrics, template_metrics], axis=1)

            metrics_df.rename(
                columns={
                    "amplitude_median": "amplitude",
                    "isi_violations_ratio": "isi_violation",
                    "isi_violations_count": "number_violation",
                    "silhouette": "silhouette_score",
                    "rp_contamination": "contamination_rate",
                    "drift_ptp": "max_drift",
                    "drift_mad": "cumulative_drift",
                    "half_width": "halfwidth",
                    "peak_trough_ratio": "pt_ratio",
                    "peak_to_valley": "duration",
                },
                inplace=True,
            )
        else:  # read from kilosort outputs (ecephys pipeline)
            raise NotImplementedError

        metrics_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        metrics_list = [
            dict(metrics_df.loc[unit_key["unit"]], **unit_key)
            for unit_key in (CuratedClustering.Unit & key).fetch("KEY")
        ]

        self.insert1(key)
        self.Cluster.insert(metrics_list, ignore_extra_fields=True)
        self.Waveform.insert(metrics_list, ignore_extra_fields=True)
