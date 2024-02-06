"""
The following DataJoint pipeline implements the sequence of steps in the spike-sorting routine featured in the
"spikeinterface" pipeline.
Spikeinterface developed by Alessio Buccino, Samuel Garcia, Cole Hurwitz, Jeremy Magland, and Matthias Hennig (https://github.com/SpikeInterface)

The DataJoint pipeline currently incorporated Spikeinterfaces approach of running Kilosort using a container

The follow pipeline features intermediary tables:
1. PreProcessing - for preprocessing steps (no GPU required)
    - create recording extractor and link it to a probe
    - bandpass filtering
    - common mode referencing
2. SIClustering - kilosort (MATLAB) - requires GPU and docker/singularity containers
    - supports kilosort 2.0, 2.5 or 3.0 (https://github.com/MouseLand/Kilosort.git)
3. PostProcessing - for postprocessing steps (no GPU required)
    - create waveform extractor object
    - extract templates, waveforms and snrs
    - quality_metrics
"""

import pathlib
from datetime import datetime

import datajoint as dj
import pandas as pd
import probeinterface as pi
import spikeinterface as si
from element_interface.utils import find_full_path
from spikeinterface import exporters, postprocessing, qualitymetrics, sorters

from element_array_ephys import get_logger, probe, readers

from . import si_preprocessing

log = get_logger(__name__)

schema = dj.schema()

ephys = None


def activate(
    schema_name,
    *,
    ephys_module,
    create_schema=True,
    create_tables=True,
):
    """
    activate(schema_name, *, create_schema=True, create_tables=True, activated_ephys=None)
        :param schema_name: schema name on the database server to activate the `spike_sorting` schema
        :param ephys_module: the activated ephys element for which this `spike_sorting` schema will be downstream from
        :param create_schema: when True (default), create schema in the database if it does not yet exist.
        :param create_tables: when True (default), create tables in the database if they do not yet exist.
    """
    global ephys
    ephys = ephys_module
    schema.activate(
        schema_name,
        create_schema=create_schema,
        create_tables=create_tables,
        add_objects=ephys.__dict__,
    )


SI_SORTERS = [s.replace("_", ".") for s in si.sorters.sorter_dict.keys()]

SI_READERS = {
    "Open Ephys": si.extractors.read_openephys,
    "SpikeGLX": si.extractors.read_spikeglx,
    "Intan": si.extractors.read_intan,
}


@schema
class PreProcessing(dj.Imported):
    """A table to handle preprocessing of each clustering task. The output will be serialized and stored as a si_recording.pkl in the output directory."""

    definition = """
    -> ephys.ClusteringTask
    ---
    execution_time: datetime   # datetime of the start of this step
    execution_duration: float  # execution duration in hours
    """

    @property
    def key_source(self):
        return (
            ephys.ClusteringTask * ephys.ClusteringParamSet
            & {"task_mode": "trigger"}
            & f"clustering_method in {tuple(SI_SORTERS)}"
        ) - ephys.Clustering

    def make(self, key):
        """Triggers or imports clustering analysis."""
        execution_time = datetime.utcnow()

        # Set the output directory
        acq_software, output_dir, params = (
            ephys.ClusteringTask * ephys.EphysRecording * ephys.ClusteringParamSet & key
        ).fetch1("acq_software", "clustering_output_dir", "params")

        for req_key in (
            "SI_SORTING_PARAMS",
            "SI_PREPROCESSING_METHOD",
            "SI_WAVEFORM_EXTRACTION_PARAMS",
            "SI_QUALITY_METRICS_PARAMS",
        ):
            if req_key not in params:
                raise ValueError(
                    f"{req_key} must be defined in ClusteringParamSet for SpikeInterface execution"
                )

        if not output_dir:
            output_dir = ephys.ClusteringTask.infer_output_dir(
                key, relative=True, mkdir=True
            )
            # update clustering_output_dir
            ephys.ClusteringTask.update1(
                {**key, "clustering_output_dir": output_dir.as_posix()}
            )
        output_dir = pathlib.Path(output_dir)
        output_full_dir = find_full_path(ephys.get_ephys_root_data_dir(), output_dir)

        recording_file = (
            output_full_dir / "si_recording.pkl"
        )  # recording cache to be created for each key

        # Create SI recording extractor object
        if acq_software == "SpikeGLX":
            spikeglx_meta_filepath = ephys.get_spikeglx_meta_filepath(key)
            spikeglx_recording = readers.spikeglx.SpikeGLX(
                spikeglx_meta_filepath.parent
            )
            spikeglx_recording.validate_file("ap")
            data_dir = spikeglx_meta_filepath.parent
        elif acq_software == "Open Ephys":
            oe_probe = ephys.get_openephys_probe_data(key)
            assert len(oe_probe.recording_info["recording_files"]) == 1
            data_dir = oe_probe.recording_info["recording_files"][0]
        else:
            raise NotImplementedError(f"Not implemented for {acq_software}")

        stream_names, stream_ids = si.extractors.get_neo_streams(
            acq_software.strip().lower(), folder_path=data_dir
        )
        si_recording: si.BaseRecording = SI_READERS[acq_software](
            folder_path=data_dir, stream_name=stream_names[0]
        )

        # Add probe information to recording object
        electrode_config_key = (
            probe.ElectrodeConfig * ephys.EphysRecording & key
        ).fetch1("KEY")
        electrodes_df = (
            (
                probe.ElectrodeConfig.Electrode * probe.ProbeType.Electrode
                & electrode_config_key
            )
            .fetch(format="frame")
            .reset_index()[["electrode", "x_coord", "y_coord", "shank"]]
        )
        channels_details = ephys.get_recording_channels_details(key)

        # Create SI probe object
        si_probe = readers.probe_geometry.to_probeinterface(electrodes_df)
        si_probe.set_device_channel_indices(channels_details["channel_ind"])
        si_recording.set_probe(probe=si_probe, in_place=True)

        # Run preprocessing and save results to output folder
        si_preproc_func = si_preprocessing.preprocessing_function_mapping[
            params["SI_PREPROCESSING_METHOD"]
        ]
        si_recording = si_preproc_func(si_recording)
        si_recording.dump_to_pickle(file_path=recording_file)

        self.insert1(
            {
                **key,
                "execution_time": execution_time,
                "execution_duration": (
                    datetime.utcnow() - execution_time
                ).total_seconds()
                / 3600,
            }
        )


@schema
class SIClustering(dj.Imported):
    """A processing table to handle each clustering task."""

    definition = """
    -> PreProcessing
    sorter_name: varchar(30)        # name of the sorter used
    ---
    execution_time: datetime        # datetime of the start of this step
    execution_duration: float       # execution duration in hours
    """

    def make(self, key):
        execution_time = datetime.utcnow()

        # Load recording object.
        output_dir = (ephys.ClusteringTask & key).fetch1("clustering_output_dir")
        output_full_dir = find_full_path(ephys.get_ephys_root_data_dir(), output_dir)
        recording_file = output_full_dir.parent / "si_recording.pkl"
        si_recording: si.BaseRecording = si.load_extractor(recording_file)

        # Get sorter method and create output directory.
        clustering_method, params = (
            ephys.ClusteringTask * ephys.ClusteringParamSet & key
        ).fetch1("clustering_method", "params")
        sorter_name = (
            "kilosort2_5" if clustering_method == "kilosort2.5" else clustering_method
        )
        sorter_dir = output_full_dir / sorter_name

        si_sorting_params = params["SI_SORTING_PARAMS"]

        # Run sorting
        si_sorting: si.sorters.BaseSorter = si.sorters.run_sorter(
            sorter_name=sorter_name,
            recording=si_recording,
            output_folder=sorter_dir,
            verbse=True,
            docker_image=True,
            **si_sorting_params,
        )

        # Run sorting
        sorting_save_path = sorter_dir / "si_sorting.pkl"
        si_sorting.dump_to_pickle(sorting_save_path)

        self.insert1(
            {
                **key,
                "sorter_name": sorter_name,
                "execution_time": execution_time,
                "execution_duration": (
                    datetime.utcnow() - execution_time
                ).total_seconds()
                / 3600,
            }
        )


@schema
class PostProcessing(dj.Imported):
    """A processing table to handle each clustering task."""

    definition = """
    -> SIClustering
    ---
    execution_time: datetime   # datetime of the start of this step
    execution_duration: float  # execution duration in hours
    """

    def make(self, key):
        execution_time = datetime.utcnow()
        JOB_KWARGS = dict(n_jobs=-1, chunk_size=30000)

        # Load sorting & recording object.
        output_dir = (ephys.ClusteringTask & key).fetch1("clustering_output_dir")
        output_dir = find_full_path(ephys.get_ephys_root_data_dir(), output_dir)
        recording_file = output_dir / "si_recording.pkl"
        sorter_dir = output_dir / key["sorter_name"]
        sorting_file = sorter_dir / "si_sorting.pkl"

        si_recording: si.BaseRecording = si.load_extractor(recording_file)
        si_sorting: si.sorters.BaseSorter = si.load_extractor(sorting_file)

        si_waveform_extraction_params = params["SI_WAVEFORM_EXTRACTION_PARAMS"]

        # Extract waveforms
        we: si.WaveformExtractor = si.extract_waveforms(
            si_recording,
            si_sorting,
            folder=sorter_dir / "waveform",  # The folder where waveforms are cached
            **si_waveform_extraction_params
            overwrite=True,
            **JOB_KWARGS,
        )

        # Calculate QC Metrics
        metrics: pd.DataFrame = si.qualitymetrics.compute_quality_metrics(
            we,
            metric_names=[
                "firing_rate",
                "snr",
                "presence_ratio",
                "isi_violation",
                "num_spikes",
                "amplitude_cutoff",
                "amplitude_median",
                "sliding_rp_violation",
                "rp_violation",
                "drift",
            ],
        )
        # Add PCA based metrics. These will be added to the metrics dataframe above.
        _ = si.postprocessing.compute_principal_components(
            waveform_extractor=we, n_components=5, mode="by_channel_local"
        )  # TODO: the parameters need to be checked
        metrics = si.qualitymetrics.compute_quality_metrics(waveform_extractor=we)

        # Save results
        self.insert1(
            {
                **key,
                "execution_time": execution_time,
                "execution_duration": (
                    datetime.utcnow() - execution_time
                ).total_seconds()
                / 3600,
            }
        )

        # all finished, insert this `key` into ephys.Clustering
        ephys.Clustering.insert1(
            {**key, "clustering_time": datetime.utcnow()}, allow_direct_insert=True
        )
