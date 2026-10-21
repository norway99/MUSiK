import numpy as np
import os
import glob
import re
import multiprocessing
from queue import Empty
import time
import tqdm
from dataclasses import dataclass, field
from typing import List, Optional, Union

from .utils import utils
from .phantom import Phantom
from .transducer_set import TransducerSet
from .simulation import Simulation, SimProperties
from .sensor import Sensor

from matplotlib import pyplot as plt

SENTINEL = "sentinel"


@dataclass
class Results:
    """Provides indexed access to saved simulation result files.

    Results manages the collection of signal files produced by simulation runs,
    allowing indexed access to individual results without loading all data into
    memory at once.

    Attributes:
        results_path: Directory path containing signal_*.npy result files.
        result_paths: Sorted list of paths to signal result files (auto-populated).
        other_signal_paths: Sorted list of paths to additional recorded signals.
        length: Number of available result files.
        result_shape: Shape of result arrays (time_steps, num_elements).

    Example:
        >>> results = Results('./experiment/results')
        >>> time_array, signals = results[0]  # Load first result
        >>> available = results.indices()     # Get list of available indices
    """
    results_path: str = None
    result_paths: list = field(init=False)
    other_signal_paths: list = field(init=False)
    length: int = field(init=False)
    result_shape: tuple = field(init=False)

    def __post_init__(self):
        self.result_paths = sorted(glob.glob(self.results_path + "/signal_*.np?"))
        self.other_signal_paths = sorted(glob.glob(self.results_path + "/key_signal*.np?"))
        self.length = len(self.result_paths)
        if self.length == 0:
            self.result_shape = None
        else:
            first = utils.load_array(self.result_paths[0])
            self.result_shape = first.shape

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        search_index = self.indices().index(index)
        data = utils.load_array(self.result_paths[search_index])
        if len(self.other_signal_paths):
            other_data = utils.load_array(self.other_signal_paths[search_index])
            return data[0], data[1:], other_data
        return data[0], data[1:]

    def indices(self):
        indices = []
        for path in self.result_paths:
            indices.append(int(re.findall(r"\d+", os.path.basename(path))[0]))
        return indices


@dataclass
class Experiment:
    """Orchestrates multi-process execution of acoustic simulations.

    Experiment is the central coordinator for running large-scale ultrasound
    simulations. It combines all simulation components (phantom, transducers,
    sensor, properties) and manages parallel execution across multiple workers
    or compute nodes.

    The experiment workflow:
    1. Create experiment with all components
    2. Save experiment configuration to disk
    3. Run simulations (single-threaded, multi-process, or SLURM)
    4. Load results for reconstruction

    Attributes:
        simulation_path: Directory for saving experiment config and results.
        sim_properties: SimProperties instance for grid/timing configuration.
        phantom: Phantom instance defining the tissue volume.
        transducer_set: TransducerSet with transducer configurations and poses.
        sensor: Sensor instance for signal capture configuration.
        nodes: Number of compute nodes for distributed execution.
        results: Results instance for accessing saved simulation outputs.
        indices: List of simulation indices to run.
        gpu: Whether to use GPU acceleration (default True).
        num_threads: Host threads for each k-Wave binary (default 4).
        prep_workers: Number of CPU processes preparing simulations (default 1).
        sim_workers: Number of simulations run concurrently on each GPU (default 1).
        gpus: GPUs to run on: an int n (the first n visible devices), a list of
            indices into the visible devices, or None to use the current device.
        workers: Deprecated alias for prep_workers.
        additional_keys: Additional k-Wave outputs to record (e.g., ['p_max']).
        repeat: Batch job mode flag (-1 for static index assignment).

    Choosing worker counts:
        Running splits into CPU-bound preparation (building the medium and k-Wave
        objects) and GPU-bound simulation (running k-Wave). The log reports both for
        every simulation ("preparation ... completed in", "simulation ... completed
        in"), which is enough to size them:

        - sim_workers (per GPU): each simulation launches the k-Wave binary afresh, and
          the GPU sits idle while input is written, the binary starts up, and output is
          read back. Extra simulation workers fill that idle time. Use 2-4 when
          simulations are small and numerous (seconds each, a small fraction of GPU
          memory); gains flatten once the GPU is continuously busy. For compute-heavy
          simulations on large grids the GPU is already busy, so keep 1. Each worker
          holds its own simulation on the GPU, so sim_workers x peak device memory per
          simulation (reported in k-Wave's log) must fit on the device.
        - prep_workers: preparation must keep up with every simulation worker. With
          preparation time t_prep and simulation time t_sim, use at least
          sim_workers x number of GPUs x t_prep / t_sim, rounded up. Each prepared
          simulation waits in host memory with its full medium, and up to
          max(prep_workers, sim_workers x number of GPUs) can be waiting at once.
        - num_threads: each k-Wave binary otherwise starts one host thread per CPU core,
          so several concurrent simulations oversubscribe the CPU.
        - Keep prep_workers + sim_workers x number of GPUs within the CPU cores available.
        - Extra GPUs don't scale linearly for small simulations: host-side work per
          simulation (memory allocation, transfers, kernel launches through the driver)
          slows as more simulations run at once.

    Example:
        >>> exp = Experiment(
        ...     simulation_path='./my_experiment',
        ...     sim_properties=props,
        ...     phantom=phantom,
        ...     transducer_set=tx_set,
        ...     sensor=sensor,
        ... )
        >>> exp.save()
        >>> exp.run(prep_workers=1, sim_workers=3, gpus=2)  # 3 simulations per GPU on 2 GPUs

        >>> # Later, load and access results
        >>> exp = Experiment.load('./my_experiment')
        >>> time, signals = exp.results[0]
    """
    simulation_path: str = None
    sim_properties: SimProperties = None
    phantom: Phantom = None
    transducer_set: TransducerSet = None
    sensor: Sensor = None
    nodes: int = None
    results: Results = None
    indices: list = None
    gpu: bool = True
    num_threads: int = 4
    prep_workers: int = 1
    sim_workers: int = 1
    gpus: Union[int, List[int], None] = None
    workers: Optional[int] = None
    additional_keys: list = field(default_factory=list)
    repeat: int = None

    def __post_init__(self):
        if self.simulation_path is None:
            self.simulation_path = os.path.join(os.getcwd(), "experiment")
            os.makedirs(self.simulation_path, exist_ok=True)
        if self.workers is not None:
            self.prep_workers, self.workers = self.workers, None
        if self.prep_workers is not None and self.prep_workers > 4:
            print(
                "prep_workers is the number of simulations being prepared simultaneously. Having many is RAM intensive and only helps if preparation can't keep up with the simulation workers"
            )
        slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
        if slurm_cpus is not None:
            print(f"Slurm environment detected. Found {slurm_cpus} cpus available")
            num_cpus = int(slurm_cpus)
            if self.prep_workers is not None and num_cpus < self.prep_workers:
                self.prep_workers = num_cpus
            self.repeat = -1
            print(f"Setting repeat to -1 to avoid asynchronous index allocation")
        os.makedirs(os.path.join(self.simulation_path, f"results"), exist_ok=True)
        self.add_results()
        self.indices = self.indices_to_run(self.indices)
        self.additional_keys = self.check_added_keys(self.additional_keys)

    def __len__(self):
        if self.transducer_set is None:
            return 0
        return sum([
            transducer.get_num_rays()
            for transducer in self.transducer_set.transmit_transducers()
        ])

    def save(self, filepath=None):
        """Save experiment configuration to disk.

        Saves all experiment components (SimProperties, Phantom, TransducerSet,
        Sensor) and metadata to the specified directory. Must be called before
        running simulations.

        Args:
            filepath: Directory path for saving. If None, uses simulation_path.
        """
        if filepath is None:
            filepath = self.simulation_path
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        self.sim_properties.save(os.path.join(filepath, f"sim_properties.json"))
        self.phantom.save(os.path.join(filepath, f"phantom"))
        self.transducer_set.save(os.path.join(filepath, f"transducer_set.json"))
        self.sensor.save(os.path.join(filepath, f"sensor.json"))
        dictionary = {
            "indices": self.indices,
            "nodes": self.nodes,
            "gpu": self.gpu,
            "num_threads": self.num_threads,
            "prep_workers": self.prep_workers,
            "sim_workers": self.sim_workers,
            "gpus": self.gpus,
            "additional_keys": self.additional_keys,
        }
        utils.dict_to_json(dictionary, os.path.join(filepath, f"experiment.json"))

    @classmethod
    def load(cls, filepath):
        """Load a saved experiment from disk.

        Reconstructs all experiment components from saved files and initializes
        the Results object for accessing simulation outputs.

        Args:
            filepath: Directory path containing saved experiment files.

        Returns:
            Experiment instance with all components loaded.

        Raises:
            FileNotFoundError: If required configuration files are missing.
        """
        experiment = cls(simulation_path=filepath)
        experiment.sim_properties = SimProperties.load(
            os.path.join(filepath, f"sim_properties.json")
        )
        experiment.phantom = Phantom.load(os.path.join(filepath, f"phantom"))
        experiment.transducer_set = TransducerSet.load(
            os.path.join(filepath, f"transducer_set.json"),
            c0=experiment.phantom.baseline[0],
        )
        experiment.sensor = Sensor.load(
            os.path.join(filepath, f"sensor.json"), experiment.transducer_set
        )
        experiment.results = Results(os.path.join(filepath, f"results"))

        experiment_dict = utils.json_to_dict(os.path.join(filepath, f"experiment.json"))
        experiment.nodes = experiment_dict["nodes"]
        experiment.gpu = experiment_dict["gpu"]
        experiment.num_threads = experiment_dict.get("num_threads", 4)
        # Older experiments saved "workers", which now means prep_workers.
        experiment.prep_workers = experiment_dict.get("prep_workers", experiment_dict.get("workers", 1))
        experiment.sim_workers = experiment_dict.get("sim_workers", experiment_dict.get("lanes", 1))
        experiment.gpus = experiment_dict.get("gpus")
        experiment.additional_keys = experiment_dict["additional_keys"]
        experiment.add_results()
        if experiment_dict["indices"] is None:
            experiment.indices = experiment.indices_to_run()
        else:
            experiment.indices = experiment_dict["indices"]
            experiment.indices = experiment.indices_to_run(experiment.indices)
        if len(experiment.results) < len(experiment):
            print(
                f"Number of simulation results ({len(experiment.results)}) is less than the expected number of simulation results ({len(experiment)}), are you sure the simulation finished running?"
            )
        elif len(experiment.results) > len(experiment):
            print(
                f"Number of simulation results ({len(experiment.results)}) is greater than the expected number of simulation results ({len(experiment)}), did the experiment parameters change since running?"
            )
        return experiment

    # get the simulation indices of any simulations that do not have results
    def indices_to_run(self, indices=None, repeat=False):
        if indices is None:
            indices = list(range(len(self)))
        if np.isscalar(indices):
            indices = [indices]
        if (self.results is None or len(self.results) == 0) or repeat:
            return indices
        else:
            return sorted(list(set(indices) - set(self.results.indices())))

    def subdivide(self, indices=None, repeat=False):
        """Split simulation indices across compute nodes.

        Divides the work evenly across self.nodes for distributed execution
        (e.g., SLURM array jobs).

        Args:
            indices: List of indices to subdivide. If None, uses pending indices.
            repeat: If True, include indices that already have results.

        Returns:
            List of numpy arrays, one per node, containing that node's indices.
            Returns None if no indices to run.
        """
        if indices is None:
            indices = self.indices_to_run(indices, repeat=repeat)
        if len(indices) == 0:
            return None
        return np.array_split(np.array(indices), self.nodes)

    def run(
        self,
        node=None,
        dry=False,
        repeat=False,
        prep_workers=None,
        dry_fast=False,
        sim_workers=None,
        gpus=None,
        workers=None,
    ):
        """Execute simulations for all pending indices.

        Runs acoustic simulations using a producer-consumer pattern: preparation
        workers (CPU) build k-Wave objects into a queue, and simulation workers
        (GPU) run them, `sim_workers` per GPU. See the class docstring for sizing.

        Args:
            node: Compute node index for distributed execution. If None,
                runs all nodes sequentially.
            dry: If True, perform a dry run without actual simulation. Use
                for initializing sensor components for reconstruction. after
                loading complete results.
            repeat: If True, re-run simulations even if results exist.
            prep_workers: Number of preparation processes. If None, uses
                self.prep_workers; if that is None too, runs serially in-process.
            dry_fast: If True and dry=True, only dry-run one ray per
                transducer (assumes identical transducers!).
            sim_workers: Simulations run concurrently on each GPU. If None,
                uses self.sim_workers.
            gpus: GPUs to run on (int count, list of visible-device indices, or
                None for the current device). If None, uses self.gpus.
            workers: Deprecated alias for prep_workers.
        """
        if prep_workers is None:
            prep_workers = workers if workers is not None else self.prep_workers
        if sim_workers is None:
            sim_workers = self.sim_workers
        if gpus is None:
            gpus = self.gpus
        assert os.path.exists(self.simulation_path), (
            "Attempting to run simulations but an experiment directory does not exist. Please save the experiment (my_experiment.save()) before running simulations."
        )

        if dry:
            indices = self.indices_to_run(repeat=True)
        else:
            indices = self.indices_to_run(repeat=repeat)

        options = dict(dry=dry, repeat=repeat, prep_workers=prep_workers, dry_fast=dry_fast, sim_workers=sim_workers, gpus=gpus)
        if node is None:
            if self.nodes is None:
                self.nodes = 1
            if dry:
                self.run(0, **options)
            else:
                for node in range(self.nodes):
                    self.run(node, **options)
        else:
            if dry:
                print("dry run of simulation")
                if not dry_fast:
                    index = 0
                    for transducer in tqdm.tqdm(
                        self.transducer_set.transmit_transducers()
                    ):
                        self.simulate(index, dry=dry)
                        index += transducer.get_num_rays()
                else:
                    print(
                        "Fast dry runs only works when all transducers are identical, use dry_fast=False if your transducers differ from each other"
                    )
                    self.simulate(0, dry=dry)
                    global_not_transducer = self.transducer_set.transmit_transducers()[
                        0
                    ].not_transducer
                    global_pulse = self.transducer_set.transmit_transducers()[
                        0
                    ].get_pulse()
                    for transducer in tqdm.tqdm(self.transducer_set.transducers):
                        transducer.not_transducer = global_not_transducer
                        transducer.pulse = global_pulse
            else:
                if prep_workers is None:
                    for index in indices:
                        self.simulate(index)
                else:
                    subdivisions = self.subdivide(repeat=repeat)
                    if subdivisions is None:
                        print("Found no more simulations to run.")
                    else:
                        devices = self.__simulation_devices(gpus)
                        print(
                            f"running with {prep_workers} preparation worker(s) and {sim_workers} simulation worker(s) "
                            f"on each of {len(devices)} device(s): {devices}\n"
                        )
                        self.__run_pipeline(subdivisions[node], prep_workers, sim_workers, devices, repeat)

    def __simulation_devices(self, gpus):
        """CUDA device ids for the simulation workers, as values for CUDA_VISIBLE_DEVICES.

        `gpus` indexes into the devices this process can see, so it composes with a
        scheduler that has already restricted CUDA_VISIBLE_DEVICES (e.g. Slurm).
        [None] means: leave device selection alone.
        """
        if gpus is None or not self.gpu:
            return [None]
        requested = list(range(gpus)) if isinstance(gpus, int) else [int(g) for g in gpus]
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        names = visible.split(",") if visible else [str(i) for i in range(max(requested) + 1)]
        if max(requested) >= len(names):
            raise ValueError(f"requested GPUs {requested} but only {len(names)} are visible ({visible})")
        return [names[i].strip() for i in requested]

    def __run_pipeline(self, indices, prep_workers, sim_workers, devices, repeat):
        """Preparation workers fill a queue of prepared simulations; simulation workers drain it."""
        if prep_workers < 1 or sim_workers < 1:
            raise ValueError("prep_workers and sim_workers must be at least 1")
        queue = multiprocessing.Queue()
        num_sim = sim_workers * len(devices)
        # Keep one prepared simulation waiting per simulation worker (or per prep worker, if more).
        queue_depth = max(prep_workers, num_sim)
        prep_procs = []
        for i, chunk in enumerate(np.array_split(np.asarray(indices), prep_workers)):
            prep_procs.append(
                multiprocessing.Process(
                    name=f"prep_{i}",
                    target=self.prep_worker,
                    args=(queue, chunk, False, queue_depth, repeat),
                )
            )
            prep_procs[-1].start()
        sim_procs = [
            multiprocessing.Process(name=f"sim_{device}_{i}", target=self.sim_worker, args=(queue, device))
            for device in devices
            for i in range(sim_workers)
        ]
        for proc in sim_procs:
            proc.start()

        # If every simulation worker dies, nothing drains the queue and the prep workers would wait forever.
        while any(proc.is_alive() for proc in prep_procs):
            if not any(proc.is_alive() for proc in sim_procs):
                for proc in prep_procs:
                    proc.terminate()
                break
            time.sleep(1)
        for proc in prep_procs:
            proc.join()
        # Stop markers land behind every queued simulation, one per live simulation worker.
        for proc in sim_procs:
            if proc.is_alive():
                queue.put(SENTINEL)
        for proc in sim_procs:
            proc.join()
        if any(proc.exitcode != 0 for proc in sim_procs):
            # Prepared simulations may be left in the queue with no one to read them; don't
            # let the queue block this process's exit waiting for them to drain.
            queue.cancel_join_thread()

        missing = [
            int(index)
            for index in indices
            if not glob.glob(os.path.join(self.simulation_path, "results", f"signal_{int(index):06d}.np?"))
        ]
        if missing:
            failed = [proc.name for proc in sim_procs if proc.exitcode != 0]
            raise RuntimeError(
                f"{len(missing)} of {len(indices)} simulations have no results "
                f"(failed simulation workers: {failed or 'none'}); first missing: {missing[:10]}"
            )
        print(f"successfully joined {len(prep_procs)} preparation and {len(sim_procs)} simulation process(es)")

    def prep_worker(self, queue, indices, dry=False, num_prep_workers=1, repeat=False):
        start = time.time()
        count = 0
        while True:
            if queue.qsize() >= num_prep_workers and not dry:
                # Poll briefly: a long sleep here leaves a fast GPU idle between simulations.
                time.sleep(0.2)
                if time.time() - start > 300:
                    print(
                        f"prep worker has been inactive for {(time.time() - start) // 60} minutes"
                    )
                    start = time.time()
                continue
            try:
                index = indices[count]
            except:
                break
            if (
                repeat == -1 or self.repeat == -1
            ):  # repeat == -1 means we are not repeating simulations, but assigning indices statically (e.g. for large batch jobs)
                if os.path.exists(
                    os.path.join(
                        self.simulation_path,
                        f"results/signal_{str(index).zfill(6)}.npy",
                    )
                ):
                    print(f"simulation index {index} already exists, skipping")
                    count += 1
                    continue
            simulation = Simulation(
                self.sim_properties,
                self.phantom,
                self.transducer_set,
                self.sensor,
                simulation_path=self.simulation_path,
                index=index,
                gpu=self.gpu,
                num_threads=self.num_threads,
                dry=dry,
                additional_keys=self.additional_keys,
            )
            simulation.prep()

            queue.put(simulation)
            count += 1
            start = time.time()
            if count == len(indices):
                break

    def sim_worker(self, queue, device=None):
        """Run prepared simulations from the queue until a stop marker arrives.

        `device` pins this worker (and the k-Wave binaries it launches) to one GPU.
        """
        if device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = device
        while True:
            try:
                simulation = queue.get(timeout=300)
            except Empty:
                print("simulation worker has waited 5 minutes for a prepared simulation")
                continue
            if isinstance(simulation, str) and simulation == SENTINEL:
                break
            simulation.run()

    def simulate(self, index, dry=False):
        simulation = Simulation(
            self.sim_properties,
            self.phantom,
            self.transducer_set,
            self.sensor,
            simulation_path=self.simulation_path,
            index=index,
            gpu=self.gpu,
            num_threads=self.num_threads,
            dry=dry,
            additional_keys=self.additional_keys,
        )
        simulation.prep()
        simulation.run()

    def add_results(self):
        self.results = Results(os.path.join(self.simulation_path, "results"))

    def visualize_sensor_mask(
        self, index=(slice(0, -1, 1), slice(0, -1, 1), 0), body_surface_mask=None
    ):
        global_mask = np.zeros(self.phantom.mask.shape)
        sensor_voxels = np.divide(self.sensor.sensor_coords, self.phantom.voxel_dims)
        phantom_centroid = np.array(self.phantom.mask.shape) // 2
        recenter_matrix = np.broadcast_to(phantom_centroid, sensor_voxels.shape)
        sensor_voxels = sensor_voxels + recenter_matrix
        sensor_voxels_disc = np.ndarray.astype(np.round(sensor_voxels), int)

        for voxel in sensor_voxels_disc:
            if np.prod(np.where(voxel >= 0, 1, 0)) == 0:
                continue
            if np.prod(np.where(voxel < global_mask.shape, 1, 0)) == 0:
                continue
            global_mask[voxel[0], voxel[1], voxel[2]] = 1

        plt.imshow(
            self.phantom.mask[tuple(index)] + global_mask[tuple(index)], cmap="gray_r"
        )
        plt.imshow(
            np.stack(
                (
                    np.zeros_like(
                        global_mask[tuple(index)]
                    ),
                    global_mask[tuple(index)]
                    * 255,
                    global_mask[tuple(index)]
                    * 255,
                    global_mask[tuple(index)] * 255,
                ),
                axis=-1,
            )
        )

    def get_sensor_mask(self, pad=0):
        if pad == 0:
            mask_shape = self.phantom.mask.shape
        else:
            mask_shape = (
                self.phantom.mask.shape[0] + pad,
                self.phantom.mask.shape[1] + pad,
                self.phantom.mask.shape[2] + pad,
            )
        global_mask = np.zeros(mask_shape)
        sensor_voxels = np.divide(self.sensor.sensor_coords, self.phantom.voxel_dims)
        phantom_centroid = np.array(mask_shape) // 2
        recenter_matrix = np.broadcast_to(phantom_centroid, sensor_voxels.shape)
        sensor_voxels = sensor_voxels + recenter_matrix
        sensor_voxels_disc = np.ndarray.astype(np.round(sensor_voxels), int)

        for voxel in sensor_voxels_disc:
            if np.prod(np.where(voxel >= 0, 1, 0)) == 0:
                continue
            if np.prod(np.where(voxel < global_mask.shape, 1, 0)) == 0:
                continue
            global_mask[voxel[0], voxel[1], voxel[2]] = 1

        return global_mask

    def plot_ray_path(self, index, ax=None, save=False, save_path=None, cmap="viridis"):
        assert index < len(self), (
            f"index {index} is outside experiment length {len(self)}"
        )
        simulation = Simulation(
            self.sim_properties,
            self.phantom,
            self.transducer_set,
            self.sensor,
            simulation_path=self.simulation_path,
            index=index,
            gpu=self.gpu,
        )
        simulation.plot_medium_path(
            index, ax=ax, save=save, save_path=save_path, cmap=cmap
        )

    def check_added_keys(self, additional_keys):
        valid_keys = []
        allowed_keys = [
            "p",
            "p_max",
            "p_min",
            "p_rms",
            "p_max_all",
            "p_min_all",
            "p_final",
            "u",
            "u_max",
            "u_min",
            "u_rms",
            "u_max_all",
            "u_min_all",
            "u_final",
            "u_non_staggered",
            "I",
            "I_avg",
        ]
        for key in additional_keys:
            if key in allowed_keys:
                valid_keys.append(key)
            else:
                print(f'warning, requested flag "{key}" is not a valid flag, ignoring')
        return list(set(valid_keys))
