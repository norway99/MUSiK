import numpy as np
import os
import glob
import re
import multiprocessing
from queue import Empty
import time
import tqdm
from dataclasses import dataclass, field

from .utils import utils
from .phantom import Phantom
from .transducer_set import TransducerSet
from .simulation import Simulation, SimProperties
from .sensor import Sensor

from matplotlib import pyplot as plt

SENTINEL = "sentinel"


@dataclass
class Results:
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
    simulation_path: str = None
    sim_properties: SimProperties = None
    phantom: Phantom = None
    transducer_set: TransducerSet = None
    sensor: Sensor = None
    nodes: int = None
    results: Results = None
    indices: list = None
    gpu: bool = True
    workers: int = 2
    additional_keys: list = field(default_factory=list)
    repeat: int = None

    def __post_init__(self):
        if self.simulation_path is None:
            self.simulation_path = os.path.join(os.getcwd(), "experiment")
            os.makedirs(self.simulation_path, exist_ok=True)
        if self.workers is not None and self.workers > 3:
            print(
                "workers is the number of simulations being prepared simultaneously on a single gpu node. Having many workers is RAM intensive and may not decrease overall runtime"
            )
        slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
        if slurm_cpus is not None:
            print(f"Slurm environment detected. Found {slurm_cpus} cpus available")
            num_cpus = int(slurm_cpus)
            if num_cpus < self.workers:
                self.workers = num_cpus
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

    # save experiment
    def save(self, filepath=None):
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
            "workers": self.workers,
            "additional_keys": self.additional_keys,
        }
        utils.dict_to_json(dictionary, os.path.join(filepath, f"experiment.json"))

    # load experiment
    @classmethod
    def load(cls, filepath):
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
        experiment.workers = experiment_dict["workers"]
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

    # subdivide
    def subdivide(self, indices=None, repeat=False):
        if indices is None:
            indices = self.indices_to_run(indices, repeat=repeat)
        if len(indices) == 0:
            return None
        return np.array_split(np.array(indices), self.nodes)

    # run simulations by node
    def run(self, node=None, dry=False, repeat=False, workers=None, dry_fast=False):
        if workers is None:
            workers = self.workers
        assert os.path.exists(self.simulation_path), (
            "Attempting to run simulations but an experiment directory does not exist. Please save the experiment (my_experiment.save()) before running simulations."
        )

        if dry:
            indices = self.indices_to_run(repeat=True)
        else:
            indices = self.indices_to_run(repeat=repeat)

        if node is None:
            if self.nodes is None:
                self.nodes = 1
            if dry:
                self.run(0, dry=dry, repeat=repeat, workers=workers, dry_fast=dry_fast)
            else:
                for node in range(self.nodes):
                    self.run(
                        node, dry=dry, repeat=repeat, workers=workers, dry_fast=dry_fast
                    )
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
                if workers is None:
                    for index in indices:
                        self.simulate(index)
                else:
                    subdivisions = self.subdivide(repeat=repeat)
                    if subdivisions is None:
                        print("Found no more simulations to run.")
                    else:
                        print("running with {} workers\n".format(workers))
                        queue = multiprocessing.Queue()

                        if workers > 2:
                            simulations = np.array_split(
                                subdivisions[node], workers - 1
                            )
                            prep_procs = []
                            for i in range(workers - 1):
                                prep_procs.append(
                                    multiprocessing.Process(
                                        name=f"prep_{i}",
                                        target=self.prep_worker,
                                        args=(
                                            queue,
                                            simulations[i],
                                            dry,
                                            workers - 1,
                                            repeat,
                                        ),
                                    )
                                )
                                prep_procs[i].start()
                                time.sleep(10)
                        else:
                            prep_procs = [
                                multiprocessing.Process(
                                    name="prep",
                                    target=self.prep_worker,
                                    args=(queue, subdivisions[node], dry),
                                ),
                            ]
                            prep_procs[0].start()

                        run_proc = multiprocessing.Process(
                            name="run",
                            target=self.run_worker,
                            args=(queue, subdivisions[node], len(prep_procs)),
                        )
                        run_proc.start()

                        for prep_proc in prep_procs:
                            prep_proc.join()
                        run_proc.join()
                        print(
                            f"successfully joined {len(prep_procs)} preparation processes and 1 run process"
                        )

    def prep_worker(self, queue, indices, dry=False, num_prep_workers=1, repeat=False):
        start = time.time()
        count = 0
        while True:
            if queue.qsize() >= num_prep_workers and not dry:
                time.sleep(5)
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
                dry=dry,
                additional_keys=self.additional_keys,
            )
            simulation.prep()

            queue.put(simulation)
            count += 1
            start = time.time()
            if count == len(indices):
                break
        queue.put(SENTINEL)

    def run_worker(self, queue, indices, num_prep_workers):
        start = time.time()
        seen_sentinel_count = 0
        count = 0
        while True:
            try:
                simulation = queue.get(False)
            except Empty:
                time.sleep(1)
                if time.time() - start > 300:
                    print(
                        f"prep worker has been inactive for {(time.time() - start) // 60} minutes"
                    )
                continue
            if simulation == SENTINEL:
                seen_sentinel_count += 1
            else:
                simulation.run()
                count += 1
                start = time.time()
            if seen_sentinel_count == num_prep_workers:
                if count == len(indices):
                    break
                else:
                    assert False, (
                        f"counting all {seen_sentinel_count} prep_workers finished but only {count}/{len(indices)} indices matched"
                    )

    def simulate(self, index, dry=False):
        simulation = Simulation(
            self.sim_properties,
            self.phantom,
            self.transducer_set,
            self.sensor,
            simulation_path=self.simulation_path,
            index=index,
            gpu=self.gpu,
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
