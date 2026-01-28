import numpy as np
import os
import matplotlib.pyplot as plt
import tqdm
import multiprocessing
from scipy import interpolate
from scipy.interpolate import (
    LinearNDInterpolator,
    NearestNDInterpolator,
    RegularGridInterpolator,
)
from scipy.ndimage import gaussian_filter1d

from .utils import utils
from .utils import geometry
from .transducer import Focused, Planewave
from .experiment import Experiment, Results

from scipy.signal import hilbert


# Module-level globals for multiprocessing worker shared state
_worker_state = {}


def _init_worker(transducer_set, results, sim_properties, sensor_aperture_type, sound_speed):
    """Initialize worker process with shared data (called once per worker)."""
    _worker_state['transducer_set'] = transducer_set
    _worker_state['results'] = results
    _worker_state['sim_properties'] = sim_properties
    _worker_state['sensor_aperture_type'] = sensor_aperture_type
    _worker_state['sound_speed'] = sound_speed


def _process_line_worker(args):
    """
    Worker function for processing a single line.
    Uses shared state initialized by _init_worker.

    Args: (index, transducer_idx, transform_array, transmit_as_receive, attenuation_factor)
    """
    index, transducer_idx, transform_array, transmit_as_receive, attenuation_factor = args

    transducer_set = _worker_state['transducer_set']
    results = _worker_state['results']
    sim_properties = _worker_state['sim_properties']
    sound_speed = _worker_state['sound_speed']

    # Get transducer by index
    transducer, _ = transducer_set[transducer_idx]

    # Reconstruct transform matrix from array
    transform = np.asarray(transform_array)

    # Process the line
    processed = transducer.preprocess(
        transducer.make_scan_line(results[index][1], transmit_as_receive),
        results[index][0],
        sim_properties,
        attenuation_factor=attenuation_factor,
    )

    # Compute coordinates from time (same as __time_to_coord)
    times = results[index][0]
    distances = times * sound_speed / 2
    # Pad distances to 3D points: [d, 0, 0] -> apply transform
    dists_padded = np.pad(
        distances[..., None], ((0, 0), (0, 2)), mode="constant", constant_values=0
    )
    # Apply homogeneous transform: add 1s column, multiply, take first 3 cols
    ones = np.ones((dists_padded.shape[0], 1))
    points_homogeneous = np.hstack([dists_padded, ones])
    coords = (transform @ points_homogeneous.T).T[:, :3]

    return times, coords, processed


class PreprocessedDataLoader:
    """
    Lazy-loading accessor for preprocessed data stored in batch files.

    Loads batches on-demand and keeps only a limited number in memory
    to prevent OOM errors with large datasets.
    """

    def __init__(self, save_dir, max_cached_batches=2):
        import glob
        self.save_dir = save_dir
        self.max_cached_batches = max_cached_batches

        # Find all batch files and build index
        self.batch_files = sorted(glob.glob(os.path.join(save_dir, "preprocess_batch_*.npz")))
        if not self.batch_files:
            raise FileNotFoundError(f"No preprocess batch files found in {save_dir}")

        # Build index mapping: for each batch, store (start_index, end_index, filepath)
        self.batch_index = []
        for batch_file in self.batch_files:
            data = np.load(batch_file, allow_pickle=True)
            start_idx = int(data['start_index'])
            end_idx = int(data['end_index'])
            self.batch_index.append((start_idx, end_idx, batch_file))
            data.close()

        # Cache for loaded batches: {batch_idx: (times, coords, processed)}
        self._cache = {}
        self._cache_order = []  # Track access order for LRU eviction

        # Total length
        self.total_length = self.batch_index[-1][1] if self.batch_index else 0

    def __len__(self):
        return self.total_length

    def _find_batch_for_index(self, idx):
        """Find which batch contains the given index."""
        for batch_idx, (start, end, _) in enumerate(self.batch_index):
            if start <= idx < end:
                return batch_idx
        raise IndexError(f"Index {idx} out of range [0, {self.total_length})")

    def _load_batch(self, batch_idx):
        """Load a batch into cache, evicting old batches if needed."""
        if batch_idx in self._cache:
            # Move to end of access order (most recently used)
            self._cache_order.remove(batch_idx)
            self._cache_order.append(batch_idx)
            return

        # Evict oldest batches if cache is full
        while len(self._cache) >= self.max_cached_batches:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]

        # Load the batch
        _, _, batch_file = self.batch_index[batch_idx]
        data = np.load(batch_file, allow_pickle=True)
        self._cache[batch_idx] = (
            list(data['times']),
            list(data['coords']),
            list(data['processed'])
        )
        self._cache_order.append(batch_idx)
        data.close()

    def get_range(self, start_idx, end_idx):
        """
        Get times, coords, processed for a range of indices.

        Returns three lists containing the data for indices [start_idx, end_idx).
        """
        times = []
        coords = []
        processed = []

        current_idx = start_idx
        while current_idx < end_idx:
            batch_idx = self._find_batch_for_index(current_idx)
            self._load_batch(batch_idx)

            batch_start, batch_end, _ = self.batch_index[batch_idx]
            batch_times, batch_coords, batch_processed = self._cache[batch_idx]

            # Calculate slice within this batch
            local_start = current_idx - batch_start
            local_end = min(end_idx - batch_start, batch_end - batch_start)

            times.extend(batch_times[local_start:local_end])
            coords.extend(batch_coords[local_start:local_end])
            processed.extend(batch_processed[local_start:local_end])

            current_idx = batch_start + local_end

        return times, coords, processed

    def clear_cache(self):
        """Clear all cached batches to free memory."""
        self._cache.clear()
        self._cache_order.clear()


class Reconstruction:
    def __init__(
        self,
        experiment=None,
    ):
        if isinstance(experiment, Experiment):
            self.simulation_path = experiment.simulation_path
            self.sim_properties = experiment.sim_properties
            self.phantom = experiment.phantom
            self.transducer_set = experiment.transducer_set
            self.sensor = experiment.sensor
            self.results = experiment.results
            self.experiment = experiment
        elif isinstance(experiment, str):
            self.experiment = Experiment.load(experiment)
            self.simulation_path = self.experiment.simulation_path
            self.sim_properties = self.experiment.sim_properties
            self.phantom = self.experiment.phantom
            self.transducer_set = self.experiment.transducer_set
            self.sensor = self.experiment.sensor
            self.results = self.experiment.results
        else:
            assert False, "Please provide an experiment to reconstruct."

    def __len__(self):
        if self.transducer_set is None:
            return 0
        return sum(
            [
                transducer.get_num_rays()
                for transducer in self.transducer_set.transmit_transducers()
            ]
        )

    def add_results(
        self,
    ):
        self.results = Results(os.path.join(self.simulation_path, "results"))


class DAS(Reconstruction):
    def __init__(self, experiment=None):
        # for transducer in experiment.transducer_set.transducers:
        # if not isinstance(transducer, Focused):
        #     print("Warning: attempting to instantiate DAS reconstruction class but transducer set does not exclusively contain focused transducers.")
        #     break
        super().__init__(experiment)

    def plot_ray_path(self, index, ax=None, save=False, save_path=None, cmap="viridis"):
        if ax is None:
            fig, ax = plt.subplots(2, 1, figsize=(15, 5))
            self.experiment.plot_ray_path(index, ax=ax)
            centerline = (
                self.sim_properties.grid_size[1] // 2 - self.sim_properties.PML_size[1]
            )
            ax[0].plot(
                self.experiment.results[index][0]
                / self.experiment.phantom.voxel_dims[0]
                * 1540
                / 2,
                self.experiment.results[index][1].T / 20 + centerline,
                linewidth=0.1,
                color="cyan",
                alpha=0.5,
            )
            ax[0].set_ylim(0, centerline * 2)
            ax[1].plot(
                self.experiment.results[index][0]
                / self.experiment.phantom.voxel_dims[0]
                * 1540
                / 2,
                self.experiment.results[index][1].T / 20 + centerline,
                linewidth=0.1,
                color="cyan",
                alpha=0.5,
            )
            ax[1].set_ylim(0, centerline * 2)
            plt.show()

    def __time_to_coord(self, t, transform):
        dists = t * self.phantom.baseline[0] / 2

        dists = np.pad(
            dists[..., None], ((0, 0), (0, 2)), mode="constant", constant_values=0
        )
        coords = transform.apply_to_points(dists)
        return coords

    def process_line(
        self,
        index,
        transducer,
        transform,
        transmit_as_receive=True,
        attenuation_factor=1,
    ):
        processed = transducer.preprocess(
            transducer.make_scan_line(self.results[index][1], transmit_as_receive),
            self.results[index][0],
            self.sim_properties,
            attenuation_factor=attenuation_factor,
        )
        coords = self.__time_to_coord(self.results[index][0], transform)
        times = self.results[index][0]
        return times, coords, processed

    def preprocess_data(
        self, global_transforms=True, workers=8, attenuation_factor=1,
        save_dir=None, batch_size=10000, resume=True
    ):
        """
        Preprocess simulation data for reconstruction.

        Parameters
        ----------
        global_transforms : bool
            Whether to apply global transforms
        workers : int
            Number of parallel workers
        attenuation_factor : float
            Time gain compensation factor
        save_dir : str, optional
            Directory to save intermediate results. When provided, results are
            saved in batches to reduce memory usage and allow resumption.
        batch_size : int
            Number of rays to process before saving (default 10000)
        resume : bool
            If True and save_dir exists with partial results, resume from last batch

        Returns
        -------
        times, coords, processed : lists or str
            If save_dir is None, returns three lists.
            If save_dir is provided, returns path to save directory.
        """
        running_index_list = np.cumsum(
            [
                transducer.get_num_rays()
                for transducer in self.transducer_set.transmit_transducers()
            ]
        )
        total_results = len(self.results)
        sensor_is_transmit_as_receive = self.sensor.aperture_type == "transmit_as_receive"

        # Helper to get transducer info for a given index
        def get_transducer_for_index(index, transducer_count, transducer, transducer_transform):
            while transducer_count < len(running_index_list) - 1 and index > running_index_list[transducer_count] - 1:
                transducer_count += 1
                transducer, transducer_transform = self.transducer_set[transducer_count]
            return transducer_count, transducer, transducer_transform

        # Helper to get transform for a given index
        def get_transform(index, transducer, transducer_transform, transducer_count):
            local_index = index - (running_index_list[transducer_count - 1] if transducer_count > 0 else 0)
            if global_transforms:
                return transducer_transform * transducer.ray_transforms[local_index]
            else:
                return transducer.ray_transforms[local_index]

        # Setup for incremental saving
        start_index = 0
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Check for existing progress to resume
            if resume:
                import glob
                existing_files = sorted(glob.glob(os.path.join(save_dir, "preprocess_batch_*.npz")))
                if existing_files:
                    for f in existing_files:
                        try:
                            data = np.load(f, allow_pickle=True)
                            batch_end = int(data['end_index'])
                            start_index = max(start_index, batch_end)
                        except Exception as e:
                            print(f"Warning: Could not load {f}: {e}")
                    if start_index > 0:
                        print(f"Resuming from index {start_index} (found {len(existing_files)} existing batches)")

        if save_dir is not None:
            # Process and save in batches - build inputs incrementally
            current_batch = start_index // batch_size
            num_batches = (total_results + batch_size - 1) // batch_size

            # Initialize transducer state for start_index
            transducer_count = 0
            transducer, transducer_transform = self.transducer_set[0]
            if start_index > 0:
                transducer_count, transducer, transducer_transform = get_transducer_for_index(
                    start_index, transducer_count, transducer, transducer_transform
                )

            for batch_idx in tqdm.tqdm(range(current_batch, num_batches), desc="Processing batches"):
                batch_start_idx = batch_idx * batch_size
                batch_end_idx = min(batch_start_idx + batch_size, total_results)

                # Skip batches before start_index
                if batch_end_idx <= start_index:
                    continue

                # Adjust for partial first batch when resuming
                actual_start = max(batch_start_idx, start_index)

                # Build inputs for this batch only - pass transducer INDEX, not object
                batch_inputs = []
                for index in range(actual_start, batch_end_idx):
                    transducer_count, transducer, transducer_transform = get_transducer_for_index(
                        index, transducer_count, transducer, transducer_transform
                    )
                    transform = get_transform(index, transducer, transducer_transform, transducer_count)
                    batch_inputs.append((
                        index, transducer_count, transform.get(),
                        sensor_is_transmit_as_receive, attenuation_factor
                    ))

                # Process batch
                if workers > 1:
                    sound_speed = self.phantom.baseline[0]
                    with multiprocessing.Pool(
                        workers,
                        initializer=_init_worker,
                        initargs=(self.transducer_set, self.results, self.sim_properties, self.sensor.aperture_type, sound_speed)
                    ) as p:
                        batch_results = []
                        for result in tqdm.tqdm(
                            p.imap(_process_line_worker, batch_inputs, chunksize=100),
                            total=len(batch_inputs),
                            desc="  Rays",
                            leave=False
                        ):
                            batch_results.append(result)
                else:
                    # For single worker, set up worker state and use same function
                    sound_speed = self.phantom.baseline[0]
                    _init_worker(self.transducer_set, self.results, self.sim_properties, self.sensor.aperture_type, sound_speed)
                    batch_results = []
                    for input_data in tqdm.tqdm(batch_inputs, desc="  Rays", leave=False):
                        batch_results.append(_process_line_worker(input_data))

                # Extract and save batch
                batch_times = [r[0] for r in batch_results]
                batch_coords = [r[1] for r in batch_results]
                batch_processed = [r[2] for r in batch_results]

                batch_file = os.path.join(save_dir, f"preprocess_batch_{batch_idx:04d}.npz")
                np.savez_compressed(
                    batch_file,
                    times=np.array(batch_times, dtype=object),
                    coords=np.array(batch_coords, dtype=object),
                    processed=np.array(batch_processed, dtype=object),
                    start_index=actual_start,
                    end_index=batch_end_idx
                )
                print(f"Saved batch {batch_idx} (indices {actual_start}-{batch_end_idx-1}) to {batch_file}")

                # Clear memory
                del batch_results, batch_times, batch_coords, batch_processed, batch_inputs

            return save_dir
        else:
            # Original behavior - process all at once, build lightweight inputs list
            transducer_count = 0
            transducer, transducer_transform = self.transducer_set[0]

            inputs = []
            for index in range(total_results):
                transducer_count, transducer, transducer_transform = get_transducer_for_index(
                    index, transducer_count, transducer, transducer_transform
                )
                transform = get_transform(index, transducer, transducer_transform, transducer_count)
                # Pass transducer index, not object
                inputs.append((
                    index, transducer_count, transform.get(),
                    sensor_is_transmit_as_receive, attenuation_factor
                ))

            results = []
            sound_speed = self.phantom.baseline[0]
            if workers > 1:
                with multiprocessing.Pool(
                    workers,
                    initializer=_init_worker,
                    initargs=(self.transducer_set, self.results, self.sim_properties, self.sensor.aperture_type, sound_speed)
                ) as p:
                    results = list(tqdm.tqdm(
                        p.imap(_process_line_worker, inputs, chunksize=100),
                        total=len(inputs)
                    ))
            else:
                # For single worker, set up worker state and use same function
                _init_worker(self.transducer_set, self.results, self.sim_properties, self.sensor.aperture_type, sound_speed)
                for input_data in tqdm.tqdm(inputs, total=len(inputs)):
                    results.append(_process_line_worker(input_data))

            times = [r[0] for r in results]
            coords = [r[1] for r in results]
            processed = [r[2] for r in results]
            return times, coords, processed

    def load_preprocessed_data(self, save_dir):
        """
        Load preprocessed data from saved batch files.

        Parameters
        ----------
        save_dir : str
            Directory containing preprocess_batch_*.npz files

        Returns
        -------
        times, coords, processed : lists
            Combined data from all batch files

        Note: This loads all data into memory. For large datasets, use
        PreprocessedDataLoader for lazy loading instead.
        """
        import glob
        batch_files = sorted(glob.glob(os.path.join(save_dir, "preprocess_batch_*.npz")))

        if not batch_files:
            raise FileNotFoundError(f"No preprocess batch files found in {save_dir}")

        times = []
        coords = []
        processed = []

        for batch_file in tqdm.tqdm(batch_files, desc="Loading batches"):
            data = np.load(batch_file, allow_pickle=True)
            times.extend(list(data['times']))
            coords.extend(list(data['coords']))
            processed.extend(list(data['processed']))

        return times, coords, processed

    def create_preprocessed_data_loader(self, save_dir):
        """
        Create a lazy-loading data accessor for preprocessed data.

        This avoids loading all batches into memory at once by loading
        batches on-demand as data is accessed.

        Parameters
        ----------
        save_dir : str
            Directory containing preprocess_batch_*.npz files

        Returns
        -------
        PreprocessedDataLoader
            Lazy-loading accessor for times, coords, processed data
        """
        return PreprocessedDataLoader(save_dir)

    def plot_scatter(self, scale=5000, workers=1):
        colorme = lambda x: (
            [1, 0, 0]
            if x % 7 == 0
            else [0, 1, 0]
            if x % 7 == 1
            else [0, 0, 1]
            if x % 7 == 2
            else [1, 1, 0]
            if x % 7 == 3
            else [1, 0, 1]
            if x % 7 == 4
            else [0, 1, 1]
            if x % 7 == 5
            else [1, 1, 1]
        )

        times, coords, processed = self.preprocess_data(workers=workers)
        coords = np.stack(coords, axis=0)
        processed = np.stack(processed, axis=0)
        times = np.stack(times, axis=0)

        transducer_lens = [t.get_num_rays() for t in self.transducer_set.transducers]
        base_color = np.array(
            [
                colorme(i)
                for i in range(len(transducer_lens))
                for j in range(transducer_lens[i] * len(times[i]))
            ]
        )

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        coords = np.reshape(coords, (-1, 3))
        processed = np.reshape(processed, (-1,))
        times = np.reshape(times, (-1,))
        base_color = np.reshape(base_color, (-1, 3))
        intensity = np.clip(processed, 0, scale) / scale

        colors = (
            np.array([1, 1, 1])
            - np.broadcast_to(intensity[:, None], base_color.shape) * base_color
        )

        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=times * 100000, alpha=0.002)

        ax.set_aspect("equal")

    def get_image(
        self, bounds=None, matsize=256, dimensions=3, downsample=1, workers=8, tgc=1
    ):
        assert dimensions in [2, 3], print("Image can be 2 or 3 dimensional")
        assert downsample > 0 and downsample <= 1, print(
            "Downsample must be a float on (0,1]"
        )

        times, coords, processed = self.preprocess_data(
            global_transforms=True, workers=workers, attenuation_factor=tgc
        )

        if bounds is None:
            flat_coords = np.concatenate(coords, axis=0).reshape(-1, 3)
            bounds = np.array(
                [
                    (np.min(flat_coords[:, 0]), np.max(flat_coords[:, 0])),
                    (np.min(flat_coords[:, 1]), np.max(flat_coords[:, 1])),
                    (np.min(flat_coords[:, 2]), np.max(flat_coords[:, 2])),
                ]
            )
        elif (
            type(bounds) is list or type(bounds) is tuple or type(bounds) is np.ndarray
        ):
            bounds = np.array(bounds)
        elif type(bounds) is float:
            bounds = np.array([(-bounds, bounds), (-bounds, bounds), (-bounds, bounds)])
        else:
            print("provide bounds as a list, tuple, numpy array, or float")
            return 0

        bounds_avg = (
            bounds[0, 1]
            - bounds[0, 0]
            + bounds[1, 1]
            - bounds[1, 0]
            + bounds[2, 1]
            - bounds[2, 0]
        ) / 3
        X = np.linspace(
            bounds[0, 0],
            bounds[0, 1],
            int((bounds[0, 1] - bounds[0, 0]) / bounds_avg * matsize),
        )
        Y = np.linspace(
            bounds[1, 0],
            bounds[1, 1],
            int((bounds[1, 1] - bounds[1, 0]) / bounds_avg * matsize),
        )
        Z = np.linspace(
            bounds[2, 0],
            bounds[2, 1],
            int((bounds[2, 1] - bounds[2, 0]) / bounds_avg * matsize),
        )

        if dimensions == 2:
            X, Y = np.meshgrid(X, Y, indexing="ij")
        else:
            X, Y, Z = np.meshgrid(X, Y, Z, indexing="ij")

        signals = []
        count = 0

        for i, transducer in tqdm.tqdm(enumerate(self.transducer_set.transducers)):
            subset_coords = np.stack(
                coords[count : int(count + transducer.get_num_rays())], axis=0
            ).reshape(-1, 3)
            subset_processed = np.stack(
                processed[count : int(count + transducer.get_num_rays())], axis=0
            ).reshape(-1)

            if downsample != 1:
                subset_processed = gaussian_filter1d(
                    subset_processed.reshape(-1, transducer.get_num_rays()),
                    int(1 / downsample),
                    axis=-1,
                ).reshape(-1)
                subset_coords = subset_coords[:: int(1 / downsample)]
                subset_processed = subset_processed[:: int(1 / downsample)]

            if dimensions == 2:
                interp = LinearNDInterpolator(subset_coords[:, :2], subset_processed)
                signals.append(interp(X, Y))
            else:
                interp = NearestNDInterpolator(subset_coords, subset_processed)
                signal = interp(X, Y, Z)

                ray_length = subset_coords.shape[0] / transducer.get_num_rays()
                convex_coords = np.stack(
                    [subset_coords[0]]
                    + [
                        subset_coords[int(i * ray_length - 1)]
                        for i in range(1, transducer.get_num_rays())
                    ]
                    + [subset_coords[-1]]
                )
                convex_hull_mask = utils.compute_convex_hull_mask(
                    convex_coords, np.stack([X, Y, Z], axis=-1)
                )

                signals.append(signal * convex_hull_mask)

            count += transducer.get_num_rays()

        combined_signals = np.stack(signals, axis=0)
        masked_signals = np.ma.masked_array(
            combined_signals, np.isnan(combined_signals)
        )
        image = np.ma.average(masked_signals, axis=0)
        image = image.filled(np.nan)

        return image

    def get_signals(
        self, bounds=None, matsize=256, dimensions=3, downsample=1, workers=8, tgc=1,
        save_dir=None, batch_size=100, resume=True, preprocess_batch_size=10000
    ):
        """
        Get interpolated signals for each transducer.

        Parameters
        ----------
        bounds : array-like or float, optional
            Bounds for the reconstruction grid
        matsize : int
            Matrix size for reconstruction
        dimensions : int
            2 or 3 dimensional reconstruction
        downsample : float
            Downsample factor (0, 1]
        workers : int
            Number of parallel workers
        tgc : float
            Time gain compensation factor
        save_dir : str, optional
            Directory to save intermediate results. When provided, both preprocessing
            and signals are saved in batches to reduce memory usage and allow resumption.
        batch_size : int
            Number of transducers to process before saving signals (default 100)
        resume : bool
            If True and save_dir exists with partial results, resume from last batch
        preprocess_batch_size : int
            Number of rays to process before saving during preprocessing (default 10000)

        Returns
        -------
        signals : list or str
            If save_dir is None, returns list of signal arrays.
            If save_dir is provided, returns path to final signals.npz file.
        """
        assert dimensions in [2, 3], print("Image can be 2 or 3 dimensional")
        assert downsample > 0 and downsample <= 1, print(
            "Downsample must be a float on (0,1]"
        )

        # Load or compute preprocessed data
        use_lazy_loading = False
        data_loader = None
        coords = None
        processed = None

        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Check if preprocessing is already complete
            import glob
            preprocess_files = glob.glob(os.path.join(save_dir, "preprocess_batch_*.npz"))

            if preprocess_files and resume:
                # Check if preprocessing is complete (last batch end_index matches total results)
                last_file = sorted(preprocess_files)[-1]
                data = np.load(last_file, allow_pickle=True)
                last_end = int(data['end_index'])

                if last_end >= len(self.results):
                    print(f"Using lazy loading for preprocessed data from {save_dir}...")
                    data_loader = self.create_preprocessed_data_loader(save_dir)
                    use_lazy_loading = True
                else:
                    print(f"Resuming preprocessing from index {last_end}...")
                    self.preprocess_data(
                        global_transforms=False, workers=workers, attenuation_factor=tgc,
                        save_dir=save_dir, batch_size=preprocess_batch_size, resume=True
                    )
                    data_loader = self.create_preprocessed_data_loader(save_dir)
                    use_lazy_loading = True
            else:
                print(f"Running preprocessing with incremental saving to {save_dir}...")
                self.preprocess_data(
                    global_transforms=False, workers=workers, attenuation_factor=tgc,
                    save_dir=save_dir, batch_size=preprocess_batch_size, resume=resume
                )
                data_loader = self.create_preprocessed_data_loader(save_dir)
                use_lazy_loading = True
        else:
            times, coords, processed = self.preprocess_data(
                global_transforms=False, workers=workers, attenuation_factor=tgc
            )

        if bounds is None:
            if use_lazy_loading:
                # With lazy loading, we can't easily compute bounds without loading all data
                # Load a sample to estimate bounds
                print("Warning: bounds=None with lazy loading - estimating from sample data")
                sample_times, sample_coords, sample_processed = data_loader.get_range(0, min(10000, len(data_loader)))
                flat_coords = np.concatenate(sample_coords, axis=0).reshape(-1, 3)
                bounds = np.array(
                    [
                        (np.min(flat_coords[:, 0]) * 1.1, np.max(flat_coords[:, 0]) * 1.1),
                        (np.min(flat_coords[:, 1]) * 1.1, np.max(flat_coords[:, 1]) * 1.1),
                        (np.min(flat_coords[:, 2]) * 1.1, np.max(flat_coords[:, 2]) * 1.1),
                    ]
                )
                print(f"Estimated bounds from sample: {bounds}")
                data_loader.clear_cache()  # Free the sample data
            else:
                flat_coords = np.concatenate(coords, axis=0).reshape(-1, 3)
                bounds = np.array(
                    [
                        (np.min(flat_coords[:, 0]), np.max(flat_coords[:, 0])),
                        (np.min(flat_coords[:, 1]), np.max(flat_coords[:, 1])),
                        (np.min(flat_coords[:, 2]), np.max(flat_coords[:, 2])),
                    ]
                )
        elif (
            type(bounds) is list or type(bounds) is tuple or type(bounds) is np.ndarray
        ):
            bounds = np.array(bounds)
        elif type(bounds) is float:
            bounds = np.array([(-bounds, bounds), (-bounds, bounds), (-bounds, bounds)])
        else:
            print("provide bounds as a list, tuple, numpy array, or float")
            return 0

        bounds_avg = (
            bounds[0, 1]
            - bounds[0, 0]
            + bounds[1, 1]
            - bounds[1, 0]
            + bounds[2, 1]
            - bounds[2, 0]
        ) / 3
        X = np.linspace(
            bounds[0, 0],
            bounds[0, 1],
            int((bounds[0, 1] - bounds[0, 0]) / bounds_avg * matsize),
        )
        Y = np.linspace(
            bounds[1, 0],
            bounds[1, 1],
            int((bounds[1, 1] - bounds[1, 0]) / bounds_avg * matsize),
        )
        Z = np.linspace(
            bounds[2, 0],
            bounds[2, 1],
            int((bounds[2, 1] - bounds[2, 0]) / bounds_avg * matsize),
        )

        if dimensions == 2:
            X, Y = np.meshgrid(X, Y, indexing="ij")
        else:
            X, Y, Z = np.meshgrid(X, Y, Z, indexing="ij")

        # Setup for incremental saving
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Check for existing progress to resume
            start_transducer = 0
            existing_batches = []
            if resume:
                import glob
                existing_files = sorted(glob.glob(os.path.join(save_dir, "signals_batch_*.npz")))
                if existing_files:
                    # Find the last completed batch
                    for f in existing_files:
                        try:
                            data = np.load(f)
                            batch_end = int(data['end_transducer'])
                            existing_batches.append(f)
                            start_transducer = max(start_transducer, batch_end)
                        except Exception as e:
                            print(f"Warning: Could not load {f}: {e}")
                    if start_transducer > 0:
                        print(f"Resuming from transducer {start_transducer} (found {len(existing_batches)} existing batches)")

        signals = []
        count = 0
        batch_signals = []
        batch_indices = []
        current_batch = 0

        # Calculate starting count for resumed runs
        if save_dir is not None and start_transducer > 0:
            for i in range(start_transducer):
                count += self.transducer_set.transducers[i].get_num_rays()
            current_batch = (start_transducer // batch_size)

        num_transducers = len(self.transducer_set.transducers)
        start_idx = start_transducer if save_dir is not None else 0

        for i, transducer in tqdm.tqdm(
            enumerate(self.transducer_set.transducers),
            total=num_transducers,
            initial=start_idx
        ):
            # Skip already processed transducers when resuming
            if save_dir is not None and i < start_transducer:
                continue

            # Get data for this transducer - either from lazy loader or pre-loaded lists
            num_rays = transducer.get_num_rays()
            if use_lazy_loading:
                _, subset_coords_list, subset_processed_list = data_loader.get_range(count, count + num_rays)
                subset_coords = np.stack(subset_coords_list, axis=0).reshape(-1, 3)
                subset_processed = np.stack(subset_processed_list, axis=0).reshape(-1)
            else:
                subset_coords = np.stack(
                    coords[count : int(count + num_rays)], axis=0
                ).reshape(-1, 3)
                subset_processed = np.stack(
                    processed[count : int(count + num_rays)], axis=0
                ).reshape(-1)

            if downsample != 1:
                subset_processed = gaussian_filter1d(
                    subset_processed.reshape(-1, transducer.get_num_rays()),
                    int(1 / downsample),
                    axis=-1,
                ).reshape(-1)
                subset_coords = subset_coords[:: int(1 / downsample)]
                subset_processed = subset_processed[:: int(1 / downsample)]

            if dimensions == 2:
                interp = LinearNDInterpolator(subset_coords[:, :2], subset_processed)
                signal = interp(X, Y)
            else:
                interp = NearestNDInterpolator(subset_coords, subset_processed)
                signal = interp(X, Y, Z)

                ray_length = subset_coords.shape[0] / transducer.get_num_rays()
                convex_coords = np.stack(
                    [subset_coords[0]]
                    + [
                        subset_coords[int(j * ray_length - 1)]
                        for j in range(1, transducer.get_num_rays())
                    ]
                    + [subset_coords[-1]]
                )
                convex_hull_mask = utils.compute_convex_hull_mask(
                    convex_coords, np.stack([X, Y, Z], axis=-1)
                )
                signal = signal * convex_hull_mask

            count += transducer.get_num_rays()

            if save_dir is not None:
                batch_signals.append(signal)
                batch_indices.append(i)

                # Save batch when batch_size is reached or at the end
                if len(batch_signals) >= batch_size or i == num_transducers - 1:
                    batch_file = os.path.join(save_dir, f"signals_batch_{current_batch:04d}.npz")
                    np.savez_compressed(
                        batch_file,
                        signals=np.array(batch_signals),
                        indices=np.array(batch_indices),
                        start_transducer=batch_indices[0],
                        end_transducer=batch_indices[-1] + 1
                    )
                    print(f"Saved batch {current_batch} (transducers {batch_indices[0]}-{batch_indices[-1]}) to {batch_file}")
                    batch_signals = []
                    batch_indices = []
                    current_batch += 1
            else:
                signals.append(signal)

        if save_dir is not None:
            # Combine all batches into final file
            print("Combining batches into final signals file...")
            import glob
            all_batch_files = sorted(glob.glob(os.path.join(save_dir, "signals_batch_*.npz")))
            all_signals = []
            for batch_file in all_batch_files:
                data = np.load(batch_file)
                for sig in data['signals']:
                    all_signals.append(sig)

            final_file = os.path.join(save_dir, "signals.npz")
            np.savez_compressed(final_file, signals=all_signals)
            print(f"Saved combined signals to {final_file}")
            return final_file

        return signals


class Compounding(Reconstruction):
    def __init__(self, experiment=None):
        super().__init__(experiment)

    def __get_element_centroids(self):  # in global coordinates
        sensor_coords = self.sensor.sensor_coords
        sensors_per_el = self.sensor.sensors_per_el
        element_centroids = np.zeros((sensors_per_el.size, 3), dtype=np.float32)
        pos = 0
        for entry in range(sensors_per_el.size):
            element_centroids[entry] = np.mean(
                sensor_coords[int(pos) : int(pos + sensors_per_el[entry]), :], axis=0
            )
            pos += sensors_per_el[entry]
        return element_centroids

    def compound(
        self,
        workers=8,
        resolution_multiplier=1,
        combine=True,
        pressure_field=None,
        pressure_field_resolution=None,
        return_local=False,
        attenuation_factor=1,
        volumetric=False,
        save_intermediates=False,
    ):
        if isinstance(self.transducer_set[0], Focused):
            # do nothing
            pass
        else:
            # still do nothing
            pass

        matrix_dims = self.phantom.matrix_dims
        voxel_dims = self.phantom.voxel_dims

        c0 = self.phantom.baseline[0]
        dt = (self.results[0][0][-1] - self.results[0][0][0]) / self.results[0][
            0
        ].shape[0]

        resolution = (
            max(dt * c0, 2 * c0 / self.transducer_set.get_lowest_frequency())
            / 4
            / resolution_multiplier
        )  # make sure this works

        x = np.arange(
            -matrix_dims[0] * voxel_dims[0] / 2 + voxel_dims[0] / 2,
            matrix_dims[0] * voxel_dims[0] / 2 + voxel_dims[0] / 2,
            step=resolution,
        )
        y = np.arange(
            -matrix_dims[1] * voxel_dims[1] / 2 + voxel_dims[1] / 2,
            matrix_dims[1] * voxel_dims[1] / 2 + voxel_dims[1] / 2,
            step=resolution,
        )
        z = np.arange(
            -matrix_dims[2] * voxel_dims[2] / 2 + voxel_dims[2] / 2,
            matrix_dims[2] * voxel_dims[2] / 2 + voxel_dims[2] / 2,
            step=resolution,
        )

        image_matrix = np.zeros((len(x), len(y), len(z)))

        # note that origin is at center of the 3d image in global coordinate system

        element_centroids = self.__get_element_centroids()

        arguments = []

        transducer_count = 0
        transducer = self.transducer_set.transmit_transducers()[transducer_count]
        transducer_transform = self.transducer_set.transmit_poses()[transducer_count]

        running_index_list = np.cumsum(
            [
                transducer.get_num_rays()
                for transducer in self.transducer_set.transmit_transducers()
            ]
        )

        for index in tqdm.tqdm(range(len(self.results))):
            if index > running_index_list[transducer_count] - 1:
                transducer_count += 1
                transducer = self.transducer_set.transmit_transducers()[
                    transducer_count
                ]
                transducer_transform = self.transducer_set.transmit_poses()[
                    transducer_count
                ]

            arguments.append(
                (
                    index,
                    running_index_list,
                    transducer_count,
                    transducer,
                    transducer_transform,
                    x,
                    y,
                    z,
                    c0,
                    dt,
                    element_centroids,
                    resolution,
                    return_local,
                    pressure_field,
                    pressure_field_resolution,
                    attenuation_factor,
                    volumetric,
                    save_intermediates,
                )
            )

        if save_intermediates and not os.path.exists(f"{self.simulation_path}/reconstruct"):
            os.makedirs(f"{self.simulation_path}/reconstruct")

        print(f"running reconstruction on {len(arguments)} rays")
        if workers > 1:
            with multiprocessing.Pool(workers) as p:
                if not save_intermediates:
                    image_matrices = list(
                        p.starmap(self.scanline_reconstruction, arguments)
                    )
                else:
                    p.starmap(self.scanline_reconstruction, arguments)
        else:
            if not save_intermediates:
                image_matrices = []
                for argument in arguments:
                    image_matrices.append(self.scanline_reconstruction(*argument))
            else:
                for argument in arguments:
                    self.scanline_reconstruction(*argument)

        if save_intermediates:
            return 0  # load in intermediate files and save sum

        if combine:
            return np.sum(np.stack(image_matrices, axis=0), axis=0)
        else:
            return image_matrices

    def scanline_reconstruction(
        self,
        index,
        running_index_list,
        transducer_count,
        transducer,
        transducer_transform,
        x,
        y,
        z,
        c0,
        dt,
        element_centroids,
        resolution,
        return_local,
        pressure_field=None,
        pressure_field_resolution=None,
        attenuation_factor=1,
        volumetric=False,
        save_intermediates=False,
    ):

        print(f"running reconstruction on ray {index}")
        # starttime=time.time()
        # fetch steering angle
        if index > running_index_list[transducer_count] - 1:
            transducer_count += 1
            transducer, transducer_transform = self.transducer_set[transducer_count]
        steering_angle = transducer.steering_angles[
            index - running_index_list[transducer_count]
        ]

        # get dt
        dt = (self.results[index][0][-1] - self.results[index][0][0]) / (
            self.results[index][0].shape[0] - 1
        )

        # run transducer signal preprocessing
        preprocessed_data = transducer.preprocess(
            self.results[index][1],
            self.results[index][0],
            self.sim_properties,
            window_factor=4,
            saft=True,
            attenuation_factor=attenuation_factor,
        )
        preprocessed_data = np.array(preprocessed_data).astype(np.float32)

        # pad the timesignal if duration < long diagonal
        if len(preprocessed_data.shape) == 2:
            preprocessed_data = np.pad(
                preprocessed_data,
                ((0, 0), (0, int(preprocessed_data.shape[1] * 2))),
            )

        if isinstance(transducer, Planewave):
            steering_transform = geometry.Transform(rotation=[steering_angle, 0, 0])
            timedelay = (
                transducer.width
                / 2
                * np.abs(np.sin(np.max(transducer.steering_angles)))
            )  # timedelay gets padded on according to the max delay
        else:
            steering_transform = transducer.ray_transforms[
                index - running_index_list[transducer_count]
            ]
            timedelay = 0

        if return_local:
            distance = np.linalg.norm(transducer_transform.translation)
            beam_transform = (
                geometry.Transform([0, 0, 0], [-distance, 0, 0]) * steering_transform
            )
        else:
            beam_transform = transducer_transform * steering_transform
            # print(f'beam_transform {beam_transform.rotation.as_euler("zyx")}')

        global_bounds = np.array(
            [[np.min(x), np.max(x)], [np.min(y), np.max(y)], [np.min(z), np.max(z)]]
        )
        xs, ys, zs = np.meshgrid(
            global_bounds[0], global_bounds[1], global_bounds[2], indexing="ij"
        )
        global_vertices = np.stack((xs.flatten(), ys.flatten(), zs.flatten()), axis=-1)
        local_vertices = beam_transform.apply_to_points(global_vertices, inverse=True)

        local_mins = np.min(local_vertices, axis=0)
        local_maxs = np.max(local_vertices, axis=0)
        local_x = np.arange(
            local_mins[0], local_maxs[0] + resolution, step=resolution, dtype=np.float32
        )
        local_y = np.arange(
            local_mins[1], local_maxs[1] + resolution, step=resolution, dtype=np.float32
        )
        if volumetric:
            local_z = np.arange(
                local_mins[2],
                local_maxs[2] + resolution,
                step=resolution,
                dtype=np.float32,
            )
        else:
            local_z = np.array([0], dtype=np.float32)
        xxx, yyy, zzz = np.meshgrid(local_x, local_y, local_z, indexing="ij")

        if isinstance(transducer, Focused):
            assert self.sensor.aperture_type == "extended_aperture", (
                "For focused transducers, the sensor aperture type must be 'extended_aperture'"
            )
            sensors_per_el = transducer.get_sensors_per_el()
            transmit_centroids = np.zeros((transducer.get_num_elements(), 3))
            pos = 0
            for entry in range(transducer.get_num_elements()):
                transmit_centroids[entry] = np.mean(
                    transducer.sensor_coords[pos : pos + sensors_per_el, :], axis=0
                )
                pos += sensors_per_el
            element_centroids = beam_transform.apply_to_points(
                element_centroids, inverse=True
            )

            transmit_dists = np.sqrt(xxx**2 + yyy**2 + zzz**2)
            denominator = transducer.get_num_elements()
        else:
            if self.sensor.aperture_type == "transmit_as_receive":
                # recreate the sensor mask
                transmit_centroids = np.linspace(
                    -transducer.not_transducer.transducer.element_pitch
                    * len(transducer.not_transducer.active_elements)
                    / 2,
                    transducer.not_transducer.transducer.element_pitch
                    * len(transducer.not_transducer.active_elements)
                    / 2,
                    len(transducer.not_transducer.active_elements),
                )
                # Account for PML?
                transmit_centroids = np.stack(
                    (
                        np.zeros_like(transmit_centroids),
                        transmit_centroids,
                        np.zeros_like(transmit_centroids),
                    ),
                    axis=1,
                )
                element_centroids = steering_transform.apply_to_points(
                    transmit_centroids, inverse=True
                )
                denominator = len(preprocessed_data)
            else:
                element_centroids = beam_transform.apply_to_points(
                    element_centroids, inverse=True
                )
                denominator = transducer.get_num_elements()

            distances = np.stack([xxx, yyy, zzz], axis=0)
            transmit_dists = np.abs(
                np.einsum("ijkl,i->jkl", distances, np.array((1, 0, 0)))
            )  # computes the dot product of the distances with the unit vector in the x direction

        local_image_matrix = np.zeros(
            (len(local_x), len(local_y), len(local_z)), dtype=np.float32
        )

        if pressure_field is not None:
            assert pressure_field_resolution is not None, (
                "Pressure field resolution must be provided if pressure field is provided"
            )
            vox_size = pressure_field_resolution
            normalized_pfield = pressure_field / np.sum(pressure_field)
            pfield_xs = np.arange(
                0, normalized_pfield.shape[0] * vox_size, step=vox_size
            )
            pfield_ys = np.arange(
                -normalized_pfield.shape[1] / 2 * vox_size + vox_size / 2,
                normalized_pfield.shape[1] / 2 * vox_size + vox_size / 2,
                step=vox_size,
            )
            f = interpolate.interp2d(
                pfield_ys, pfield_xs, normalized_pfield, kind="linear", fill_value=0
            )
            apodizations = np.repeat(
                f(
                    local_y,
                    local_x,
                ),
                len(local_z),
            ).reshape(len(local_x), len(local_y), len(local_z))
        else:
            apodizations = np.ones((len(local_x), len(local_y), len(local_z)))

        el2el_dists = np.linalg.norm(element_centroids, axis=1) + transducer.width / 2

        lx, ly, lz = np.meshgrid(local_x, local_y, local_z, indexing="ij")
        apodizations_precompute = apodizations[
            : len(local_x), : len(local_y), : len(local_z)
        ].astype(np.float32)
        denominator = np.array(denominator, dtype=np.float32)

        for i, (centroid, rf_series) in enumerate(
            zip(element_centroids, preprocessed_data)
        ):
            element_dists = np.sqrt(
                (lx - centroid[0]) ** 2
                + (ly - centroid[1]) ** 2
                + (lz - centroid[2]) ** 2
            )

            if self.sensor.aperture_type == "transmit_as_receive":
                travel_times = np.round(
                    (transmit_dists + element_dists) / c0 / dt
                ).astype(np.int16)
                windowed_times = travel_times
            else:
                travel_times = np.round(
                    (transmit_dists + element_dists + timedelay) / c0 / dt
                ).astype(np.int16)
                windowed_times = np.where(
                    transmit_dists + element_dists + timedelay < el2el_dists[i] * 1.5,
                    0,
                    travel_times,
                )  # Added a factor of 1.2 to windowing condition
            
            # Clip windowed_times to valid indices
            # Values beyond rf_series length are masked out below
            windowed_times_clipped = np.clip(windowed_times, 0, len(rf_series) - 1)
            
            # Create a mask for valid indices
            valid_mask = (windowed_times >= 0) & (windowed_times < len(rf_series))
            
            # Get RF values and mask to zero out invalid ones
            rf_values = rf_series[windowed_times_clipped]
            rf_values = np.where(valid_mask, rf_values, 0.0)

            local_image_matrix += (
                rf_values * apodizations_precompute / denominator
            )

        local_image_matrix = np.abs(hilbert(local_image_matrix, axis=0)).astype(
            np.float32
        )

        flat = local_image_matrix.flatten()
        local_coords = np.stack((xxx.flatten(), yyy.flatten(), zzz.flatten()), axis=-1)

        # # RegularGridInterpolator (faster)
        # interpolator = RegularGridInterpolator((local_x, local_y, local_z), local_image_matrix, method='nearest', bounds_error=False, fill_value=None)
        # if not volumetric:
        #     z = np.array([0])
        # gx,gy,gz = np.meshgrid(x, y, z, indexing='ij')
        # global_points = np.stack((gx, gy, gz), axis=-1).reshape(-1,3)
        # global_points = beam_transform.apply_to_points(global_points, inverse=True).astype(np.float32)
        # global_signal = interpolator(global_points).reshape(len(x), len(y), len(z))

        # NearestNDInterpolator (slower)
        local_2_global = beam_transform.apply_to_points(local_coords)
        interpolator = NearestNDInterpolator(local_2_global, flat)
        if not volumetric:
            z = np.array([0])
        gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")
        global_signal = interpolator(gx, gy, gz).reshape(len(x), len(y), len(z))

        if save_intermediates:
            utils.save_array(
                global_signal,
                f"{self.simulation_path}/reconstruct/intermediate_image_{str(index).zfill(6)}.npy",
            )
            return 1
        return global_signal

    def selective_compound(
        self,
        transducers,
        workers=8,
        resolution_multiplier=1,
        local=False,
        pressure_field=None,
        combine=True,
    ):
        if isinstance(self.transducer_set[0], Focused):
            # do nothing
            pass
        else:
            # still do nothing
            pass

        matrix_dims = self.phantom.matrix_dims
        voxel_dims = self.phantom.voxel_dims
        c0 = self.phantom.baseline[0]
        # dt = self.kgrid.dt # not the correct way to access the kgrid
        # ------------------------------------------------------------------------------------------------------------
        # TODO: fix this
        dt = (self.results[0][0][-1] - self.results[0][0][0]) / self.results[0][
            0
        ].shape[0]
        # ------------------------------------------------------------------------------------------------------------

        resolution = (
            max(dt * c0, 2 * c0 / self.transducer_set.get_lowest_frequency())
            / 4
            / resolution_multiplier
        )

        x = np.arange(
            -matrix_dims[0] * voxel_dims[0] / 2 + voxel_dims[0] / 2,
            matrix_dims[0] * voxel_dims[0] / 2 + voxel_dims[0] / 2,
            step=resolution,
        )
        y = np.arange(
            -matrix_dims[1] * voxel_dims[1] / 2 + voxel_dims[1] / 2,
            matrix_dims[1] * voxel_dims[1] / 2 + voxel_dims[1] / 2,
            step=resolution,
        )
        z = np.arange(
            -matrix_dims[2] * voxel_dims[2] / 2 + voxel_dims[2] / 2,
            matrix_dims[2] * voxel_dims[2] / 2 + voxel_dims[2] / 2,
            step=resolution,
        )

        image_matrix = np.zeros((len(x), len(y), len(z)))

        # Note: origin is at center of the 3d image in global coordinate system

        element_centroids = self.__get_element_centroids()

        arguments = []

        transducer_count = 0
        transducer, transducer_transform = self.transducer_set[transducer_count]
        running_index_list = np.cumsum(
            [
                transducer.get_num_rays()
                for transducer in self.transducer_set.transducers
            ]
        )

        for index in tqdm.tqdm(range(len(self.results))):
            if index > running_index_list[transducer_count] - 1:
                transducer_count += 1
                transducer, transducer_transform = self.transducer_set[transducer_count]

            if transducer_count not in transducers:
                continue
            arguments.append(
                (
                    index,
                    running_index_list,
                    transducer_count,
                    transducer,
                    transducer_transform,
                    x,
                    y,
                    z,
                    c0,
                    dt,
                    element_centroids,
                    resolution,
                    pressure_field,
                )
            )

        print(f"running reconstruction on {len(arguments)} rays")
        if workers > 1:
            with multiprocessing.Pool(workers) as p:
                if not local:
                    image_matrices = list(
                        p.starmap(self.scanline_reconstruction, arguments)
                    )
                else:
                    p.starmap(self.scanline_reconstruction, arguments)
        else:
            if not local:
                for argument in arguments:
                    image_matrices.append(self.scanline_reconstruction(*argument))
            else:
                for argument in arguments:
                    self.scanline_reconstruction(*argument)

        if combine:
            return np.sum(np.stack(image_matrices, axis=0), axis=0)
        else:
            return image_matrices


def compute_local_image_matrix_matrix(
    local_image_matrix,
    lx,
    ly,
    lz,
    element_centroids,
    preprocessed_data,
    transmit_dists,
    c0,
    dt,
    apodizations,
    denominator,
    el2el_dists,
    timedelay,
):
    nx, ny, nz = lx.shape
    flat_apodizations = apodizations.flatten()
    local_image_matrix_flat = local_image_matrix.flatten()
    for i in range(element_centroids.shape[0]):
        centroid = element_centroids[i]
        rf_series = preprocessed_data[i].flatten()

        # Compute element distances
        element_dists = np.sqrt(
            (lx - centroid[0]) ** 2 + (ly - centroid[1]) ** 2 + (lz - centroid[2]) ** 2
        )

        # Compute travel times
        travel_times = np.round(
            (transmit_dists + element_dists + timedelay) / c0 / dt
        ).astype(np.int32)

        # Window to reduce intra-transducer interference
        # windowed_times = np.where(transmit_dists + element_dists + timedelay < el2el_dists[i], 0, travel_times)
        windowed_times = np.clip(travel_times, 0, rf_series.shape[0] - 1).flatten()

        # Update local_image_matrix
        value = rf_series[windowed_times] * flat_apodizations / denominator
        local_image_matrix_flat += value

    local_image_matrix += local_image_matrix_flat.reshape(nx, ny, nz)
