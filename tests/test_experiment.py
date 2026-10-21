import os
import time
from types import SimpleNamespace

import numpy as np
import pytest

from musik.experiment import Experiment, Results
from musik.phantom import Phantom
from musik.sensor import Sensor
from musik.simulation import SimProperties
from musik.transducer import Focused
from musik.transducer_set import TransducerSet
from musik.utils import utils
from musik.utils.geometry import Transform


def test_results_discovers_indices_and_loads_signals(tmp_path):
    expected = np.array([[0, 1, 2], [10, 11, 12], [20, 21, 22]])
    utils.save_array(expected, tmp_path / "signal_000007", compression=True)

    results = Results(str(tmp_path))

    assert len(results) == 1
    assert results.indices() == [7]
    assert results.result_shape == (3, 3)
    time, signals = results[7]
    np.testing.assert_array_equal(time, expected[0])
    np.testing.assert_array_equal(signals, expected[1:])


def test_results_loads_additional_signal_data(tmp_path):
    utils.save_array(np.array([[0, 1], [2, 3]]), tmp_path / "signal_000002")
    utils.save_array(np.array([[8, 9]]), tmp_path / "key_signal_000002")

    _, _, other = Results(str(tmp_path))[2]

    np.testing.assert_array_equal(other, [[8, 9]])


def test_empty_results_and_missing_index(tmp_path):
    results = Results(str(tmp_path))

    assert len(results) == 0
    assert results.result_shape is None
    with pytest.raises(ValueError):
        results[3]


def make_lightweight_experiment(tmp_path, completed_indices=()):
    results_path = tmp_path / "results"
    results_path.mkdir(exist_ok=True)
    for index in completed_indices:
        utils.save_array(np.ones((2, 2)), results_path / f"signal_{index:06d}")
    transducers = [
        SimpleNamespace(transmit=True, get_num_rays=lambda: 2),
        SimpleNamespace(transmit=False, get_num_rays=lambda: 5),
    ]
    transducer_set = SimpleNamespace(
        transmit_transducers=lambda: [item for item in transducers if item.transmit]
    )
    return Experiment(
        simulation_path=str(tmp_path),
        transducer_set=transducer_set,
        nodes=2,
        additional_keys=["p_max", "invalid", "p_max"],
    )


def test_experiment_length_pending_indices_and_allowed_keys(tmp_path, capsys):
    experiment = make_lightweight_experiment(tmp_path, completed_indices=(0,))

    assert len(experiment) == 2
    assert experiment.indices == [1]
    assert experiment.indices_to_run(repeat=True) == [0, 1]
    assert experiment.additional_keys == ["p_max"]
    assert "not a valid flag" in capsys.readouterr().out


def test_experiment_subdivide_splits_pending_work(tmp_path):
    experiment = make_lightweight_experiment(tmp_path)

    subdivisions = experiment.subdivide()

    assert [part.tolist() for part in subdivisions] == [[0], [1]]
    assert experiment.subdivide(indices=[]) is None


def test_get_sensor_mask_uses_global_coordinates_and_ignores_out_of_bounds(tmp_path):
    experiment = make_lightweight_experiment(tmp_path)
    experiment.phantom = SimpleNamespace(
        mask=np.zeros((5, 5, 5)), voxel_dims=np.ones(3)
    )
    experiment.sensor = SimpleNamespace(
        sensor_coords=np.array([[0, 0, 0], [1, 0, 0], [100, 0, 0]])
    )

    mask = experiment.get_sensor_mask()

    assert mask[2, 2, 2] == 1
    assert mask[3, 2, 2] == 1
    assert np.count_nonzero(mask) == 2


def test_full_experiment_configuration_round_trip(tmp_path):
    path = tmp_path / "experiment"
    phantom = Phantom(matrix_dims=(3, 3, 3), voxel_dims=(0.001,) * 3)
    transducer = Focused(label="probe", elements=2, ray_num=2)
    transducer.make_sensor_coords(phantom.baseline[0])
    transducer.pulse = np.array([0.0, 1.0])
    transducers = TransducerSet([transducer], [Transform()], seed=3)
    sensor = Sensor(aperture_type="extended_aperture", transducer_set=transducers)
    props = SimProperties(
        grid_size=(0.006, 0.006, 0.006),
        voxel_size=(0.001, 0.001, 0.001),
        PML_size=(1, 1, 1),
    )
    original = Experiment(
        simulation_path=str(path),
        sim_properties=props,
        phantom=phantom,
        transducer_set=transducers,
        sensor=sensor,
        nodes=1,
        gpu=False,
        workers=1,
        sim_workers=3,
        gpus=[0, 1],
        num_threads=2,
        additional_keys=["p_max"],
    )

    original.save()
    loaded = Experiment.load(path)

    assert len(loaded) == 2
    assert loaded.gpu is False
    assert (loaded.prep_workers, loaded.sim_workers, loaded.gpus, loaded.num_threads) == (1, 3, [0, 1], 2)
    assert loaded.additional_keys == ["p_max"]
    np.testing.assert_array_equal(loaded.phantom.mask, phantom.mask)
    np.testing.assert_allclose(loaded.transducer_set.poses[0].get(), np.eye(4))
    np.testing.assert_allclose(loaded.sensor.sensor_coords, sensor.sensor_coords)



FAILING_INDICES = set()


class FakeSimulation:
    """Stands in for a k-Wave simulation: records which process ran each index."""

    def __init__(self, *args, simulation_path, index, **kwargs):
        self.simulation_path, self.index = simulation_path, index

    def prep(self):
        pass

    def run(self):
        if self.index in FAILING_INDICES:
            raise RuntimeError(f"simulated failure at index {self.index}")
        time.sleep(0.2)
        device = int(os.environ.get("CUDA_VISIBLE_DEVICES", "-1"))
        np.save(os.path.join(self.simulation_path, "results", f"signal_{self.index:06d}.npy"), np.array([[os.getpid(), device]]))


def make_pipeline_experiment(tmp_path, monkeypatch, rays, failing=(), **kwargs):
    import musik.experiment

    monkeypatch.setattr(musik.experiment, "Simulation", FakeSimulation)
    FAILING_INDICES.clear()
    FAILING_INDICES.update(failing)
    transducer = SimpleNamespace(transmit=True, get_num_rays=lambda: rays)
    return Experiment(
        simulation_path=str(tmp_path),
        transducer_set=SimpleNamespace(transmit_transducers=lambda: [transducer]),
        nodes=1,
        **kwargs,
    )


def test_sim_workers_run_simulations_concurrently(tmp_path, monkeypatch):
    experiment = make_pipeline_experiment(tmp_path, monkeypatch, rays=12, prep_workers=1, sim_workers=3)

    experiment.run()

    results = Results(str(tmp_path / "results"))
    assert results.indices() == list(range(12))
    pids = {int(np.load(path)[0, 0]) for path in results.result_paths}
    assert len(pids) > 1


def test_failed_sim_worker_is_reported_and_others_finish(tmp_path, monkeypatch):
    experiment = make_pipeline_experiment(tmp_path, monkeypatch, rays=8, failing={5}, prep_workers=1, sim_workers=2)

    with pytest.raises(RuntimeError, match="1 of 8 simulations have no results"):
        experiment.run()

    assert sorted(Results(str(tmp_path / "results")).indices()) == [0, 1, 2, 3, 4, 6, 7]


def test_run_does_not_hang_when_every_sim_worker_fails(tmp_path, monkeypatch):
    experiment = make_pipeline_experiment(tmp_path, monkeypatch, rays=6, failing=set(range(6)), prep_workers=1, sim_workers=1)

    start = time.time()
    with pytest.raises(RuntimeError, match="6 of 6 simulations have no results"):
        experiment.run()
    assert time.time() - start < 30


def result_devices(tmp_path):
    results = Results(str(tmp_path / "results"))
    return {int(np.load(path)[0, 1]) for path in results.result_paths}, len(results)


def test_gpus_pin_sim_workers_to_each_device(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    experiment = make_pipeline_experiment(tmp_path, monkeypatch, rays=12, prep_workers=2, sim_workers=2, gpus=[0, 1])

    experiment.run()

    assert result_devices(tmp_path) == ({0, 1}, 12)


def test_gpus_index_into_already_visible_devices(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    experiment = make_pipeline_experiment(tmp_path, monkeypatch, rays=8, sim_workers=1, gpus=2)

    experiment.run()

    assert result_devices(tmp_path) == ({3, 5}, 8)
    with pytest.raises(ValueError, match="only 2 are visible"):
        experiment.run(gpus=3, repeat=True)


def test_workers_is_an_alias_for_prep_workers(tmp_path):
    assert Experiment(simulation_path=str(tmp_path / "new"), workers=3).prep_workers == 3

    path = tmp_path / "old"
    phantom = Phantom(matrix_dims=(3, 3, 3), voxel_dims=(0.001,) * 3)
    transducer = Focused(label="probe", elements=2, ray_num=2)
    transducer.make_sensor_coords(phantom.baseline[0])
    transducers = TransducerSet([transducer], [Transform()], seed=3)
    Experiment(
        simulation_path=str(path),
        sim_properties=SimProperties(grid_size=(0.006,) * 3, voxel_size=(0.001,) * 3, PML_size=(1, 1, 1)),
        phantom=phantom,
        transducer_set=transducers,
        sensor=Sensor(aperture_type="extended_aperture", transducer_set=transducers),
        nodes=1,
    ).save()
    # An experiment saved before prep_workers / sim_workers / gpus existed.
    utils.dict_to_json({"indices": None, "nodes": 1, "gpu": True, "workers": 4, "additional_keys": []}, str(path / "experiment.json"))

    loaded = Experiment.load(str(path))

    assert (loaded.prep_workers, loaded.sim_workers, loaded.gpus, loaded.num_threads) == (4, 1, None, 4)


def test_experiment_passes_num_threads_to_each_simulation(tmp_path, monkeypatch):
    import musik.experiment

    seen = []

    class Recorder:
        def __init__(self, *args, num_threads, **kwargs):
            seen.append(num_threads)

        def prep(self):
            pass

        def run(self):
            pass

    monkeypatch.setattr(musik.experiment, "Simulation", Recorder)
    experiment = make_lightweight_experiment(tmp_path)

    experiment.simulate(0)
    experiment.num_threads = 2
    experiment.simulate(1)

    assert seen == [4, 2]


def test_failed_run_with_large_queued_simulations_exits(tmp_path):
    """Real prepared simulations are megabytes each; if every simulation worker dies they
    fill the queue's pipe. The run must still raise and the process must still exit."""
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(f"""
        from types import SimpleNamespace
        import numpy as np
        import musik.experiment
        from musik.experiment import Experiment

        class Big:
            def __init__(self, *args, index, **kwargs):
                self.index, self.payload = index, None
            def prep(self):
                self.payload = np.ones(2_000_000)  # 16 MB, like a prepared medium
            def run(self):
                raise RuntimeError("simulated failure")

        musik.experiment.Simulation = Big
        transducer = SimpleNamespace(transmit=True, get_num_rays=lambda: 12)
        experiment = Experiment(
            simulation_path=r"{tmp_path}",
            transducer_set=SimpleNamespace(transmit_transducers=lambda: [transducer]),
            nodes=1, prep_workers=1, sim_workers=2,
        )
        try:
            experiment.run()
        except RuntimeError as error:
            print("raised:", error)
    """)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)

    assert "raised: 12 of 12 simulations have no results" in result.stdout
