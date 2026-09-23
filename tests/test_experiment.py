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
        additional_keys=["p_max"],
    )

    original.save()
    loaded = Experiment.load(path)

    assert len(loaded) == 2
    assert loaded.gpu is False
    assert loaded.additional_keys == ["p_max"]
    np.testing.assert_array_equal(loaded.phantom.mask, phantom.mask)
    np.testing.assert_allclose(loaded.transducer_set.poses[0].get(), np.eye(4))
    np.testing.assert_allclose(loaded.sensor.sensor_coords, sensor.sensor_coords)

