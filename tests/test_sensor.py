from types import SimpleNamespace

import numpy as np
import pytest

from musik.sensor import Sensor
from musik.transducer import Focused
from musik.transducer_set import TransducerSet
from musik.utils.geometry import Transform


def make_transducer_set():
    first = Focused(label="first", elements=2, ray_num=1)
    second = Focused(label="second", elements=2, ray_num=1)
    first.make_sensor_coords(1500)
    second.make_sensor_coords(1500)
    return TransducerSet(
        [first, second],
        [Transform(), Transform(translation=(1, 0, 0))],
    )


def test_pressure_field_sensor_has_no_point_coordinates():
    sensor = Sensor(aperture_type="pressure_field")

    assert sensor.sensor_coords is None
    assert sensor.element_lookup.size == 0
    assert sensor.sensors_per_el.size == 0


def test_microphone_requires_coordinates():
    with pytest.raises(Exception, match="sensor coordinates"):
        Sensor(aperture_type="microphone")


def test_extended_aperture_requires_transducer_set_and_poses():
    with pytest.raises(Exception, match="transducer set"):
        Sensor(aperture_type="extended_aperture")

    transducers = TransducerSet([Focused(elements=1, ray_num=1)])
    transducers.transducers[0].make_sensor_coords(1500)
    with pytest.raises(AssertionError, match="has not been assigned"):
        Sensor(aperture_type="extended_aperture", transducer_set=transducers)


def test_extended_aperture_collects_global_coordinates_and_element_lookup():
    transducers = make_transducer_set()

    sensor = Sensor(aperture_type="extended_aperture", transducer_set=transducers)

    assert sensor.sensor_coords.shape == (4, 3)
    np.testing.assert_array_equal(sensor.element_lookup, [0, 1, 2, 3])
    np.testing.assert_array_equal(sensor.sensors_per_el, [1, 1, 1, 1])
    np.testing.assert_allclose(sensor.sensor_coords[2:, 0], 1)


def test_sensor_save_load_round_trip(tmp_path):
    sensor = Sensor(
        aperture_type="extended_aperture", transducer_set=make_transducer_set()
    )
    path = tmp_path / "sensor.json"

    sensor.save(path)
    loaded = Sensor.load(path, transducer_set=sensor.transducer_set)

    assert loaded.aperture_type == sensor.aperture_type
    np.testing.assert_array_equal(loaded.sensor_coords, sensor.sensor_coords)
    np.testing.assert_array_equal(loaded.element_lookup, sensor.element_lookup)


def test_make_sensor_mask_for_transmit_as_receive_and_pressure_field():
    indexed = np.array([[[0, 2], [3, 0]], [[0, 0], [1, 0]]])
    not_transducer = SimpleNamespace(indexed_mask=indexed)

    transmit_sensor = Sensor(aperture_type="pressure_field")
    transmit_sensor.aperture_type = "transmit_as_receive"
    transmit_mask, coords = transmit_sensor.make_sensor_mask(None, not_transducer, 1)
    field_mask, field_coords = Sensor(aperture_type="pressure_field").make_sensor_mask(
        None, not_transducer, 1
    )

    np.testing.assert_array_equal(transmit_mask, indexed > 0)
    assert coords is None
    assert field_coords is None
    assert np.all(field_mask[:, :, 1] == 1)
    assert np.count_nonzero(field_mask) == 4


def test_hash_fn_returns_data_or_zeros_for_out_of_bounds_coordinate():
    sensor = Sensor(aperture_type="pressure_field")
    grid = np.array([2, 2, 2])
    hashes = np.array([0, 1, 2])
    data = np.array([[10, 11], [20, 21], [30, 31]])

    np.testing.assert_array_equal(
        sensor.hash_fn(np.array([1, 0, 0]), hashes, data, grid), [20, 21]
    )
    np.testing.assert_array_equal(
        sensor.hash_fn(np.array([-1, 0, 0]), hashes, data, grid), [0, 0]
    )


def test_voxel_to_element_averages_sensor_points_per_element():
    sensor = Sensor(
        aperture_type="microphone", sensor_coords=np.array([[0, 0, 0], [1, 0, 0]])
    )
    sensor.aperture_type = "extended_aperture"
    sensor.sensors_per_el = np.array([2])
    props = SimpleNamespace(
        matrix_size=np.array([2, 2, 2]), PML_size=np.zeros(3, dtype=int)
    )
    coords = np.array([[0, 0, 0], [1, 0, 0]])
    sensor_data = {"p": np.array([[1, 3], [5, 7]])}

    signals, other = sensor.voxel_to_element(props, None, coords, sensor_data, [])

    np.testing.assert_array_equal(signals, [[2, 6]])
    assert other == []


def test_sort_pressure_field_restores_grid_shape_and_additional_fields():
    sensor = Sensor(aperture_type="pressure_field")
    sensor_data = {
        "p": np.arange(12).reshape(3, 4),
        "p_max": np.arange(4).reshape(1, 4),
    }

    signals, other = sensor.sort_pressure_field(
        sensor_data, ["p_max"], grid_shape=(2, 2), PML_size=(0, 0)
    )

    assert signals.shape == (2, 2, 3)
    assert other[0].shape == (2, 2)

