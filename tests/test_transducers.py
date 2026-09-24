import copy

import numpy as np
import pytest

from musik.transducer import Focused, Planewave, Transducer
from musik.transducer_set import TransducerSet
from musik.utils.geometry import Transform


def test_transducer_honors_source_strength_and_active_elements():
    transducer = Transducer(
        source_strength=123,
        elements=4,
        active_elements=np.array([0, 2]),
    )

    assert transducer.source_strength == 123
    np.testing.assert_array_equal(transducer.active_elements, [0, 2])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"imaging_ndims": 1}, "either 2D or 3D"),
        ({"ray_num": 0}, "should not contain entries <= 0"),
        (
            {"imaging_ndims": 3, "sweep": [1], "ray_num": [2]},
            "Dimensions do not match",
        ),
    ],
)
def test_transducer_validates_ray_configuration(kwargs, message):
    with pytest.raises(Exception, match=message):
        Transducer(**kwargs)


def test_focused_ray_transforms_cover_requested_2d_sweep():
    transducer = Focused(sweep=np.pi / 2, ray_num=3)
    headings = np.array(
        [ray.apply_to_point([1, 0, 0]) for ray in transducer.ray_transforms]
    )

    assert transducer.get_num_rays() == 3
    np.testing.assert_allclose(
        np.arctan2(headings[:, 1], headings[:, 0]),
        [-np.pi / 4, 0, np.pi / 4],
        atol=1e-6,
    )


def test_focused_3d_ray_count_is_product_of_dimensions():
    transducer = Focused(
        imaging_ndims=3,
        sweep=np.array([0.2, 0.4]),
        ray_num=np.array([2, 3]),
    )

    assert transducer.get_num_rays() == 6


def test_planewave_honors_transmit_flag_and_steering_angles():
    transducer = Planewave(
        sweep=0.4,
        ray_num=3,
        transmit=False,
        steering_angles=[-0.1, 0, 0.1],
    )

    assert transducer.transmit is False
    np.testing.assert_allclose(transducer.steering_angles, [-0.1, 0, 0.1])
    assert all(np.allclose(ray.get(), np.eye(4)) for ray in transducer.ray_transforms)


def test_make_sensor_coords_for_centroids():
    transducer = Focused(elements=4, width=0.008, sensor_sampling_scheme="centroid")

    transducer.make_sensor_coords(c0=1500)

    assert transducer.sensors_per_el == 1
    assert transducer.sensor_coords.shape == (4, 3)
    np.testing.assert_allclose(
        transducer.sensor_coords[:, 1], [-0.003, -0.001, 0.001, 0.003]
    )
    np.testing.assert_allclose(transducer.sensor_coords[:, [0, 2]], 0)


def test_make_sensor_coords_samples_element_height_by_wavelength():
    transducer = Focused(
        elements=2,
        height=0.003,
        max_frequency=1e6,
        sensor_sampling_scheme="grid",
    )

    transducer.make_sensor_coords(c0=1500)

    assert transducer.sensors_per_el == 4
    assert transducer.sensor_coords.shape == (8, 3)
    np.testing.assert_allclose(
        transducer.sensor_coords[:4, 2], np.linspace(-0.0015, 0.0015, 4)
    )


def test_transducer_save_load_round_trip_does_not_mutate_input():
    transducer = Focused(label="probe", elements=3, ray_num=2)
    transducer.make_sensor_coords(1500)
    transducer.pulse = np.array([0.0, 1.0, 0.0])
    saved = transducer.save()
    original = copy.deepcopy(saved)

    loaded = Focused.load(saved)

    assert saved.keys() == original.keys()
    assert "ray_transforms" in saved
    np.testing.assert_array_equal(saved["sensor_coords"], original["sensor_coords"])
    np.testing.assert_array_equal(saved["pulse"], original["pulse"])
    assert loaded.label == "probe"
    np.testing.assert_array_equal(loaded.sensor_coords, transducer.sensor_coords)
    np.testing.assert_array_equal(loaded.pulse, transducer.pulse)
    assert loaded.get_num_rays() == 2


def test_window_and_envelope_detection():
    transducer = Transducer()
    transducer.pulse = np.ones(2)
    scan_lines = np.ones((2, 12))

    windowed = transducer.window(scan_lines.copy(), window_factor=2)
    envelope = transducer.envelope_detection(np.cos(np.linspace(0, 4 * np.pi, 128)))

    assert np.all(windowed[:, :4] == 0)
    assert np.all(windowed[:, 4:] == 1)
    assert envelope.shape == (128,)
    assert np.all(envelope >= 0)


def test_transducer_set_defaults_are_isolated_and_rng_always_exists():
    first = TransducerSet()
    second = TransducerSet()

    first.add_transducer(Focused(ray_num=1))

    assert len(first) == 1
    assert len(second) == 0
    assert first.rng is not second.rng

    first.generate_extrinsics()
    assert isinstance(first.poses[0], Transform)


def test_transducer_set_add_remove_keeps_counts_and_poses_synchronized():
    transmitting = Focused(label="tx", ray_num=1, transmit=True)
    receiving = Focused(label="rx", ray_num=1, transmit=False)
    transducers = TransducerSet()

    transducers.add_transducer(transmitting, pose=Transform())
    transducers.add_transducer(receiving, pose=Transform(translation=(1, 0, 0)))

    assert len(transducers) == len(transducers.poses) == 2
    assert transducers.n_transmit == 1
    assert transducers.find_transducer("rx") == 1

    transducers.remove_transducer(label="tx")

    assert len(transducers) == len(transducers.poses) == 1
    assert transducers.n_transmit == 0


def test_assign_pose_rejects_index_at_length():
    transducers = TransducerSet([Focused(ray_num=1)])

    with pytest.raises(AssertionError, match="Index out of range"):
        transducers.assign_pose(1, Transform())


def test_transducer_set_generates_reproducible_spherical_poses():
    first = TransducerSet([Focused(ray_num=1), Focused(ray_num=1)], seed=22)
    second = TransducerSet([Focused(ray_num=1), Focused(ray_num=1)], seed=22)

    first.generate_extrinsics("spherical", {"r_mean": 0.04})
    second.generate_extrinsics("spherical", {"r_mean": 0.04})

    for left, right in zip(first.poses, second.poses):
        np.testing.assert_allclose(left.get(), right.get())


def test_transducer_set_save_load_round_trip(tmp_path):
    transducer = Focused(label="probe", elements=2, ray_num=2)
    transducer.make_sensor_coords(1500)
    transducer.pulse = np.array([0.0, 1.0])
    original = TransducerSet(
        [transducer],
        [Transform(rotation=(0.1, 0.2, 0.3), translation=(1, 2, 3))],
        seed=9,
    )
    path = tmp_path / "transducers.json"

    original.save(path)
    loaded = TransducerSet.load(path, c0=1500)

    assert len(loaded) == 1
    assert loaded.seed == 9
    assert loaded[0][0].label == "probe"
    np.testing.assert_allclose(loaded[0][1].get(), original[0][1].get(), atol=1e-6)

