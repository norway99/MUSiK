import numpy as np
import pytest

from musik.utils.geometry import (
    Transform,
    create_sphere,
    generate_pose_cylindrical,
    generate_pose_spherical,
    generate_random,
)


def test_transform_applies_translation_and_scale():
    transform = Transform(translation=(2, 4, 6))

    np.testing.assert_allclose(transform.apply_to_point((1, 1, 1), scale=2), [2, 3, 4])


def test_transform_inverse_round_trip_for_points():
    transform = Transform(rotation=(0.3, -0.2, 0.1), translation=(1, 2, 3))
    points = np.array([[0.0, 0.0, 0.0], [1.0, -2.0, 4.0]])

    transformed = transform.apply_to_points(points)

    np.testing.assert_allclose(
        transform.apply_to_points(transformed, inverse=True), points, atol=1e-6
    )


def test_transform_composition_matches_sequential_application():
    outer = Transform(rotation=(np.pi / 2, 0, 0), translation=(3, 0, 0))
    inner = Transform(rotation=(0, np.pi / 4, 0), translation=(0, 2, 0))
    point = np.array([1.0, 0.0, 0.0])

    expected = outer.apply_to_point(inner.apply_to_point(point))

    np.testing.assert_allclose(
        (outer * inner).apply_to_point(point), expected, atol=1e-6
    )


def test_transform_serialization_round_trip():
    transform = Transform(rotation=(0.1, 0.2, -0.3), translation=(4, 5, 6))

    loaded = Transform.load(transform.save())

    np.testing.assert_allclose(loaded.get(), transform.get(), atol=1e-6)


def test_transform_from_matrix_requires_ndarray():
    with pytest.raises(Exception, match="matrix format"):
        Transform(rotation=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], from_matrix=True)


def test_pad_to_cube_preserves_channels_and_fills_edges():
    array = np.stack([np.ones((2, 3, 4)), np.ones((2, 3, 4)) * 2])

    padded = Transform().padtocube(array)

    assert padded.shape == (2, 6, 6, 6)
    assert set(np.unique(padded[0])) == {1}
    assert set(np.unique(padded[1])) == {2}


def test_create_sphere_is_centered_in_global_coordinates():
    sphere = create_sphere(
        centroid=(0, 0, 0),
        radius=1,
        voxel_dims=(0.5, 0.5, 0.5),
        matrix_dims=(5, 5, 5),
    )

    assert sphere.shape == (5, 5, 5)
    assert sphere[2, 2, 2] != 0
    assert np.count_nonzero(sphere) == 7


def test_spherical_pose_is_reproducible_and_points_inward():
    first = generate_pose_spherical(r_mean=0.05, rng=np.random.default_rng(7))
    second = generate_pose_spherical(r_mean=0.05, rng=np.random.default_rng(7))

    np.testing.assert_allclose(first[0], second[0])
    np.testing.assert_allclose(first[1], second[1])
    assert np.linalg.norm(first[1]) == pytest.approx(0.05)
    heading = Transform(first[0]).apply_to_point([1, 0, 0])
    assert np.dot(heading, first[1]) < 0


def test_cylindrical_pose_respects_radius_and_height_range():
    orientation, position = generate_pose_cylindrical(
        r_mean=0.03, z_range=0.01, rng=np.random.default_rng(3)
    )

    assert orientation.shape == (3,)
    assert np.linalg.norm(position[:2]) == pytest.approx(0.03)
    assert abs(position[2]) <= 0.01


def test_generate_random_respects_bounds_and_seed():
    rotation, translation = generate_random(size=0.2, rng=np.random.default_rng(11))

    assert np.all((0 <= rotation) & (rotation <= 2 * np.pi))
    assert np.all((-0.2 <= translation) & (translation <= 0.2))

