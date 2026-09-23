from pathlib import Path

import numpy as np
import pytest

from musik.simulation import SimProperties, _make_medium, tempdir


def test_make_medium_casts_maps_and_uses_configured_properties():
    sim_phantom = np.array(
        [
            [[[1500, 2400]]],
            [[[1000, 1500]]],
        ],
        dtype=np.float16,
    )
    props = SimProperties(alpha_coeff=0.5, alpha_power=1.5, bona=4)

    medium = _make_medium(sim_phantom, props)

    assert medium.sound_speed.dtype == np.float32
    assert medium.density.dtype == np.float32
    assert medium.alpha_coeff == 0.5
    assert medium.alpha_power == 1.5
    assert medium.BonA == 4
    assert np.all(np.isfinite(medium.sound_speed**medium.alpha_power))


def test_sim_properties_computes_grid_metadata():
    props = SimProperties(
        grid_size=(0.02, 0.01, 0.01),
        voxel_size=(0.001, 0.001, 0.001),
        PML_size=(2, 2, 2),
    )

    assert props.matrix_size.shape == (3,)
    assert np.all(props.matrix_size >= [24, 14, 14])
    assert props.bounds.shape == (8, 3)
    assert props.t_end == pytest.approx(0.02 / 1540 * 2.2)


@pytest.mark.parametrize("value, expected", [(2, 2), (12, 3), (49, 7), (97, 97)])
def test_largest_prime_factor(value, expected):
    assert SimProperties().largest_prime_factor(value) == expected


def test_calc_matrix_size_does_not_mutate_inputs():
    props = SimProperties()
    grid_size = np.array([0.01, 0.002, 0.002])
    original = grid_size.copy()

    props.calc_matrix_size(
        grid_size,
        voxel_size=(0.001, 0.001, 0.001),
        PML_size=(1, 1, 1),
        transducer_dims=(0.01, 0.008),
    )

    np.testing.assert_array_equal(grid_size, original)


def test_optimize_simulation_parameters_updates_resolution_and_bounds():
    props = SimProperties(grid_size=(0.02, 0.01, 0.01), PML_size=(2, 2, 2))

    props.optimize_simulation_parameters(frequency=1e6, sos=1500, grid_lambda=3)

    np.testing.assert_allclose(props.voxel_size, [0.00025] * 3)
    assert props.bounds.shape == (8, 3)


def test_sim_properties_save_load_preserves_array_invariants(tmp_path):
    props = SimProperties(
        grid_size=(0.02, 0.01, 0.01),
        voxel_size=(0.001, 0.001, 0.001),
        PML_size=(2, 2, 2),
    )
    path = tmp_path / "properties.json"
    props.save(path)

    loaded = SimProperties.load(path)

    for name in ("grid_size", "voxel_size", "PML_size", "matrix_size", "bounds"):
        assert isinstance(getattr(loaded, name), np.ndarray)
        np.testing.assert_allclose(getattr(loaded, name), getattr(props, name))
    assert loaded.t_end == props.t_end


def test_tempdir_removes_directory_after_context():
    with tempdir() as directory:
        path = Path(directory)
        assert path.is_dir()
        (path / "file.txt").write_text("temporary")

    assert not path.exists()

