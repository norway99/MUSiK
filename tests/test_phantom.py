import numpy as np
import pytest

from musik.phantom import Phantom
from musik.tissue import Tissue


def test_tissue_save_and_load_ignores_unknown_fields():
    tissue = Tissue().load({"name": "bone", "c": 3000, "unknown": "ignored"})

    assert tissue.name == "bone"
    assert tissue.c == 3000
    assert "unknown" not in tissue.save()


def test_phantom_initializes_water_and_zero_mask():
    phantom = Phantom(voxel_dims=(1, 2, 3), matrix_dims=(3, 4, 5), seed=4)

    np.testing.assert_array_equal(phantom.voxel_dims, [1, 2, 3])
    assert phantom.mask.shape == (3, 4, 5)
    assert set(phantom.tissues) == {"water"}


def test_add_assign_and_remove_tissue():
    phantom = Phantom(matrix_dims=(3, 3, 3))
    tissue = Tissue(name="bone", label=2, c=2500, rho=1800)
    region = np.zeros((3, 3, 3), dtype=bool)
    region[1, 1, 1] = True

    phantom.add_tissue(tissue, mask=region)
    assert phantom[1, 1, 1] == 2
    phantom.remove_tissue("bone")

    assert phantom[1, 1, 1] == 0
    assert "bone" not in phantom.tissues


def test_mask_assignment_rejects_unknown_tissue_label():
    phantom = Phantom(matrix_dims=(2, 2, 2))

    with pytest.raises(AssertionError, match="valid tissue label"):
        phantom[0] = 99


def test_set_default_tissue_updates_unassigned_voxels():
    phantom = Phantom(matrix_dims=(2, 2, 2))
    tissue = Tissue(name="gel", label=3)

    phantom.set_default_tissue(tissue)

    assert phantom.default_tissue == 3
    assert np.all(phantom.mask == 3)


def test_add_tissue_sphere_updates_mask_and_invalidates_complete():
    phantom = Phantom(matrix_dims=(5, 5, 5), voxel_dims=(1, 1, 1))
    tissue = Tissue(name="target", label=4)
    phantom.complete = np.zeros((2, 5, 5, 5))

    phantom.add_tissue_sphere((0, 0, 0), 1, tissue)

    assert np.count_nonzero(phantom.mask == 4) == 7
    assert phantom.complete is None


def test_generate_constant_tissue_has_sound_speed_and_density_channels():
    phantom = Phantom(matrix_dims=(2, 2, 2))
    tissue = Tissue(name="constant", c=1600, rho=1050, sigma=0)

    generated = phantom.generate_tissue(tissue, (2, 3, 4), (1, 1, 1))

    assert generated.shape == (2, 2, 3, 4)
    assert np.all(generated[0] == 1600)
    assert np.all(generated[1] == 1050)


def test_seed_makes_heterogeneous_tissue_reproducible():
    tissue = Tissue(name="speckle", c=1540, rho=1000, sigma=10, scale=1)
    first = Phantom(seed=12).generate_tissue(tissue, (4, 4, 4), (1, 1, 1))
    second = Phantom(seed=12).generate_tissue(tissue, (4, 4, 4), (1, 1, 1))

    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(first[1], first[0] / 1540 * 1000)


def test_get_complete_materializes_each_tissue_and_caches_result():
    phantom = Phantom(matrix_dims=(2, 2, 2), voxel_dims=(1, 1, 1))
    phantom.add_tissue(Tissue(name="target", label=1, c=1600, rho=1100, sigma=0))
    phantom[0] = 1

    first = phantom.get_complete()
    second = phantom.get_complete()

    assert first.shape == (2, 2, 2, 2)
    assert np.all(first[0, 0] == 1600)
    assert np.all(first[1, 0] == 1100)
    assert second is first


def _shell_phantom():
    """Nested shells covering white-noise, correlated, and constant tissues."""
    shape = (40, 36, 32)
    grid = np.stack(np.meshgrid(*[np.arange(n) - n / 2 for n in shape], indexing="ij"))
    radius = np.sqrt((grid**2).sum(0))
    phantom = Phantom(matrix_dims=shape, voxel_dims=(1, 1, 1), seed=5)
    phantom.add_tissue(Tissue(name="white", label=1, c=1590, rho=1080, sigma=20, scale=0.25))
    phantom.add_tissue(Tissue(name="textured", label=2, c=1560, rho=1050, sigma=15, scale=3))
    phantom.add_tissue(Tissue(name="fluid", label=3, c=1500, rho=1000, sigma=0))
    mask = np.zeros(shape, dtype=np.float32)
    mask[radius < 18] = 1
    mask[radius < 12] = 2
    mask[radius < 4] = 3
    return phantom, mask


def _lag_correlation(field, selection, axis=0, lag=1):
    count = field.shape[axis] - lag
    a, b = np.take(field, range(count), axis=axis), np.take(field, range(lag, lag + count), axis=axis)
    both = np.take(selection, range(count), axis=axis) & np.take(selection, range(lag, lag + count), axis=axis)
    return np.corrcoef(a[both], b[both])[0, 1]


def test_make_complete_matches_texture_model():
    phantom, mask = _shell_phantom()

    complete = phantom.make_complete(mask, (1, 1, 1))

    assert complete.shape == (2, *mask.shape)
    assert complete.dtype == np.float32
    white, textured, fluid = (mask == 1), (mask == 2), (mask == 3)
    # Sub-voxel scatterers: white noise with std sigma * sqrt(scale / voxel).
    assert complete[0][white].mean() == pytest.approx(1590, abs=0.5)
    assert complete[0][white].std() == pytest.approx(20 * 0.25**0.5, rel=0.05)
    assert abs(_lag_correlation(complete[0], white)) < 0.05
    # Scatterers spanning 3 voxels: linearly upsampled coarse noise, correlated across neighbours.
    assert complete[0][textured].mean() == pytest.approx(1560, abs=1.5)
    assert _lag_correlation(complete[0], textured) > 0.7
    # Constant tissue and the density channel are exact.
    assert np.all(complete[0][fluid] == 1500) and np.all(complete[1][fluid] == 1000)
    np.testing.assert_allclose(complete[1][white], complete[0][white] / 1590 * 1080, rtol=1e-6)
    np.testing.assert_allclose(complete[1][textured], complete[0][textured] / 1560 * 1050, rtol=1e-6)


def test_make_complete_is_seeded_and_fills_unknown_labels_with_default():
    phantom, mask = _shell_phantom()
    mask[:3] = 9  # no tissue has label 9
    second, _ = _shell_phantom()

    complete = phantom.make_complete(mask, (1, 1, 1))

    np.testing.assert_array_equal(complete, second.make_complete(mask, (1, 1, 1)))
    # The default tissue is water (label 0, constant).
    assert np.all(complete[0][:3] == 1500) and np.all(complete[1][:3] == 1000)


def test_phantom_save_load_round_trip(tmp_path):
    phantom = Phantom(
        matrix_dims=(3, 3, 3), voxel_dims=(0.1, 0.2, 0.3), seed=8
    )
    phantom.add_tissue(Tissue(name="target", label=2, c=1700, rho=1200))
    phantom[1, 1, 1] = 2
    phantom.complete = phantom.get_complete()

    path = tmp_path / "phantom"
    phantom.save(path)
    loaded = Phantom.from_source(path)

    np.testing.assert_array_equal(loaded.mask, phantom.mask)
    np.testing.assert_array_equal(loaded.complete, phantom.complete)
    np.testing.assert_allclose(loaded.voxel_dims, phantom.voxel_dims)
    assert loaded.tissues["target"].save() == phantom.tissues["target"].save()

