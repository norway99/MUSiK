import numpy as np
import pytest

from musik.reconstruction import PreprocessedDataLoader, Reconstruction


def write_batch(path, number, start, end):
    values = np.arange(start, end)
    np.savez(
        path / f"preprocess_batch_{number:04d}.npz",
        start_index=start,
        end_index=end,
        times=values,
        coords=values + 10,
        processed=values + 20,
    )


def test_preprocessed_loader_reads_ranges_across_batches(tmp_path):
    write_batch(tmp_path, 0, 0, 3)
    write_batch(tmp_path, 1, 3, 6)
    loader = PreprocessedDataLoader(tmp_path, max_cached_batches=2)

    times, coords, processed = loader.get_range(1, 5)

    assert len(loader) == 6
    assert times == [1, 2, 3, 4]
    assert coords == [11, 12, 13, 14]
    assert processed == [21, 22, 23, 24]


def test_preprocessed_loader_uses_lru_cache_and_can_clear_it(tmp_path):
    write_batch(tmp_path, 0, 0, 2)
    write_batch(tmp_path, 1, 2, 4)
    loader = PreprocessedDataLoader(tmp_path, max_cached_batches=1)

    loader.get_range(0, 1)
    assert list(loader._cache) == [0]
    loader.get_range(2, 3)
    assert list(loader._cache) == [1]
    loader.clear_cache()

    assert loader._cache == {}
    assert loader._cache_order == []


def test_preprocessed_loader_rejects_missing_batches_and_bad_indices(tmp_path):
    with pytest.raises(FileNotFoundError):
        PreprocessedDataLoader(tmp_path)

    write_batch(tmp_path, 0, 0, 2)
    loader = PreprocessedDataLoader(tmp_path)
    with pytest.raises(IndexError, match="out of range"):
        loader.get_range(2, 3)


def test_reconstruction_requires_experiment():
    with pytest.raises(AssertionError, match="provide an experiment"):
        Reconstruction()

