import json

import numpy as np

from musik.utils import utils


def test_numpy_encoder_handles_scalars_and_arrays():
    encoded = json.loads(
        json.dumps(
            {
                "integer": np.int64(2),
                "float": np.float64(1.5),
                "array": np.array([1, 2]),
            },
            cls=utils.NpEncoder,
        )
    )

    assert encoded == {"integer": 2, "float": 1.5, "array": [1, 2]}


def test_json_round_trip_with_numpy_values(tmp_path):
    path = tmp_path / "values.json"
    utils.dict_to_json({"shape": np.array([2, 3]), "value": np.float32(4)}, path)

    assert utils.json_to_dict(path) == {"shape": [2, 3], "value": 4.0}


def test_array_round_trip_for_compressed_and_uncompressed_files(tmp_path):
    expected = np.arange(12).reshape(3, 4)

    utils.save_array(expected, tmp_path / "compressed", compression=True)
    utils.save_array(expected, tmp_path / "plain.anything", compression=False)

    np.testing.assert_array_equal(utils.load_array(tmp_path / "compressed"), expected)
    np.testing.assert_array_equal(utils.load_array(tmp_path / "plain"), expected)


def test_load_array_rejects_unknown_extension(capsys, tmp_path):
    assert utils.load_array(tmp_path / "values.txt") is None
    assert "extension not recognized" in capsys.readouterr().out


def test_generate_distance_matrix_uses_default_and_explicit_centers():
    default = utils.generate_distance_matrix((3, 3))
    explicit = utils.generate_distance_matrix((2, 3), center=(0, 1))

    assert default[1, 1] == 0
    assert default[0, 0] == np.sqrt(2)
    np.testing.assert_allclose(explicit, [[1, 0, 1], [np.sqrt(2), 1, np.sqrt(2)]])


def test_fill_3d_holes_fills_enclosed_region_only():
    shell = np.ones((5, 5, 5), dtype=int)
    shell[1:4, 1:4, 1:4] = 0
    shell[0, 0, 0] = 0

    filled = utils.fill_3d_holes(shell)

    assert filled[2, 2, 2] == 1
    assert filled[0, 0, 0] == 0


def test_compute_convex_hull_mask_marks_inside_points():
    vertices = np.array(
        [[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=float
    )
    query = np.array([[[[0.25, 0.25, 0.25], [2, 2, 2]]]])

    mask = utils.compute_convex_hull_mask(vertices, query)

    assert mask[0, 0, 0] == 1
    assert np.isnan(mask[0, 0, 1])

