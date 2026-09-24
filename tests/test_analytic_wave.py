import numpy as np

from musik import analytic_wave


def test_exponential_expression_is_causal():
    time = np.array([-1.0, 0.0, 1.0])

    result = analytic_wave.exponential_expression(2, 3, 2, time, 0)

    np.testing.assert_allclose(result, [0, 6, 6 * np.exp(-0.5)])


def test_convolution_processes_each_signal_row():
    fourier = np.array([[1, 0, 0], [0, 1, 0]])
    exponential = np.array([1, 2, 1])

    result = analytic_wave.convolution(fourier, exponential)

    np.testing.assert_array_equal(
        result[0], np.convolve(exponential, fourier[0], mode="same")
    )
    np.testing.assert_array_equal(
        result[1], np.convolve(exponential, fourier[1], mode="same")
    )


def test_fourier_expression_is_finite_at_origin():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]])

    result = analytic_wave.fourier_expression(points, 1, 1, [0], 1, 1500)

    assert result.shape == (2,)
    assert np.all(np.isfinite(result))


def test_compute_signal_shape_matches_points_and_time():
    points = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    frequencies = np.linspace(1e6, 1.1e6, 5)
    time = np.linspace(0, 1e-5, 5)

    signal = analytic_wave.compute_signal(
        points, 0.01, frequencies, [0], 0.01, 1500, 1000, 1, 1e-6, time, 1e6
    )

    assert signal.shape == (2, 5)


def test_extract_pressure_uses_distance_based_sample():
    signal = np.array([[1, -2, 3], [4, 5, -6]], dtype=complex)
    points = np.array([[1, 0, 0], [2, 0, 0]])

    pressure = analytic_wave.extract_pressure(signal, 1, points)

    np.testing.assert_array_equal(pressure, [2, 6])

