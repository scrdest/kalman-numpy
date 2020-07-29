import typing

import numpy as np


def covariance_mult(covariance: np.ndarray, mult_matrix: np.ndarray) -> np.ndarray:
    new_cov = mult_matrix @ (covariance @ mult_matrix.T)
    return new_cov


def apply_transition(
    mean: np.ndarray,
    covariance: np.ndarray,
    transition_matrix: np.ndarray
):

    result_mean = transition_matrix @ mean
    result_cov = covariance_mult(
        covariance=covariance,
        mult_matrix=transition_matrix
    )

    return (result_mean, result_cov)


def calculate_control(
    control_matrix: typing.Optional[np.ndarray],
    control_vector: typing.Optional[np.ndarray]
) -> np.ndarray:

    result = (
        None if None in (control_matrix, control_vector)
        else control_matrix @ control_vector
    )

    return result


def apply_control(
    pre_ctrl_mean: np.ndarray,
    calculated_control: np.ndarray
) -> np.ndarray:

    post_ctrl_mean = pre_ctrl_mean
    if calculated_control is not None:
        post_ctrl_mean = pre_ctrl_mean + calculated_control

    return post_ctrl_mean


def apply_uncertainty(
    pre_noise_cov: np.ndarray,
    uncertainty: np.ndarray
) -> np.ndarray:

    noisy_cov = pre_noise_cov
    if uncertainty is not None:
        noisy_cov = pre_noise_cov + uncertainty

    return noisy_cov


def calculate_kalman_gain(
    expect_covariance: np.ndarray,
    measure_covariance: np.ndarray
):

    covsum = expect_covariance + measure_covariance

    gain_val = expect_covariance @ np.linalg.inv(covsum)

    return gain_val


def apply_kalman_gain(
    expected_mean: np.ndarray,
    measured_mean: np.ndarray,
    expected_covariance: np.ndarray,
    measured_covariance: np.ndarray
):
    mean_difference = measured_mean - expected_mean

    gain = calculate_kalman_gain(
        expect_covariance=expected_covariance,
        measure_covariance=measured_covariance
    )

    kalmanized_mean = expected_mean + (gain @ mean_difference)
    kalmanized_covariance = expected_covariance - (gain @ expected_covariance)

    return kalmanized_mean, kalmanized_covariance