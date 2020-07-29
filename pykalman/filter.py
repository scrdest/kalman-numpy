import typing

import numpy as np

from pykalman.backend import apply_transition, calculate_control, apply_control, apply_uncertainty, apply_kalman_gain
from pykalman.data import FilterData, DynamicsData


def predict(
    curr_mean: np.ndarray,
    curr_covariance: np.ndarray,
    transition_matrix: np.ndarray,
    control_val: typing.Optional[np.ndarray] = None,
    external_noise: typing.Optional[np.ndarray] = None
):

    raw_mean, raw_cov = apply_transition(
        mean=curr_mean,
        covariance=curr_covariance,
        transition_matrix=transition_matrix
    )

    post_control_mean = apply_control(
        pre_ctrl_mean=raw_mean,
        calculated_control=control_val
    )

    post_noise_cov = apply_uncertainty(
        pre_noise_cov=raw_cov,
        uncertainty=external_noise
    )

    return post_control_mean, post_noise_cov


def measure(true_pos: np.ndarray, measurement_noise=None):
    meas_noise = 1. if measurement_noise is None else measurement_noise
    meas_mean = np.random.normal(loc=true_pos, scale=meas_noise)
    meas_cov = np.eye(true_pos.shape[0])
    return meas_mean, meas_cov


def kalman_step(
    curr_mean: np.ndarray,
    curr_covariance: np.ndarray,
    transition_matrix: np.ndarray,
    control: typing.Optional[np.ndarray] = None,
    noise: typing.Optional[np.ndarray] = None,
    true_pos: typing.Optional[np.ndarray] = None,
    measurement_noise: typing.Optional[float] = None
):

    predicted_mean, predicted_cov = predict(
        curr_mean=curr_mean,
        curr_covariance=curr_covariance,
        transition_matrix=transition_matrix,
        control_val=control,
        external_noise=noise
    )

    _true_pos = curr_mean if true_pos is None else true_pos
    measured_mean, measured_cov = measure(
        true_pos=_true_pos,
        measurement_noise=measurement_noise
    )

    kalmanized_mean, kalmanized_cov = apply_kalman_gain(
        expected_mean=predicted_mean,
        measured_mean=measured_mean,
        expected_covariance=predicted_cov,
        measured_covariance=measured_cov
    )

    return kalmanized_mean, kalmanized_cov


def main(
    data: typing.Optional[FilterData] = None,
    controller: typing.Optional[typing.Callable] = None
):

    _data = data or DynamicsData()
    true_position = np.array([3, 0.25])

    curr_mean = _data.mean
    curr_cov = _data.covariance
    transition_mat = _data.transition_matrix()

    control_matrix = None

    print(_data)

    for i in range(15):
        print(f'='*100)
        print(f'Iteration {i}')
        print(f'Mean:\n {curr_mean}')
        print(f'Cov:\n {curr_cov}')

        control_vector = None if controller is None else controller()

        control_val = calculate_control(
            control_matrix=control_matrix,
            control_vector=control_vector
        )

        curr_noise = (
            (0.05 * np.random.random(1))
            * np.eye(curr_cov.shape[0])
        )

        curr_mean, curr_cov = kalman_step(
            curr_mean=curr_mean,
            curr_covariance=curr_cov,
            transition_matrix=transition_mat,
            control=control_val,
            noise=curr_noise,
            true_pos=true_position
        )
