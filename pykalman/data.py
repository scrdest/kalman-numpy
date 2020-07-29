import numpy as np
import pprint


class FilterData:
    def __init__(self, *args, **kwargs):
        self.mean = self.build_mean(*args, **kwargs)
        self.covariance = self.build_covariance(*args, **kwargs)


    def __str__(self) -> str:
        fmt_str = '{name}: \n Mean:\n {mu} \n\n Covariance:\n {cov}'.format(
            name=repr(self),
            mu=pprint.pformat(self.mean),
            cov=pprint.pformat(self.covariance)
        )
        return fmt_str


    @classmethod
    def build_mean(cls, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError


    @classmethod
    def build_covariance(cls, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError


    @classmethod
    def transition_matrix(cls, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError



class DynamicsData(FilterData):
    def __init__(
        self,
        pos=0, vel=0,
        pos_var=1, vel_var=1,
        pos_cov_vel=0, vel_cov_pos=0,
        *args, **kwargs
    ):
        super().__init__(
            pos=pos,
            vel=vel,
            pos_var=pos_var,
            vel_var=vel_var,
            pos_cov_vel=pos_cov_vel,
            vel_cov_pos=vel_cov_pos,
            *args, **kwargs
        )

    @classmethod
    def build_mean(cls, pos=0, vel=0, *args, **kwargs):
        mean = np.array([
            pos,
            vel
        ])
        return mean

    @classmethod
    def build_covariance(
        cls,
        pos_var=1, vel_var=1,
        pos_cov_vel=0, vel_cov_pos=0,
        *args, **kwargs
    ):
        cov = np.array([
            [pos_var, pos_cov_vel],
            [vel_cov_pos, vel_var]
        ])
        return cov


    @classmethod
    def transition_matrix(cls, timestep=1, *args, **kwargs):
        transition_matrix = np.array([
            [1, timestep],
            [0, 1]
        ])
        return transition_matrix
