from rllab.baselines.base import Baseline
from rllab.misc.overrides import overrides
import numpy as np


class LinearFeatureBaseline(Baseline):
    def __init__(self, env_spec, reg_coeff=1e-5):
        self._coeffs = None
        self._reg_coeff = reg_coeff

    @overrides
    def get_param_values(self, **tags):
        return self._coeffs

    @overrides
    def set_param_values(self, val, **tags):
        self._coeffs = val

    def _features(self, path):
        o = np.clip(path["observations"], -10, 10)
        l = len(path["rewards"])
        al = np.arange(l).reshape(-1, 1) / 100.0
        return np.concatenate([o, o ** 2, al, al ** 2, al ** 3, np.ones((l, 1))], axis=1)

    @overrides
    def fit(self, paths):
        featmat = np.concatenate([self._features(path) for path in paths])
        # print('featmat shape', featmat.shape) # (50000, 40)
        returns = np.concatenate([path["returns"] for path in paths])
        # print('returns shape', returns.shape) # (50000)
        reg_coeff = self._reg_coeff
        for _ in range(5):
            self._coeffs = np.linalg.lstsq(
                featmat.T.dot(featmat) + reg_coeff * np.identity(featmat.shape[1]),
                featmat.T.dot(returns)
            )[0]
            if not np.any(np.isnan(self._coeffs)):
                break
            reg_coeff *= 10
        # print('coeff shape', self._coeffs.shape) # (40)

    @overrides
    def predict(self, path):
        if self._coeffs is None:
            return np.zeros(len(path["rewards"]))
        # print('feature shape', self._features(path).shape) # (100,40)
        return self._features(path).dot(self._coeffs)

    def gradient(self, observations):
        # TODO: it fits with paths, how to infer with samples
        # raise NotImplementedError
        if self._coeffs is None:
            return np.ones(observations.shape)
        obs_size = observations.shape[1]
        # l = self._coeffs.shape[1]
        c1 = self._coeffs[0: obs_size]
        c2 = self._coeffs[obs_size : obs_size * 2]
        # return np.concatenate([c1 + 2 * observations[i, :] * c2 for i in range(observations.shape[0])])
        return c1.T + observations * c2.T * 2
