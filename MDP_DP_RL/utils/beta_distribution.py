import numpy as np
from typing import Tuple
from scipy.special import digamma


class BetaDistribution:
    SMALL_POS = 1e-8

    def __init__(self, mu, nu) -> None:
        if 0 < mu < 1 and nu > 0:
            self.mu = mu
            self.nu = nu
            self.alpha = mu * nu
            self.beta = (1.0 - mu) * nu
        else:
            raise ValueError("mu= %.3f, nu= %.3f" % (mu, nu))

    def get_samples(self, n: int) -> np.ndarray:
        sp = BetaDistribution.SMALL_POS
        return np.vectorize(lambda x: min(1. - sp,
                                          max(sp, x)))(np.random.beta(a=self.alpha,
                                                                      b=self.beta, size=n))

    def get_mu_nu_scores(self, x: float) -> Tuple[float, float]:
        dig_a = digamma(self.alpha)
        dig_b = digamma(self.beta)
        dig_n = digamma(self.nu)
        lx = np.log(x)
        llx = np.log(1. - x)
        temp = dig_b - dig_a + lx - llx
        r1 = temp * self.nu
        r2 = temp * self.mu + dig_n - dig_b + llx
        return r1, r2
