from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from chronpy.data import LightCurve
from chronpy.modwt.wmtsa.modwt import pyramid
from chronpy.psd import PSD

if TYPE_CHECKING:
    NDArray = np.ndarray


def modwt_psd(
    x,
    dt,
    norm=None,
    wavelet='Haar',
    level: int | None = None,
    estimator='b',  # 'b' for biased, 'u' for unbiased
):
    if estimator not in ('b', 'u'):
        raise ValueError('estimator must be either "b" or "u"')
    x = np.ascontiguousarray(x)
    if estimator == 'b':
        x = np.append(x, np.flip(x))
    else:
        raise NotImplementedError('Only biased estimator is implemented')
    dt = float(dt)
    assert x.ndim == 1, 'Only 1D arrays are supported'
    assert dt > 0, 'dt must be positive'
    max_level = int(np.log2(x.size))
    if level is None:
        level = max_level
    else:
        level = min(int(level), max_level)
    # v, *w = pywt.swt(x, wavelet, level, trim_approx=True, norm=True)
    # w = np.array(w)
    w, v = pyramid(x, wavelet, level)
    octave = np.cumprod(np.full(level + 1, 2))
    p_bins = octave * dt
    f_bins = 1.0 / p_bins
    if norm is None:
        norm = 2.0 / (x.size * x.var(ddof=1))  # Leahy normalization
    else:
        norm = float(norm)
    if estimator == 'b':
        # the factor of 0.5 is to account for the fact that the reflection
        # boundary condition is used
        norm = norm * 0.5
    power = np.square(np.abs(w)).sum(1) * 0.5 * norm / dt
    f_bins = np.ascontiguousarray(np.flip(f_bins))
    power = np.ascontiguousarray(np.flip(power))
    edof = np.maximum(np.ascontiguousarray(x.size / np.flip(octave[1:])), 1.0)
    return f_bins, power, edof


class MODWTVariance(PSD):
    @classmethod
    def from_lc(cls, lc: LightCurve | list[LightCurve], norm: str = 'leahy'):
        if isinstance(lc, LightCurve):
            lc = [lc]
        elif not all(isinstance(i, LightCurve) for i in lc):
            raise ValueError('lc must be a LightCurve or a list of LightCurve')

        t = lc[0].t
        for i in lc:
            if not i.evenly_sampled:
                raise ValueError('LightCurve must be evenly sampled')
            if not np.allclose(i.t, t):
                raise ValueError('LightCurves must have the same times')

        exposure = np.vstack([i.exposure for i in lc])
        rate = np.array([i.rate for i in lc])
        rate_total = np.sum(rate, axis=0)
        dt = float(lc[0].dt)
        df = 1.0 / (rate.shape[1] * dt)
        norm = 2.0 / np.sum(rate / exposure)
        f_bins, power, edof = modwt_psd(rate_total, dt, norm=norm)
        return cls(f_bins, power, edof, dt, df)

    @property
    def perr(self) -> NDArray:
        if self._perr is None:
            self._perr = np.sqrt(2 / self._dof) * self._power
        return self._perr


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    n = 2**10 + 1
    x = np.random.default_rng(42).poisson(1000, n)
    f_bins, power, edof = modwt_psd(x, 1e-4, norm=2.0 / x.sum())

    d = power / np.diff(f_bins)
    plt.step(f_bins, np.append(d, d[-1]), where='mid')
    plt.loglog()
    plt.axhline(2, color='k', ls='--')
    from chronpy.data import LightCurve
    from chronpy.psd import Fit
    from chronpy.psd.models import Const

    lc = LightCurve(np.arange(x.size) * 1e-4, x)
    psd = MODWTVariance.from_lc(lc)
    print(Fit(psd, Const()).mle())
