from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    NDArray = np.ndarray


class LightCurve:
    def __init__(
        self,
        t: NDArray,
        counts: NDArray,
        tbins: NDArray = None,
        exposure: NDArray = None,
    ):
        t = np.array(t, dtype=float, order='C', ndmin=1)
        dt = t[1:] - t[:-1]
        if np.any(dt < 0):
            raise ValueError('t must be sorted in ascending order')
        self._t = t

        self._evenly_sampled = np.allclose(dt, dt[0])
        if self._evenly_sampled:
            self._dt = np.mean(dt)
        else:
            self._dt = None

        counts = np.array(counts, dtype=float, order='C', ndmin=1)
        if counts.size != t.size:
            raise ValueError('counts must have the same size as t')
        if np.any(counts < 0):
            raise ValueError('counts must be non-negative')
        if not np.allclose(np.round(counts), counts):
            raise ValueError('counts must be integers')
        self._counts = counts

        if tbins is not None:
            tbins = np.array(tbins, dtype=float, order='C', ndmin=1)
            if tbins.size != t.size + 1:
                raise ValueError('tbins must have size t.size + 1')
            if tbins[0] > t[0] or tbins[-1] < t[-1]:
                raise ValueError('t must be within the range of tbins')
            if np.any(tbins[1:] - tbins[:-1] < 0):
                raise ValueError('tbins must be sorted in ascending order')
        elif self._evenly_sampled:
            tbins = np.hstack([t[0] - 0.5 * self._dt, t + 0.5 * self._dt])
        self._tbins = tbins

        if exposure is not None:
            exposure = np.array(exposure, dtype=float, order='C', ndmin=1)
            if exposure.size != t.size:
                raise ValueError('exposure must have the same size as t')
            if np.any(exposure <= 0):
                raise ValueError('exposure must be positive')
        elif self._evenly_sampled:
            exposure = np.full(self._t.size, self._dt)
        elif self._tbins is not None:
            exposure = np.diff(self._tbins)
        else:
            raise ValueError(
                'exposure or tbins must be provided if t is not evenly sampled'
            )
        self._exposure = exposure

    @property
    def t(self) -> NDArray:
        return self._t

    @property
    def counts(self) -> NDArray:
        return self._counts

    @property
    def rate(self) -> NDArray:
        return self._counts / self._exposure

    @property
    def tbins(self) -> NDArray | None:
        return self._tbins

    @property
    def exposure(self) -> NDArray:
        return self._exposure

    @property
    def dt(self) -> float | None:
        return self._dt

    @property
    def evenly_sampled(self) -> bool:
        return self._evenly_sampled
