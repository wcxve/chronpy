from __future__ import annotations

from collections.abc import Sequence
from sys import stdout
from typing import TYPE_CHECKING, Literal, NamedTuple

import numexpr as ne
import numpy as np
from scipy.stats import norm
from tqdm.auto import tqdm, trange

if TYPE_CHECKING:
    from numpy import ndarray as NDArray

__all__ = ['blocks_binned', 'blocks_tte']


class BayesianBlocksData(NamedTuple):
    """Data container for Bayesian Blocks algorithm."""

    voronoi: NDArray
    """Voronoi tessellation of the data points."""

    n: NDArray
    """The counts of each bin."""

    t: NDArray
    """The exposure of each bin."""

    n_remainder: NDArray
    """The counts remained after each bin."""

    t_remainder: NDArray
    """The exposure remained after each bin."""

    n_data: int
    """The total number of data points."""


class ChangePointPosterior(NamedTuple):
    """The posterior of the change point."""

    locs: NDArray
    """The location of the change point."""

    prob: NDArray
    """The probability distribution of the change point."""


class ChangePointSignificance(NamedTuple):
    """The significance of the change point."""

    llr: float
    """The likelihood ratio test statistics, i.e., -2 log likelihood ratio."""

    lor: float
    """The log of odds ratio."""

    # sig: NDArray
    # """The significance of the change point."""


class BayesianBlocksResult(NamedTuple):
    """The result of the Bayesian Blocks algorithm."""

    data: BayesianBlocksData
    """The data used for the Bayesian Blocks algorithm."""

    edge: NDArray
    """The edges of the blocks."""

    counts: NDArray
    """The counts of each block."""

    exposure: NDArray
    """The exposure of each block."""

    cp: NDArray
    """The index of change points."""

    prob: tuple[ChangePointPosterior, ...]
    """The probability distribution of each change point."""

    significance: tuple[ChangePointSignificance, ...]
    """The significance of each change point."""

    iteration: int | None
    """The number of iterations."""

    convergence: bool | None
    """Whether the iteration converges."""

    ncp_prior: float | NDArray
    """The prior on the number of change points for each iteration."""


def _sanitize_input(
    t: NDArray | Sequence[NDArray],
    live_time: NDArray | Sequence[NDArray] | None,
    tstart: float | None,
    tstop: float | None,
    ltstart: float | Sequence[float] | None,
    ltstop: float | Sequence[float] | None,
) -> tuple[
    Sequence[NDArray],
    Sequence[NDArray],
    float,
    float,
    NDArray,
    NDArray,
]:
    if not isinstance(t, np.ndarray | Sequence):
        raise TypeError('`t` must be a numpy array or a list of numpy arrays')

    if isinstance(t, Sequence):
        if not all(isinstance(ti, np.ndarray) for ti in t):
            raise TypeError('elements of `t` must be numpy arrays')
    else:
        t = [t]

    if any(np.any(ti[:-1] > ti[1:]) for ti in t):
        raise ValueError('elements of `t` must be ordered')

    if (tstart is None) ^ (tstop is None):
        raise ValueError('`tstart` and `tstop` must be both provided')
    if tstart is not None and tstart >= tstop:
        raise ValueError('`tstart` must be less than `tstop`')
    if (ltstart is None) ^ (ltstop is None):
        raise ValueError('`ltstart` and `ltstop` must be both provided')
    if ltstart is not None and ltstart >= ltstop:
        raise ValueError('`ltstart` must be less than `ltstop`')

    if (
        (tstart is not None)
        and (live_time is not None)
        and ((ltstart is None) or (ltstop is None))
    ):
        raise ValueError(
            '`ltstart` and `ltstop` must be provided if `tstart`, `tstop` '
            'and `live_time` are provided'
        )

    if live_time is not None:
        if not isinstance(live_time, np.ndarray | Sequence):
            raise TypeError(
                '`live_time` must be a numpy array or a list of numpy arrays'
            )
        if isinstance(live_time, Sequence):
            if not all(isinstance(lti, np.ndarray) for lti in live_time):
                raise TypeError('elements of `live_time` must be numpy arrays')
        else:
            live_time = [live_time]

        if len(t) != len(live_time):
            raise ValueError(
                f'length of `t` ({len(t)})  and `live_time` '
                f'({len(live_time)}) are not the same'
            )

        if not all(isinstance(lti, np.ndarray) for lti in live_time):
            raise TypeError('elements of `live_time` must be numpy arrays')

        len_t = [len(ti) for ti in t]
        len_lt = [len(lti) for lti in live_time]
        if len_t != len_lt:
            raise ValueError(
                f'length of each element of `t` ({len_t}) and '
                f'`live_time` ({len_lt}) are not the same'
            )
    else:
        live_time = t

    t = [np.array(i, dtype=np.float64, order='C') for i in t]
    live_time = [np.array(i, dtype=np.float64, order='C') for i in live_time]

    if tstart is not None:
        mask = [(tstart <= ti) & (ti <= tstop) for ti in t]
        if not any(i.any() for i in mask):
            raise ValueError('no data points between `tstart` and `tstop`')
        t = [ti[maski] for ti, maski in zip(t, mask, strict=True)]
        live_time = [
            lti[maski] for lti, maski in zip(live_time, mask, strict=True)
        ]
    else:
        tstart = min(ti[0] for ti in t)
        tstop = max(ti[-1] for ti in t)

    if ltstart is None:
        ltstart = [lt[0] for lt in live_time]
        ltstop = [lt[-1] for lt in live_time]

    if not isinstance(ltstart, float | Sequence):
        raise TypeError('`ltstart` must be a float or a list of float')

    if not isinstance(ltstop, float | Sequence):
        raise TypeError('`ltstop` must be a float or a list of float')

    if isinstance(ltstart, Sequence) and any(np.shape(i) for i in ltstart):
        raise ValueError('`ltstart` must be a scalar or a list of scalar')
    ltstart = np.array(ltstart, dtype=np.float64, order='C', ndmin=1)

    if isinstance(ltstop, Sequence) and any(np.shape(i) for i in ltstop):
        raise ValueError('`ltstop` must be a scalar or a list of scalar')
    ltstop = np.array(ltstop, dtype=np.float64, order='C', ndmin=1)

    if len(ltstart) != len(t):
        raise ValueError(
            f'length of `ltstart` ({len(ltstart)}) and `t` ({len(t)}) '
            'are not the same'
        )

    if len(ltstop) != len(t):
        raise ValueError(
            f'length of `ltstop` ({len(ltstop)}) and `t` ({len(t)}) '
            'are not the same'
        )

    for i, j, k in zip(ltstart, ltstop, live_time, strict=True):
        if i >= j:
            raise ValueError('`ltstart` must be less than `ltstop`')

        if i > k[0]:
            raise ValueError(
                'filtered `live_time` is less than `ltstart`, check '
                'the `live_time` and `ltstart`'
            )
        if j < k[-1]:
            raise ValueError(
                'filtered `live_time` is greater than `ltstop`, check '
                'the `live_time` and `ltstop`'
            )

    return t, live_time, tstart, tstop, ltstart, ltstop


# BUG: the exposure and fitness function maybe wrong for multiple series
def _get_data_from_tte(
    t: NDArray | list[NDArray],
    live_time: NDArray | list[NDArray] | None = None,
    tstart: float | None = None,
    tstop: float | None = None,
    ltstart: float | None = None,
    ltstop: float | None = None,
) -> BayesianBlocksData:
    t, live_time, tstart, tstop, ltstart, ltstop = _sanitize_input(
        t, live_time, tstart, tstop, ltstart, ltstop
    )

    t_unq = []
    lt_unq = []
    counts = []
    for ti, lt in zip(t, live_time, strict=True):
        unq, idx, c = np.unique(lt, return_index=True, return_counts=True)
        t_unq.append(ti[idx + c - 1])
        lt_unq.append(unq)
        counts.append(c)

    t_all = np.hstack(t_unq)
    argsort = t_all.argsort()
    t_all = np.sort(t_all)
    edges = np.hstack([tstart, 0.5 * (t_all[1:] + t_all[:-1]), tstop])

    n_total = sum(map(len, t))
    n_row = len(t_unq)
    n_col = t_all.size
    n_data = list(map(len, t_unq))

    # zeros array for convenience
    zeros = np.zeros(shape=(n_row, 1), dtype=int)

    # initialize data matrix
    n = np.zeros(shape=(n_row, n_col), dtype=int)

    # the index of the first element of each series
    n_idx = np.hstack([0, n_data]).cumsum()

    # fill the data matrix
    for i in range(n_row):
        n[i][n_idx[i] : n_idx[i + 1]] = counts[i]
    n = n[:, argsort]
    n_cumsum = np.hstack((zeros, n.cumsum(axis=1)))
    n_remainder = n_cumsum[:, -1:] - n_cumsum

    t_idx = np.hstack((zeros, np.where(n, 1, 0).cumsum(axis=1)))
    t_cumsum = np.zeros_like(n_cumsum, dtype=np.float64)
    for i in range(n_row):
        unq_i = lt_unq[i]
        t_cs = np.hstack(
            (ltstart[i], 0.5 * (unq_i[1:] + unq_i[:-1]), ltstop[i])
        )
        t_cumsum[i] = t_cs[t_idx[i]]
    t_remainder = t_cumsum[:, -1:] - t_cumsum

    n = np.ascontiguousarray(n, dtype=int)
    t = np.ascontiguousarray(np.diff(t_cumsum), dtype=float)
    n_remainder = np.ascontiguousarray(n_remainder, dtype=int)
    t_remainder = np.ascontiguousarray(t_remainder, dtype=float)

    if n_row == 1:
        n = n[0]
        t = t[0]
        n_remainder = n_remainder[0]
        t_remainder = t_remainder[0]

    return BayesianBlocksData(
        voronoi=np.ascontiguousarray(edges, dtype=float),
        n=n,
        t=t,
        n_remainder=n_remainder,
        t_remainder=t_remainder,
        n_data=n_total,
    )


def _estimate_run_time(ndata: int) -> str:
    # run time estimated based on 8 performance-cores of M1 Max and vecLib
    s = 300 * (ndata / 660000) ** 2
    h = int(s / 3600)
    m = int((s - h * 3600) / 60)
    s = round(s - h * 3600 - m * 60)
    string = ''
    if h:
        string += f'{h} hr '
    if m or h:
        string += f'{m} min '
    string += f'{s} sec'
    return string


# -----------------------------------------------------------------
# Speed tricks: resolve once for fitness function used in the loop
# -----------------------------------------------------------------
# N - n: number of elements in each block
# T - t: width or duration of each block
_fitness1 = ne.NumExpr(
    ex='(N - n) * log((N - n) / (T - t)) + best - ncp_prior',
    signature=(
        ('N', np.float64),
        ('n', np.float64),
        ('T', np.float64),
        ('t', np.float64),
        ('ncp_prior', np.float64),
        ('best', np.float64),
    ),
    optimization='aggressive',
    truediv=True,
)


def _loop1(
    n_remainder: NDArray,
    t_remainder: NDArray,
    ncp_prior: float,
    desc: str | None = None,
    progress: bool = True,
) -> NDArray:
    ne.set_num_threads(ne._init_num_threads())

    t_remainder = np.asarray(t_remainder, dtype=np.float64)
    n_remainder = np.asarray(n_remainder, dtype=np.float64)
    n = len(n_remainder) - 1

    # arrays to store the best configuration
    tmp = np.empty(n, dtype=np.float64)
    best = np.zeros(n + 1, dtype=np.float64)
    last = np.zeros(n + 1, dtype=np.int64)

    # -----------------------------------------------------------------
    # Start core loop, add one cell at each iteration
    # -----------------------------------------------------------------
    range_ = (
        trange(1, n + 1, desc=desc, file=stdout)
        if progress
        else range(1, n + 1)
    )
    for r in range_:
        ar = tmp[:r]
        _fitness1(
            n_remainder[:r],
            n_remainder[r],
            t_remainder[:r],
            t_remainder[r],
            ncp_prior,
            best[:r],
            out=ar,
            order='K',
            casting='no',
            ex_uses_vml=False,
        )

        imax = ar.argmax()
        last[r] = imax
        best[r] = ar[imax]

    # -----------------------------------------------------------------
    # Now find change points by iteratively peeling off the last block
    # -----------------------------------------------------------------
    idx = n
    cp = [idx]
    while True:
        idx = last[idx]
        cp.append(idx)
        if idx == 0:
            break

    return np.flip(cp)


# -----------------------------------------------------------------
# Speed tricks: resolve once for fitness function used in the loop
# -----------------------------------------------------------------
# N - n: number of elements in each block
# T - t: width or duration of each block
_fitness2 = ne.NumExpr(
    ex='(N - n) * log((N - n) / (T - t))',
    signature=(
        ('N', np.float64),
        ('n', np.float64),
        ('T', np.float64),
        ('t', np.float64),
    ),
    optimization='aggressive',
    truediv=True,
)


def _loop2(
    n_remainder: NDArray,
    t_remainder: NDArray,
    ncp_prior: float,
    desc: str | None = None,
    progress: bool = True,
) -> NDArray:
    ne.set_num_threads(ne._init_num_threads())

    n_remainder = np.asarray(n_remainder, dtype=np.float64, order='C')
    t_remainder = np.asarray(t_remainder, dtype=np.float64, order='C')
    n = n_remainder.shape[0]
    N = n_remainder.shape[1] - 1

    # arrays to store the intermediate results and the best configuration
    # for each col, get the index of row where the value is non-zero
    col, row = np.where(np.transpose(n_remainder[:, :-1] > n_remainder[:, 1:]))
    col += 1
    idx = np.column_stack((col, row))
    f = np.full((n, N), 0.0, dtype=np.float64)
    f_sum = np.full(N, -ncp_prior, dtype=np.float64)
    tmp = np.empty(N, dtype=np.float64)
    best = np.zeros(N + 1, dtype=np.float64)
    last = np.zeros(N + 1, dtype=np.int64)

    # -----------------------------------------------------------------
    # Start core loop, add one cell at each iteration
    # -----------------------------------------------------------------
    idx = tqdm(idx, desc=desc, file=stdout) if progress else idx
    for r, i in idx:
        f_sum_r = f_sum[:r]
        tmp_r = tmp[:r]

        _fitness2(
            n_remainder[i, :r],
            n_remainder[i, r],
            t_remainder[i, :r],
            t_remainder[i, r],
            out=tmp_r,
            order='K',
            casting='no',
            ex_uses_vml=False,
        )

        np.add(f_sum_r, tmp_r - f[i, :r], out=f_sum_r)
        f[i, :r] = tmp_r
        np.add(f_sum_r, best[:r], out=tmp_r)

        imax = tmp_r.argmax()
        last[r] = imax
        best[r] = tmp_r[imax]

    # -----------------------------------------------------------------
    # Now find change points by iteratively peeling off the last block
    # -----------------------------------------------------------------
    idx = N
    cp = [idx]
    while True:
        idx = last[idx]
        cp.append(idx)
        if idx == 0:
            break

    return np.flip(cp)


def _get_cp_prob(
    data: BayesianBlocksData,
    cp: NDArray[int],
    i: int,
) -> ChangePointPosterior:
    assert 0 < i < len(cp) - 1

    nr = data.n_remainder
    tr = data.t_remainder

    # ｜- - - -｜- - - -｜
    # start   cp_i    stop
    start = cp[i - 1]
    stop = cp[i + 1]

    nr_start = nr[..., start : start + 1]
    nr_stop = nr[..., stop : stop + 1]

    tr_start = tr[..., start : start + 1]
    tr_stop = tr[..., stop : stop + 1]

    cp_slice = slice(start + 1, stop)
    n_left = nr_start - nr[..., cp_slice]
    n_right = nr[..., cp_slice] - nr_stop
    t_left = tr_start - tr[..., cp_slice]
    t_right = tr[..., cp_slice] - tr_stop

    lnL = np.zeros_like(n_left, dtype=float)
    mask = n_left != 0
    lnL[mask] += n_left[mask] * np.log(n_left[mask] / t_left[mask])
    mask = n_right != 0
    lnL[mask] += n_right[mask] * np.log(n_right[mask] / t_right[mask])

    if lnL.ndim == 2:
        lnL = lnL.sum(axis=0)

    prob = np.exp(lnL - lnL.max())
    prob /= prob.sum()

    return ChangePointPosterior(locs=data.voronoi[cp_slice], prob=prob)


def _get_cp_significance(
    data: BayesianBlocksData,
    cp: NDArray,
    i: int,
    ncp_prior: float,
) -> ChangePointSignificance:
    assert 0 < i < len(cp) - 1

    # ｜- - - -｜- - - -｜
    # start   cp_i    stop
    idx = cp[i - 1 : i + 2]
    nr = data.n_remainder[..., idx]
    tr = data.t_remainder[..., idx]

    n = nr[..., :-1] - nr[..., 1:]
    t = tr[..., :-1] - tr[..., 1:]

    fitness_null_hyp = 0.0
    n_all = n.sum(axis=-1)
    t_all = t.sum(axis=-1)
    mask = n_all > 0
    fitness_null_hyp += np.sum(n_all[mask] * np.log(n_all[mask] / t_all[mask]))

    mask = n > 0
    fitness_alt_hyp = np.sum(n[mask] * np.log(n[mask] / t[mask]))
    ln_lr = fitness_alt_hyp - fitness_null_hyp

    return ChangePointSignificance(llr=2 * ln_lr, lor=ln_lr - ncp_prior)


def _bayesian_blocks(
    data: BayesianBlocksData,
    p0: float,
    iteration: int,
    show_progress: bool,
    data_type: Literal['EVT', 'BIN'],
) -> BayesianBlocksResult:
    len8_rjust = lambda s: str(s).rjust(2).rjust(5).ljust(8)
    len8_ljust = lambda s: str(s).ljust(2).rjust(5).ljust(8)

    n = data.n_data
    n_remainder = data.n_remainder
    t_remainder = data.t_remainder

    timing = _estimate_run_time(np.sum(data.n.astype(bool)))
    if show_progress:
        print(f'\nBayesian Blocks: about {timing} to go')

    ncp_prior = 4 - np.log(73.53 * p0 * (n**-0.478))

    if n_remainder.ndim > 1 or data_type == 'BIN':
        core_loop = _loop2
    else:
        core_loop = _loop1

    cp = core_loop(
        n_remainder, t_remainder, ncp_prior, f'  {data_type} ', show_progress
    )
    ncp = cp.size
    fpr = 1 - (1 - p0) ** ncp
    if show_progress:
        print(f'  NCP : {ncp}\n  FPR : {fpr:.2e}\n')

    # -----------------------------------------------------------------
    # Iterate if desired
    # -----------------------------------------------------------------
    nit = None
    ncp_hist = [ncp]
    ncp_prior_hist = [ncp_prior]
    cp_hist = [cp]
    fpr_hist = [fpr]
    convergence = None

    if iteration > 0:
        if show_progress:
            print(
                'Bayesian Blocks: iteration starts, '
                f'each step takes about {timing}'
            )
        convergence = False
        for nit in range(1, iteration + 1):
            fpr_single = 1 - (1 - p0) ** (1 / ncp_hist[-1])
            ncp_prior = 4 - np.log(73.53 * fpr_single * (n**-0.478))
            ncp_prior_hist.append(ncp_prior)

            cp = core_loop(
                n_remainder,
                t_remainder,
                ncp_prior,
                f'{nit} '.rjust(6),
                show_progress,
            )
            cp_hist.append(cp)

            ncp = cp.size
            ncp_hist.append(ncp)

            fpr = 1 - (1 - fpr_single) ** ncp
            fpr_hist.append(fpr)
            if show_progress:
                print(
                    f'  NCP : {len8_rjust(ncp_hist[-2])} -> {len8_ljust(ncp)}'
                    '\n'
                    f'  FPR : {fpr_hist[-2]:.2e} -> {fpr:.2e}'
                    '\n'
                )

            if ncp == ncp_hist[-2]:
                convergence = True
                break

            if ncp in ncp_hist[:-2]:
                convergence = True

                fpr_hist = np.array(fpr_hist)

                mask = fpr_hist <= p0
                if any(mask):
                    idx = fpr_hist[mask].argmax()
                    idx = np.where(mask)[0][idx]
                else:
                    idx = fpr_hist.argmin()

                fpr = fpr_hist[idx]
                cp = cp_hist[idx]
                ncp = ncp_hist[idx]

                break

        if show_progress:
            cstr = 'converged' if convergence else 'not converged'
            print(
                f'Bayesian Blocks: iteration {cstr} within {nit} step(s)\n'
                f'  NCP : {len8_rjust(ncp_hist[0])} -> {len8_ljust(ncp)}\n'
                f'  FPR : {fpr_hist[0]:.2e} -> {fpr:.2e}\n'
            )

    nr = data.n_remainder[..., cp]
    tr = data.t_remainder[..., cp]
    counts = nr[..., :-1] - nr[..., 1:]
    exposure = tr[..., :-1] - tr[..., 1:]

    prob = (ChangePointPosterior(locs=data.voronoi[:1], prob=np.array([1.0])),)
    prob += tuple(_get_cp_prob(data, cp, i) for i in range(1, len(cp) - 1))
    prob += (
        ChangePointPosterior(locs=data.voronoi[-1:], prob=np.array([1.0])),
    )

    significance = (ChangePointSignificance(np.inf, np.inf),)
    significance += tuple(
        _get_cp_significance(data, cp, i, ncp_prior)
        for i in range(1, len(cp) - 1)
    )
    significance += (ChangePointSignificance(np.inf, np.inf),)

    return BayesianBlocksResult(
        data=data,
        edge=data.voronoi[cp],
        counts=counts,
        exposure=exposure,
        cp=cp,
        prob=prob,
        significance=significance,
        iteration=nit,
        convergence=convergence,
        ncp_prior=ncp_prior if nit is None else np.asarray(ncp_prior_hist),
    )


def blocks_tte(
    t: NDArray | list[NDArray],
    live_time: NDArray | None = None,
    p0: float = 0.05,
    iteration: int = 0,
    tstart: float | None = None,
    tstop: float | None = None,
    ltstart: float | None = None,
    ltstop: float | None = None,
    show_progress: bool = True,
) -> BayesianBlocksResult:
    """Run the Bayesian Blocks algorithm on the time-tagged event data.

    Parameters
    ----------
    t : ndarray or sequence of ndarray
        The time-tagged event data. If a sequence of ndarray is given, each
        element is treated as a separate series.
    live_time : ndarray or sequence of ndarray, optional
        The live time of each event. If not given, `live_time` is identical
        to `t`.
    p0 : float, optional
        The false positive rate. The default is 0.05.
    iteration : int, optional
        The number of iterations to run to ensure overall false positive rate
        converge to `p0`. The default is 0.
    tstart : float, optional
        The start time of the blocks. The default is the minimum of `t`.
    tstop : float, optional
        The stop time of the blocks. The default is the maximum of `t`.
    ltstart : float or sequence of float, optional
        The corresponding live time at `tstart` for each data in `t`.
    ltstop : float or sequence of float, optional
        The corresponding live time at `tstop` for each data in `t`.
    show_progress : bool, optional
        Whether to show the information of the progress. The default is True.

    Returns
    -------
    BayesianBlocksResult
        The result of the Bayesian Blocks algorithm.
    """
    p0 = float(p0)
    iteration = int(iteration)
    show_progress = bool(show_progress)
    if tstart is not None:
        tstart = float(tstart)
    if tstop is not None:
        tstop = float(tstop)

    if p0 <= 0.0 or p0 >= 1.0:
        raise ValueError('`p0` must be within (0,1)')

    if iteration < 0:
        raise ValueError('`iteration` must be non-negative')

    data = _get_data_from_tte(t, live_time, tstart, tstop, ltstart, ltstop)
    return _bayesian_blocks(data, p0, iteration, show_progress, 'EVT')


def blocks_binned(
    tbins: NDArray,
    counts: NDArray,
    exposure: NDArray | None = None,
    p0: float = 0.05,
    iteration: int = 0,
    show_progress: bool = True,
) -> BayesianBlocksResult:
    """Run the Bayesian Blocks algorithm on the binned data.

    Parameters
    ----------
    tbins : ndarray
        The time bins, with shape (nt + 1,).
    counts : ndarray
        The counts in each time bin, with shape (nd, nt) or (nt,).
    exposure : ndarray, optional
        The exposure in each time bin, with shape (nd, nt) or (nt,).
        The default is the width of time bins.
    p0 : float, optional
        The false positive rate. The default is 0.05.
    iteration : int, optional
        The number of iterations to run to ensure overall false positive rate
        converge to `p0`. The default is 0.
    show_progress : bool, optional
        Whether to show the information of the progress. The default is True.

    Returns
    -------
    BayesianBlocksResult
        The result of the Bayesian Blocks algorithm.
    """
    p0 = float(p0)
    iteration = int(iteration)
    show_progress = bool(show_progress)

    if p0 <= 0.0 or p0 >= 1.0:
        raise ValueError('`p0` must be within (0,1)')

    if iteration < 0:
        raise ValueError('`iteration` must be non-negative')

    if tbins.ndim != 1:
        raise ValueError('`tbins` must be 1D array')

    if counts.ndim not in (1, 2):
        raise ValueError('`counts` must be 1D or 2D array')

    if not np.allclose(counts, np.asarray(counts, dtype=int)):
        raise ValueError('`counts` must be integer')

    counts = np.atleast_2d(counts)

    if tbins.size != counts.shape[1] + 1:
        raise ValueError(
            f'`tbins` ({len(tbins)},) not matched to `counts` {counts.shape}'
        )

    dt = np.diff(tbins)
    if np.any(dt <= 0):
        raise ValueError('`tbins` must be monotonically increasing')

    if exposure is not None:
        if exposure.ndim not in (1, 2):
            raise ValueError('`exposure` must be 1D or 2D array')

        if np.any(exposure <= 0.0):
            raise ValueError('`exposure` must be positive')
    else:
        exposure = np.diff(tbins)

    exposure = np.atleast_2d(exposure)
    if counts.shape != exposure.shape:
        raise ValueError('`counts` and `exposure` must have the same shape')

    if np.any(dt < exposure):
        raise ValueError('`exposure` must be less than the width of each bin')

    zeros = np.zeros(shape=(counts.shape[0], 1), dtype=int)
    n_cumsum = np.hstack((zeros, counts.cumsum(axis=1)))
    n_remainder = np.array(n_cumsum[:, -1:] - n_cumsum, dtype=int, order='C')

    t_cumsum = np.hstack((zeros, exposure.cumsum(axis=1)))
    t_remainder = np.array(t_cumsum[:, -1:] - t_cumsum, dtype=float, order='C')

    data = BayesianBlocksData(
        voronoi=np.ascontiguousarray(tbins, dtype=float),
        n=np.ascontiguousarray(counts, dtype=float),
        t=np.ascontiguousarray(exposure, dtype=float),
        n_remainder=n_remainder,
        t_remainder=t_remainder,
        n_data=int(np.prod(counts.shape)),
    )
    return _bayesian_blocks(data, p0, iteration, show_progress, 'BIN')


class DurationResult(NamedTuple):
    """The result of the duration probability."""

    start: float
    """The start time of the duration."""

    stop: float
    """The stop time of the duration."""

    duration: float
    """The duration between two change points."""

    start_posterior: tuple[NDArray, NDArray]
    """The posterior probability distribution of the start time."""

    stop_posterior: tuple[NDArray, NDArray]
    """The posterior probability distribution of the stop time."""

    duration_posterior: tuple[NDArray, NDArray]
    """The posterior probability distribution of the duration."""

    def ci(self, cl: float | int = 1):
        cl = float(cl)
        assert cl > 0.0
        if cl >= 1.0:
            cl = 1 - 2.0 * norm.sf(cl)

        cl_lower = 0.5 - 0.5 * cl
        cl_upper = 0.5 + 0.5 * cl

        def get_ci(x, prob, map):
            mean = np.sum(prob * x)
            std = np.sqrt(np.sum(prob * np.square(x - mean)))
            lower, median, upper = np.interp(
                x=[cl_lower, 0.5, cl_upper],
                xp=prob.cumsum(),
                fp=x,
                left=x[0],
                right=x[-1],
            )
            return CredibleInterval(
                map=map,
                median=median,
                mean=mean,
                std=std,
                interval=(lower, upper),
                error=Error(
                    map=(lower - map, upper - map),
                    mean=(lower - mean, upper - mean),
                    median=(lower - median, upper - median),
                ),
                cl=cl,
            )

        return DurationCI(
            start=get_ci(*self.start_posterior, self.start),
            stop=get_ci(*self.stop_posterior, self.stop),
            duration=get_ci(*self.duration_posterior, self.duration),
        )


class Error(NamedTuple):
    """The errors corresponding to different point estimates."""

    map: tuple[float, float]
    """The error corresponding to the maximum a posteriori estimate."""

    median: tuple[float, float]
    """The error corresponding to the median of the posterior distribution."""

    mean: tuple[float, float]
    """The error corresponding to the mean of the posterior distribution."""


class CredibleInterval(NamedTuple):
    """The credible interval."""

    map: float
    """The maximum a posteriori estimate."""

    median: float
    """The median of the posterior distribution."""

    mean: float
    """The mean of the posterior distribution."""

    std: float
    """The standard deviation of the posterior distribution."""

    interval: tuple[float, float]
    """The credible interval."""

    error: Error
    """The errors corresponding to different point estimates."""

    cl: float
    """The credible level."""


class DurationCI(NamedTuple):
    """The credible interval of the duration."""

    start: CredibleInterval
    """The credible interval of the start time."""

    stop: CredibleInterval
    """The credible interval of the stop time."""

    duration: CredibleInterval
    """The credible interval of the duration."""


def get_duration_prob(result: BayesianBlocksResult, idx0: int, idx1: int):
    idx0 = int(idx0)
    if idx0 < 0:
        idx0 += len(result.cp)
        if idx0 < 0:
            raise ValueError('`idx0` is out of range')
    if idx0 > len(result.cp):
        raise ValueError('`idx0` is out of range')

    idx1 = int(idx1)
    if idx1 < 0:
        idx1 += len(result.cp)
        if idx1 < 0:
            raise ValueError('`idx1` is out of range')
    if idx1 > len(result.cp):
        raise ValueError('`idx1` is out of range')

    if idx0 > idx1:
        idx0, idx1 = idx1, idx0

    if idx0 == idx1:
        raise ValueError('`idx0` and `idx1` must be different')

    duration = result.edge[idx1] - result.edge[idx0]
    cp0_posterior = result.prob[idx0]
    cp1_posterior = result.prob[idx1]
    x = cp1_posterior.locs[:, None] - cp0_posterior.locs
    prob = cp1_posterior.prob[:, None] * cp0_posterior.prob
    mask = x > 0
    x = x[mask]
    prob = prob[mask]
    prob /= prob.sum()
    argsort = x.argsort()
    return DurationResult(
        float(result.edge[idx0]),
        float(result.edge[idx1]),
        float(duration),
        cp0_posterior,
        cp1_posterior,
        (x[argsort], prob[argsort]),
    )
