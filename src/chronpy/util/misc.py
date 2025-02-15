from __future__ import annotations

import warnings
from threading import Lock
from typing import TYPE_CHECKING

import jax
from jax import lax
from jax.experimental import io_callback
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Callable


def get_parallel_number(n: int | None) -> int:
    """Check and return the available parallel number in JAX.

    Parameters
    ----------
    n : int, optional
        The desired number of parallel processes in JAX.

    Returns
    -------
    int
        The available number of parallel processes.
    """
    n_max = jax.local_device_count()

    if n is None:
        return n_max
    else:
        n = int(n)
        if n <= 0:
            raise ValueError(
                f'number of parallel processes must be positive, got {n}'
            )

    if n > n_max:
        warnings.warn(
            f'number of parallel processes ({n}) is more than the number of '
            f'available devices ({n_max}), reset to {n_max}',
            Warning,
        )
        n = n_max

    return n


def progress_bar_factory(
    neval: int,
    ncores: int,
    init_str: str | None = None,
    run_str: str | None = None,
    update_rate: int = 50,
) -> Callable[[Callable], Callable]:
    """Add a progress bar to JAX ``fori_loop`` kernel, see [1]_ for details.

    Parameters
    ----------
    neval : int
        The total number of evaluations.
    ncores : int
        The number of cores.
    init_str : str, optional
        The string displayed before progress bar when initialization.
    run_str : str, optional
        The string displayed before progress bar when run.
    update_rate : int, optional
        The update rate of the progress bar. The default is 50.

    Returns
    -------
    progress_bar_fori_loop : callable
        Factory that adds a progress bar to function input.

    References
    ----------
    .. [1] `How to add a progress bar to JAX scans and loops
            <https://www.jeremiecoullon.com/2021/01/29/jax_progress_bar/>`_
    """
    neval = int(neval)
    ncores = int(ncores)
    neval_single = neval // ncores

    if neval % ncores != 0:
        raise ValueError('neval must be multiple of ncores')

    if init_str is None:
        init_str = 'Compiling... '
    else:
        init_str = str(init_str)

    if run_str is None:
        run_str = 'Running'
    else:
        run_str = str(run_str)

    if neval > update_rate:
        print_rate = max(1, int(neval_single / update_rate))
    else:
        print_rate = 1

    # lock serializes access to idx_counter since callbacks are multithreaded
    lock = Lock()
    idx_counter = 0  # resource counter
    remainder = neval_single % print_rate
    bar = tqdm(range(neval))
    bar.set_description(init_str, refresh=True)

    def _update_tqdm(increment):
        bar.set_description(run_str, refresh=False)
        bar.update(int(increment))

    def _close_tqdm():
        nonlocal idx_counter

        bar.update(remainder)

        with lock:
            idx_counter += 1

        if idx_counter == ncores:
            bar.close()

    def _update_progress_bar(iter_num):
        _ = lax.cond(
            iter_num == 1,
            lambda _: io_callback(_update_tqdm, None, 0),
            lambda _: None,
            operand=None,
        )
        _ = lax.cond(
            iter_num % print_rate == 0,
            lambda _: io_callback(_update_tqdm, None, print_rate),
            lambda _: None,
            operand=None,
        )
        _ = lax.cond(
            iter_num == neval_single,
            lambda _: io_callback(_close_tqdm, None),
            lambda _: None,
            operand=None,
        )

    def progress_bar_fori_loop(fn):
        """Decorator that adds a progress bar to `body_fun` used in
        `lax.fori_loop`.
        Note that `body_fun` must be looping over a tuple who's first element
        is `np.arange(num_samples)`.
        This means that `iter_num` is the current iteration number
        """

        def _wrapper_progress_bar(i, vals):
            result = fn(i, vals)
            _update_progress_bar(i + 1)
            return result

        return _wrapper_progress_bar

    return progress_bar_fori_loop
