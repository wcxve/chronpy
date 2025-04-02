from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import optimistix as optx
from jax import lax
from numpyro import handlers
from numpyro.distributions import Gamma
from numpyro.distributions.util import validate_sample
from numpyro.infer.util import constrain_fn

from chronpy import __version__
from chronpy.data import LightCurve
from chronpy.psd.models import Component, Model
from chronpy.util.misc import get_parallel_number, progress_bar_factory

if TYPE_CHECKING:
    NDArray = np.ndarray


class PSD(metaclass=ABCMeta):
    def __init__(
        self,
        freq_bins: NDArray,
        power: NDArray,
        dof: NDArray,
        dt: float,
        df: float,
    ):
        self._freq_bins = np.array(freq_bins, dtype=float, order='C', ndmin=1)
        self._power = np.array(power, dtype=float, order='C', ndmin=1)
        self._dof = np.array(dof, dtype=float, order='C', ndmin=1)
        if not (
            self._freq_bins.size - 1 == self._power.size == self._dof.size
        ):
            raise ValueError('freq_bins, power, and dof are not matched')
        if np.any(self._power < 0):
            raise ValueError('power must be non-negative')
        if np.any(self._dof <= 0):
            raise ValueError('dof must be positive')
        if np.any(self._freq_bins < 0):
            raise ValueError('freq_bins must be non-negative')
        if np.any(np.diff(self._freq_bins) <= 0):
            raise ValueError('freq_bins must be sorted in ascending order')
        self._bins_width = np.diff(self._freq_bins)
        self._density = self._power / self._bins_width
        self._freq = 0.5 * (self._freq_bins[:-1] + self._freq_bins[1:])
        self._df = float(df)
        self._dt = float(dt)
        self._perr = None
        self._derr = self.perr / self._bins_width

    @classmethod
    @abstractmethod
    def from_lc(cls, lc: LightCurve | list[LightCurve], norm: str = 'leahy'):
        pass

    @property
    def freq(self) -> NDArray:
        return self._freq

    @property
    def power(self) -> NDArray:
        return self._power

    @property
    @abstractmethod
    def perr(self) -> NDArray:
        pass

    @property
    def density(self) -> NDArray:
        return self._density

    @property
    def derr(self) -> NDArray:
        return self._derr

    @property
    def freq_bins(self) -> NDArray:
        return self._freq_bins

    @property
    def bins_width(self) -> NDArray:
        return self._bins_width

    @property
    def df(self) -> float:
        return self._df

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def dof(self) -> NDArray:
        return self._dof


class Periodogram(PSD):
    @classmethod
    def from_lc(cls, lc: LightCurve | list[LightCurve], norm: str = 'leahy'):
        if isinstance(lc, LightCurve):
            lc = [lc]
        elif not all(isinstance(i, LightCurve) for i in lc):
            raise ValueError('lc must be a LightCurve or a list of LightCurve')

        norm = str(norm).lower()

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
        freq = np.fft.rfftfreq(rate_total.size, dt)[1:]  # exclude 0 frequency
        mod = np.abs(np.fft.rfft(rate_total))[1:]  # exclude 0 frequency
        if norm == 'leahy':
            # this norm makes the mean PSD of wn being equal to 2,
            # the dimension of power is frequency, and the PSD is dimensionless
            power = mod * mod * 2.0 / np.sum(rate / exposure) * df
        else:
            raise ValueError('norm must be "leahy"')

        # exclude the Nyquist frequency, whose dof of power is 1
        if freq[-1] >= 0.5 / dt:
            freq = freq[:-1]
            power = power[:-1]

        freq_bins = np.hstack([freq[0] - 0.5 * df, freq + 0.5 * df])
        return cls(freq_bins, power, np.full(freq.size, 2), dt, df)

    def rebin_log(self, f: float = 0.01) -> PSD:
        bins = self.freq_bins
        idx = [0]
        next_edge = 0.0
        for i in range(1, len(bins)):
            if bins[i] >= next_edge:
                idx.append(i)
                df = bins[idx[-1]] - bins[idx[-2]]
                next_edge = bins[idx[-1]] + df * (1.0 + f)

        if idx[-1] != len(bins) - 1:
            idx[-1] = len(bins) - 1
        freq_bins = np.array([bins[i] for i in idx])
        power = np.add.reduceat(self.power, idx[:-1])
        dof = np.add.reduceat(self.dof, idx[:-1])
        return type(self)(freq_bins, power, dof, self.dt, self.df)

    def rebin_significance(self, s: float = 3.0) -> PSD:
        assert s >= 1, 's must be greater than 1'
        power = self.power
        dof = self.dof

        n = len(power)
        idx = np.empty(n, np.int64)
        idx[0] = 0

        ng = 1
        imax = n - 1
        p_group = 0.0
        dof_group = 0.0
        for i, (pi, vi) in enumerate(zip(power, dof, strict=False)):
            p_group += pi
            dof_group += vi
            x = p_group * (1 - s / np.sqrt(0.5 * dof_group))

            if i == imax:
                if x < 0 and ng > 1:
                    # if the last group is not significant,
                    # then combine the last two groups to ensure all
                    # groups meet the count requirement
                    ng -= 1
                break

            if x >= 0:
                idx[ng] = i + 1
                ng += 1
                p_group = 0.0
                dof_group = 0.0

        idx = idx[:ng]
        dof = np.add.reduceat(dof, idx)
        power = np.add.reduceat(power, idx)
        bins = self.freq_bins
        freq_bins = bins[np.append(idx, len(bins) - 1)]
        return type(self)(freq_bins, power, dof, self.dt, self.df)

    @property
    def perr(self) -> NDArray:
        if self._perr is None:
            self._perr = np.sqrt(0.5 * self._dof) * self._power
        return self._perr


class PowerDist(Gamma):
    """The probability density function of the power spectrum.

    We know that if X ~ Chi_v^2, then cX ~ Gamma(alpha=v/2, scale=c/2).

    For the power spectrum, we know that vI/S ~ Chi_v^2, where I is the
    observed power, v is the dof of I, and S is the true power.
    Then the underlying distribution of I is Gamma(alpha=v/2, scale=2S/v).

    If I is obtained by summing over the power of m adjacent frequency bins,
    I = sum_{m} I_m, then there is no analytical expression for the
    distribution of I, since the Gamma dist's scale of each I_m is different.
    If we assume that the expected powers of I_m are the same, then the
    distribution of I is Gamma(alpha=sum_{m} v_m/2, scale=2S/(sum_{m} v_m)).
    However, this is usually not the case for the observed powers of I_m,
    and this assumption may underestimate the variance of I, thus leading to
    a deviance that is systematically larger than the fit dof, i.e.,
    number of data points minus number of parameters.

    Averaging the powers of l different power spectrum is valid, and the
    distribution of the average power I_j is
    Gamma(alpha=l*v_j/2, scale=2S/(l*v_j)).
    """

    def __init__(self, dof, s):
        half_dof = 0.5 * dof
        super().__init__(concentration=half_dof, rate=half_dof / s)

    @validate_sample
    def log_prob(self, value):
        n = self.concentration
        rate = self.rate
        gof = n * (jnp.log(n / value) - 1.0)
        return n * jnp.log(rate) - rate * value - gof


class Fit:
    def __init__(self, psd: PSD, model: Model):
        if not isinstance(psd, PSD):
            raise ValueError('psd must be an instance of PSD')
        if isinstance(model, Component):
            model = Model(model)
        elif not isinstance(model, Model):
            raise ValueError('model must be an instance of Model or Component')
        self.psd = psd
        self.model = model
        self._loss = None
        self._transform = None
        self._ndata = len(psd.freq)
        self._numpyro_model = None

    @property
    def ndata(self):
        return self._ndata

    @property
    def nparam(self):
        return len(self.params_names)

    @property
    def numpyro_model(self):
        if self._numpyro_model is None:

            def _():
                params = {
                    p: numpyro.sample(p, d)
                    for p, d in self.model.prior.items()
                }
                power_model = jax.jit(self.model.power)(
                    params, self.psd.freq_bins
                )
                numpyro.deterministic('S', power_model)
                pdist = PowerDist(self.psd.dof, power_model)
                I_data = numpyro.primitives.mutable('I_data', self.psd.power)
                with numpyro.plate('freq', len(self.psd.freq)):
                    numpyro.sample(name='I_obs', fn=pdist, obs=I_data)
                numpyro.deterministic(
                    name='loglike', value=pdist.log_prob(I_data)
                )

            self._numpyro_model = _
        return self._numpyro_model

    def run_nuts(
        self,
        num_warmup=2000,
        num_samples=2000,
        num_chains=4,
        init=None,
        chain_method='parallel',
        progress=True,
        seed=42,
    ):
        if init is not None:
            init = self.model.default | dict(init)
        else:
            init = self.model.default
        mcmc = numpyro.infer.MCMC(
            numpyro.infer.NUTS(
                model=self.numpyro_model,
                init_strategy=numpyro.infer.util.init_to_value(values=init),
            ),
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            chain_method=chain_method,
            progress_bar=progress,
        )
        mcmc.run(
            jax.random.PRNGKey(seed), extra_fields=('energy', 'num_steps')
        )
        samples = mcmc.get_samples(group_by_chain=True)

        # stats of samples
        rename = {'num_steps': 'n_steps'}
        sample_stats = {}
        for k, v in mcmc.get_extra_fields(group_by_chain=True).items():
            name = rename.get(k, k)
            value = jax.device_get(v).copy()
            sample_stats[name] = value
            if k == 'num_steps':
                sample_stats['tree_depth'] = np.log2(value).astype(int) + 1

        # attrs for each group of arviz.InferenceData
        attrs = {
            'chronpy_version': __version__,
            'inference_library': 'numpyro',
            'inference_library_version': numpyro.__version__,
        }

        return self._generate_idata(samples, attrs, sample_stats)

    def _generate_idata(self, samples, attrs, sample_stats=None):
        samples = jax.tree.map(jax.device_get, samples)
        params = {k: v for k, v in samples.items() if k in self.params_names}
        posterior = {k: v for k, v in samples.items() if k != 'loglike'}
        posterior_predictive = self.simulate(params)
        posterior_predictive = {
            'I_obs': posterior_predictive,
            'total': posterior_predictive.sum(-1),
        }
        loglike = {
            'I_obs': samples['loglike'],
            'total': samples['loglike'].sum(-1),
        }
        obs_data = {'I_obs': self.psd.power, 'total': self.psd.power.sum()}

        # coords and dims of arviz.InferenceData
        coords = {'freq': self.psd.freq}

        dims = {'S': ['freq'], 'I_obs': ['freq']}

        # create InferenceData
        return az.from_dict(
            posterior=posterior,
            posterior_predictive=posterior_predictive,
            sample_stats=sample_stats,
            log_likelihood=loglike,
            observed_data=obs_data,
            coords=coords,
            dims=dims,
            posterior_attrs=attrs,
            posterior_predictive_attrs=attrs,
            sample_stats_attrs=attrs,
            log_likelihood_attrs=attrs,
            observed_data_attrs=attrs,
        )

    @property
    def loss(self):
        if self._loss is None:
            params_names = self.params_names

            def get_loglike(unconstr_arr):
                sites = constrain_fn(
                    model=self.numpyro_model,
                    model_args=(),
                    model_kwargs={},
                    params=dict(zip(params_names, unconstr_arr, strict=False)),
                    return_deterministic=True,
                )
                return sites['loglike']

            residual = jax.jit(lambda x: jnp.sqrt(-2.0 * get_loglike(x)))
            deviance = jax.jit(lambda x: jnp.sum(-2.0 * get_loglike(x)))

            self._loss = {'residual': residual, 'deviance': deviance}

        return self._loss

    @property
    def params_names(self):
        return list(self.model.prior.keys())

    @property
    def transform(self):
        from numpyro.distributions.transforms import biject_to

        if self._transform is None:
            self._transform = {
                k: biject_to(v.support) for k, v in self.model.prior.items()
            }
        return self._transform

    def mle_lm(self, init=None):
        if init is not None:
            init = self.model.default | dict(init)
        else:
            init = self.model.default
        lm_solver = optx.LevenbergMarquardt(rtol=0.0, atol=1e-8)
        residual = jax.jit(lambda x, aux: self.loss['residual'](x))

        def lm(init):
            res = optx.least_squares(
                fn=residual,
                solver=lm_solver,
                y0=init,
                max_steps=4096,
                throw=True,
            )
            grad_norm = jnp.linalg.norm(res.state.f_info.compute_grad())
            deviance = jnp.square(res.state.f_info.residual).sum()
            return res.value, deviance, grad_norm

        _lm = jax.jit(lm)
        t = self.transform
        popt, f, g = _lm(
            jnp.array([t[k].inv(init[k]) for k in self.params_names])
        )
        params_mle = {
            k: t[k](v)
            for k, v in dict(
                zip(self.params_names, popt, strict=False)
            ).items()
        }
        cov = self._calc_cov(params_mle)
        err = {k: np.sqrt(cov[i, i]) for i, k in enumerate(self.params_names)}
        return (
            {k: (float(params_mle[k]), err[k]) for k in self.params_names},
            f,
            g,
        )

    def mle(self, init=None):
        if init is not None:
            init = self.model.default | dict(init)
        else:
            init = self.model.default

        @jax.jit
        def deviance(x, _):
            return self.loss['deviance'](x)

        bfgs = optx.BFGS(rtol=0.0, atol=1e-8)

        def mle(init):
            res = optx.minimise(
                fn=deviance,
                solver=bfgs,
                y0=init,
                max_steps=4096,
                throw=True,
            )
            grad_norm = jnp.linalg.norm(res.state.f_info.grad)
            return res.value, res.state.f_info.f, grad_norm

        _mle = jax.jit(mle)
        t = self.transform
        popt, f, g = _mle(
            jnp.array([t[k].inv(init[k]) for k in self.params_names])
        )
        params_mle = {
            k: t[k](v)
            for k, v in dict(
                zip(self.params_names, popt, strict=False)
            ).items()
        }
        cov = self._calc_cov(params_mle)
        err = {k: np.sqrt(cov[i, i]) for i, k in enumerate(self.params_names)}
        return (
            {k: (float(params_mle[k]), err[k]) for k in self.params_names},
            f,
            g,
        )

    def _calc_cov(self, params):
        @jax.jit
        @jax.hessian
        def hess(params):
            params = jnp.array(
                [t[k].inv(params[n]) for n, k in enumerate(self.params_names)]
            )
            return self.loss['deviance'](params)

        t = self.transform
        params = np.array([params[k] for k in self.params_names])
        cov = 2.0 * jnp.linalg.inv(hess(params))
        return cov

    def simulate(self, params, n=1, seed=42):
        n = int(n)
        params = dict(params)
        params = np.array([params[k] for k in self.params_names], float)
        if params.ndim == 2 and n != 1:
            raise ValueError('params must be 1D if n > 1')
        power_fn = jax.jit(self.model.power)
        for _ in range(params.ndim - 1):
            power_fn = jax.jit(jax.vmap(power_fn, in_axes=(0, None)))
        params = dict(zip(self.params_names, params, strict=False))
        rng = np.random.default_rng(seed)
        power = power_fn(params, self.psd.freq_bins)
        dof = self.psd.dof
        sample_shape = (n,) + power.shape if n != 1 else power.shape
        sim_data = power * rng.chisquare(dof, size=sample_shape) / dof
        return sim_data

    def batch_fit(
        self,
        data,
        init=None,
        parallel: bool = True,
        n_parallel: int | None = None,
        progress: bool = True,
        update_rate: int = 50,
        run_str: str = 'Fitting',
        seed=42,
    ):
        rng_key = jax.random.PRNGKey(seed)
        if init is None:
            rng_key = jax.random.split(rng_key, num=self.nparam)
            rng_key = dict(zip(self.params_names, rng_key, strict=False))
            init = {
                k: v.sample(rng_key[k], (len(data),))
                for k, v in self.model.prior.items()
            }
        else:
            init = dict(init)
            assert set(self.params_names).issubset(init), (
                'init must contain all params'
            )
        return init_bacth_fit(self)(
            init, data, parallel, n_parallel, progress, update_rate, run_str
        )


def init_bacth_fit(fit):
    lm_solver = optx.LevenbergMarquardt(rtol=0.0, atol=1e-6)
    numpyro_model = fit.numpyro_model
    params_names = fit.params_names
    transform = fit.transform

    def get_sites_(unconstr_arr):
        sites = constrain_fn(
            model=numpyro_model,
            model_args=(),
            model_kwargs={},
            params=dict(zip(params_names, unconstr_arr, strict=False)),
            return_deterministic=True,
        )
        params = {k: sites[k] for k in params_names}
        models = sites['S']
        loglike = sites['loglike']
        return {'params': params, 'models': models, 'loglike': loglike}

    @jax.jit
    def fit_once(i: int, args: tuple) -> tuple:
        """Loop core, fit simulation data once."""
        result, init = args

        # substitute observation data with simulation data
        new_data = {'I_data': result['data'][i]}
        get_sites = jax.jit(handlers.substitute(fn=get_sites_, data=new_data))
        residual = lambda p: -2.0 * get_sites(p)['loglike']

        # fit simulation data
        res = optx.least_squares(
            fn=lambda p, _: residual(p),
            solver=lm_solver,
            y0=init[i],
            max_steps=1024,
            throw=False,
        )
        fitted_params = res.value
        grad_norm = jnp.linalg.norm(res.state.f_info.compute_grad())
        sites = get_sites(fitted_params)

        # update best fit params to result
        result['params'] = jax.tree.map(
            lambda x, y: x.at[i].set(y),
            result['params'],
            sites['params'],
        )

        # update the best fit model to result
        result['models'] = result['models'].at[i].set(sites['models'])

        # update the deviance information to result
        dev = {
            'total': -2.0 * sites['loglike'].sum(),
            'point': -2.0 * sites['loglike'],
        }
        res_dev = result['deviance']
        res_dev['total'] = res_dev['total'].at[i].set(dev['total'])
        res_dev['point'] = res_dev['point'].at[i].set(dev['point'])

        valid = jnp.bitwise_not(
            jnp.isnan(dev['total'])
            | jnp.isnan(grad_norm)
            | jnp.greater(grad_norm, 1e-3)
        )
        result['valid'] = result['valid'].at[i].set(valid)

        return result, init

    def sequence_fit(
        result: dict,
        init: NDArray,
        run_str: str,
        progress: bool,
        update_rate: int,
    ):
        """Fit simulation data in sequence."""
        n = len(result['valid'])

        if progress:
            pbar_factory = progress_bar_factory(
                n, 1, run_str=run_str, update_rate=update_rate
            )
            fn = pbar_factory(fit_once)
        else:
            fn = fit_once

        fit_jit = jax.jit(lambda *args: lax.fori_loop(0, n, fn, args)[0])
        result = fit_jit(result, init)
        return result

    def parallel_fit(
        result: dict,
        init: NDArray,
        run_str: str,
        progress: bool,
        update_rate: int,
        n_parallel: int,
    ) -> dict:
        """Fit simulation data in parallel."""
        n = len(result['valid'])
        n_parallel = int(n_parallel)
        batch = n // n_parallel

        if progress:
            pbar_factory = progress_bar_factory(
                n, n_parallel, run_str=run_str, update_rate=update_rate
            )
            fn = pbar_factory(fit_once)
        else:
            fn = fit_once

        fit_pmap = jax.pmap(lambda *args: lax.fori_loop(0, batch, fn, args)[0])
        reshape = lambda x: x.reshape((n_parallel, -1) + x.shape[1:])
        result = fit_pmap(
            jax.tree.map(reshape, result),
            reshape(init),
        )

        return jax.tree.map(jnp.concatenate, result)

    def run(
        init_params: dict,
        data: NDArray,
        parallel: bool = True,
        n_parallel: int | None = None,
        progress: bool = True,
        update_rate: int = 50,
        run_str: str = 'Fitting',
    ) -> dict:
        """Simulate data and then fit the simulation data.

        Parameters
        ----------
        init_params : dict
            The initial parameters values in unconstrained space.
        data : dict
            The model values corresponding to `free_params`.
        parallel : bool, optional
            Whether to fit in parallel, by default True.
        n_parallel : int, optional
            The number of parallel processes when `parallel` is ``True``.
            Defaults to ``jax.local_device_count()``.
        progress : bool, optional
            Whether to show progress bar, by default True.
        update_rate : int, optional
            The update rate of the progress bar, by default 50.
        run_str : str, optional
            The string to ahead progress bar during the run when `progress` is
            True. The default is 'Fitting'.

        Returns
        -------
        result : dict
            The simulation and fitting result.
        """
        init_params = jax.tree.map(jnp.array, init_params)
        n = len(data)
        n_parallel = get_parallel_number(n_parallel)
        if n % n_parallel != 0:
            raise ValueError(
                f'n ({n}) must be a multiple of n_parallel ({n_parallel})'
            )

        assert set(init_params) == set(params_names)
        assert n > 0

        # check if all params shapes are the same
        shapes = list(jax.tree.map(jnp.shape, init_params).values())
        assert all(i == shapes[0] for i in shapes)

        # get initial parameters arrays in unconstrained space,
        t = transform
        init = jnp.array([t[k].inv(init_params[k]) for k in params_names]).T
        assert init.ndim <= 2
        if init.ndim == 2:
            assert init.shape[0] == n

        if init.ndim == 1:
            init = jnp.full((n, len(init)), init)

        # fit result container
        result = {
            'data': data,
            'params': {k: jnp.empty(n) for k in params_names},
            'models': jnp.empty((n, fit.ndata)),
            'deviance': {
                'total': jnp.empty(n),
                'point': jnp.empty((n, fit.ndata)),
            },
            'valid': jnp.full(n, True, bool),
        }

        # fit simulation data
        if parallel:
            res = parallel_fit(
                result,
                init,
                run_str,
                progress,
                update_rate,
                n_parallel,
            )
        else:
            res = sequence_fit(result, init, run_str, progress, update_rate)
        return res

    return run
