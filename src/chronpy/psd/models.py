from __future__ import annotations

from abc import ABCMeta, abstractmethod
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import nnx
from numpyro.distributions import LogUniform, Uniform

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray as NDArray

jax.config.update('jax_enable_x64', True)


def build_namespace(names: list[str]) -> dict[str, list[str]]:
    """Build a namespace from a sequence of names.

    Parameters
    ----------
    names : sequence of str
        A sequence of names.

    Returns
    -------
    namespace: dict
        A dict of non-duplicate names and suffixes in original name order.
    """
    namespace = []
    names_ = []
    suffixes_n = []
    counter = {}

    for name in names:
        names_.append(name)

        if name not in namespace:
            counter[name] = 1
            namespace.append(name)
        else:
            counter[name] += 1
            namespace.append(f'{name}#{counter[name]}')

        suffixes_n.append(counter[name])

    suffixes = [f'_{{{n}}}' if n > 1 else '' for n in suffixes_n]

    return {
        'namespace': list(map(''.join, zip(names_, suffixes, strict=False))),
        'suffix_num': [str(n) if n > 1 else '' for n in suffixes_n],
    }


def evaluate_stack(stack, result):
    if isinstance(stack, str):
        return result[stack]
    operator, (lhs, rhs) = stack
    lhs_value = evaluate_stack(lhs, result)
    rhs_value = evaluate_stack(rhs, result)
    return operator(lhs_value, rhs_value)


def extract_stack(component, names):
    if isinstance(component, CompositeComponent):
        op = component.op
        lhs = extract_stack(component.lhs, names)
        rhs = extract_stack(component.rhs, names)
        return op, (lhs, rhs)
    else:
        return names[component]


class Model(nnx.Module, metaclass=ABCMeta):
    def __init__(self, component: Component):
        if not isinstance(component, Component):
            raise ValueError('component must be an instance of Component')

        comps = [component]
        unique_comps = []
        while comps:
            comp = comps.pop(0)
            if isinstance(comp, CompositeComponent):
                comps.extend([comp.lhs, comp.rhs])
            elif comp not in unique_comps:
                unique_comps.append(comp)
        unique_comps_names = [type(comp).__name__ for comp in unique_comps]
        self._comps = dict(
            zip(
                build_namespace(unique_comps_names)['namespace'],
                unique_comps,
                strict=False,
            )
        )
        names = dict(
            zip(self._comps.values(), self._comps.keys(), strict=False)
        )
        self._stacks = extract_stack(component, names)

    def _power(self, freq_bins):
        result = {
            names: comp.power(freq_bins) for names, comp in self._comps.items()
        }
        return evaluate_stack(self._stacks, result)

    @partial(jax.jit, static_argnums=0)
    def power(self, params, freq_bins):
        graph_def, state = nnx.split(self)
        params_state = state.to_pure_dict()['_comps']

        for k, v in params.items():
            comp, param = k.split('.')
            params_state[comp]['_params'][param] = v

        state.replace_by_pure_dict({'_comps': params_state})
        return nnx.call((graph_def, state))._power(freq_bins)[0]

    @property
    def prior(self):
        return {
            f'{c}.{p}': d
            for c, v in self._comps.items()
            for p, d in v.prior.items()
        }

    @property
    def default(self):
        return {
            f'{c}.{p}': v
            for c, v in self._comps.items()
            for p, v in v.params.items()
        }


class Component(nnx.Module, metaclass=ABCMeta):
    _config: tuple[tuple[str, float, float, float, bool], ...]

    def __init__(self):
        params = {}
        for name, default, *_ in self._config:
            params[name] = nnx.Param(float(default))
        self._params = params

    @partial(jax.jit, static_argnums=0)
    def power(self, freq_bins):
        return self.integral(freq_bins)

    @abstractmethod
    def integral(self, freq_bins):
        pass

    @property
    def params(self):
        return {k: v.value for k, v in self._params.items()}

    @property
    def prior(self):
        # prior should be initialized when creating nnx.Param()
        return {
            c[0]: LogUniform(c[2], c[3]) if c[4] else Uniform(c[2], c[3])
            for c in self._config
        }

    def __add__(self, other):
        if not isinstance(other, Component):
            raise NotImplementedError(
                'unsupported operand type(s) for +: Component and '
                + type(other).__name__
            )
        return CompositeComponent(self, other, jnp.add)

    def __sub__(self, other):
        if not isinstance(other, Component):
            raise NotImplementedError(
                'unsupported operand type(s) for -: Component and '
                + type(other).__name__
            )
        return CompositeComponent(self, other, jnp.subtract)


class CompositeComponent(Component):
    def __init__(self, lhs: Component, rhs: Component, op: Callable):
        self.lhs = lhs
        self.rhs = rhs
        self.op = op
        self._config = ()
        super().__init__()

    def integral(self, freq_bins):
        return self.op(
            self.lhs.integral(freq_bins),
            self.rhs.integral(freq_bins),
        )


class NumIntComponent(Component):
    """Base class for components that require numerical integration.

    Parameters
    ----------
    integrator : str, optional
        The numerical integrator to use. Either 'trapz' or 'simpson'.
    nsub : int, optional
        The number of subintervals to use for numerical integration.
        The default is 1 for 'trapz' and 2 for 'simpson'.
    """

    def __init__(self, integrator='simpson', nsub=None):
        if integrator not in ('trapz', 'simpson'):
            raise ValueError('integrator must be either "trapz" or "simpson"')
        self._integrator = integrator
        if nsub is None:
            nsub = 1 if integrator == 'trapz' else 2
        if nsub < 1:
            raise ValueError('nsub must be greater than or equal to 1')
        if integrator == 'simpson':
            if nsub < 2:
                raise ValueError(
                    'nsub must be greater than or equal to 2 for '
                    "Simpson's rule"
                )
            if nsub % 2:
                raise ValueError("nsub must be even for Simpson's rule")
        self._n = nsub
        super().__init__()

    @partial(jax.jit, static_argnums=0)
    def integral(self, freq_bins: NDArray):
        if self._integrator == 'trapz':
            n_point = self._n + 1
            x = jnp.linspace(freq_bins[:-1], freq_bins[1:], n_point, axis=1)
            y = self.continuum(x)
            return jax.scipy.integrate.trapezoid(y, x)
        else:
            n_point = self._n // 2 + 1
            x = jnp.linspace(freq_bins[:-1], freq_bins[1:], n_point, axis=1)
            return self._simpson(x[:, :-1], x[:, 1:]).sum(axis=1)

    def _simpson(self, freq_low, freq_high):
        freq_mid = 0.5 * (freq_low + freq_high)
        f_low = self.continuum(freq_low)
        f_mid = self.continuum(freq_mid)
        f_high = self.continuum(freq_high)
        return (freq_high - freq_low) / 6.0 * (f_low + 4.0 * f_mid + f_high)

    @abstractmethod
    def continuum(self, freq):
        pass


class Const(Component):
    _config = (('C', 1.8, 1e-2, 4, False),)

    def integral(self, freq_bins):
        return self.params['C'] * (freq_bins[1:] - freq_bins[:-1])


def _powerlaw_integral(freq_bins, alpha):
    cond = jnp.full(len(freq_bins), jnp.not_equal(alpha, 1.0))
    one_minus_alpha = jnp.where(cond, 1.0 - alpha, 1.0)
    f1 = jnp.power(freq_bins, one_minus_alpha) / one_minus_alpha
    f1 = f1[1:] - f1[:-1]
    f2 = jnp.log(freq_bins)
    f2 = f2[1:] - f2[:-1]
    return jnp.where(cond[:-1], f1, f2)


class PL(Component):
    _config = (
        ('alpha', 1.5, -5.0, 5.0, False),
        ('A', 10.0, 1e-6, 1e6, False),
    )

    def integral(self, freq_bins):
        alpha = self.params['alpha']
        A = self.params['A']
        return A * _powerlaw_integral(freq_bins, alpha)


class BrokenPL(Component):
    _config = (
        ('alpha1', 0.5, -5.0, 5.0, False),
        ('alpha2', 2.0, -5.0, 5.0, False),
        ('fb', 2.0, 1e-2, 1e2, False),
        ('A', 10.0, 1e-6, 1e6, False),
    )

    def integral(self, freq_bins):
        alpha1 = self.params['alpha1']
        fb = self.params['fb']
        alpha2 = self.params['alpha2']
        A = self.params['A']
        mask = freq_bins[:-1] <= fb
        freq_bins_ = freq_bins / fb
        p1 = _powerlaw_integral(freq_bins_, alpha1)
        p2 = _powerlaw_integral(freq_bins_, alpha2)
        p = jnp.where(mask, p1, p2)
        idx = jnp.flatnonzero(mask, size=freq_bins.size - 1)[-1]
        pb1 = _powerlaw_integral(jnp.hstack([freq_bins_[idx], fb]), alpha1)
        pb2 = _powerlaw_integral(jnp.hstack([fb, freq_bins_[idx + 1]]), alpha2)
        p.at[idx].set(sum(pb1 + pb2))
        return A * p


class BentPL(NumIntComponent):
    _config = (
        ('alpha', 1.5, -5.0, 5.0, False),
        ('fb', 2.0, 1e-2, 1e2, False),
        ('A', 10.0, 1e-6, 1e6, False),
    )

    def continuum(self, freq):
        alpha = self.params['alpha']
        fb = self.params['fb']
        A = self.params['A']
        return A * 2.0 / (1.0 + jnp.power(freq / fb, alpha))


class SmoothlyBrokenPL(NumIntComponent):
    _config = (
        ('A', 10.0, 1e-6, 1e6, False),
        ('fb', 2.0, 1e-2, 1e2, False),
        ('alpha1', 0.5, -5.0, 5.0, False),
        ('alpha2', 2.0, -5.0, 5.0, False),
        # ('rho', 1.0, 1e-2, 1e2, False),
    )

    def continuum(self, freq):
        A = self.params['A']
        fb = self.params['fb']
        alpha1 = self.params['alpha1']
        alpha2 = self.params['alpha2']
        rho = 1.0  # self.params['rho']

        f = freq / fb
        x = (alpha2 - alpha1) / rho * jnp.log(f)

        threshold = 30
        alpha = jnp.where(x > threshold, alpha2, alpha1)
        r = jnp.where(jnp.abs(x) > threshold, 0.5, 0.5 * (1.0 + jnp.exp(x)))

        return A * jnp.power(f, -alpha) * jnp.power(r, -rho)


class Lorentz(Component):
    _config = (
        ('f', 1.0, 1e-6, 1e6, False),
        ('fwhm', 1e-3, 1e-3, 1e3, False),
        ('A', 10.0, 1e-6, 1e6, False),
    )

    def integral(self, freq_bins):
        f = self.params['f']
        fwhm = self.params['fwhm']
        A = self.params['A']
        gamma2 = jnp.square(0.5 * fwhm)
        integral = gamma2 * jnp.arctan((freq_bins - f) / gamma2)
        return A * (integral[1:] - integral[:-1])
