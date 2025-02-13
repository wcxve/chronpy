import jax
import jax.numpy as jnp
import pytest

from chronpy.psd.models import (
    PL,
    BentPL,
    BrokenPL,
    Const,
    Lorentz,
    Model,
    SmoothlyBrokenPL,
)


@pytest.mark.parametrize(
    'model, params',
    [
        pytest.param(
            PL,
            {
                'PL.A': 2.0,
                'PL.alpha': 2.0,
            },
            id='PL',
        ),
        pytest.param(
            BrokenPL,
            {
                'BrokenPL.A': 10.0,
                'BrokenPL.alpha1': 1.5,
                'BrokenPL.fb': 1.0,
                'BrokenPL.alpha2': 2.0,
            },
            id='BrokenPL',
        ),
        pytest.param(
            Lorentz,
            {
                'Lorentz.A': 10.0,
                'Lorentz.f': 1.0,
                'Lorentz.fwhm': 1e-1,
            },
            id='Lorentz',
        ),
        pytest.param(
            Const,
            {'Const.C': 2.0},
            id='Const',
        ),
        pytest.param(
            BentPL,
            {
                'BentPL.A': 10.0,
                'BentPL.alpha': 1.5,
                'BentPL.fb': 2.0,
            },
            id='BentPL',
        ),
        pytest.param(
            SmoothlyBrokenPL,
            {
                'SmoothlyBrokenPL.A': 10.0,
                'SmoothlyBrokenPL.fb': 2.0,
                'SmoothlyBrokenPL.alpha1': 0.5,
                'SmoothlyBrokenPL.alpha2': 2.0,
            },
            id='SmoothlyBrokenPL',
        ),
    ],
)
def test_model(model, params):
    f = jnp.geomspace(1e-3, 1e3, 1001)
    p = jax.jit(Model(model()).power)(params, f)
    assert p.shape == (1000,)
    assert jnp.all(jnp.isfinite(p))
    assert jnp.all(p >= 0.0)
