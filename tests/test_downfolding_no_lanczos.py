"""The removal and attachment Green's functions of a CI state by block
Lanczos (`solvers/dynamic_backends.py`): the kernel `block_lanczos`, which
sees a product and the start vectors alone, and `removal_spectrum` and
`attachment_spectrum`, which start it from [a_{u sigma}|K>]_u (the
(N-1)-electron sector) or [a+_{u sigma}|K>]_u (the (N+1)-electron sector)
of the engine's sector operations. G_tu(z) = sum_m w[t, m] w[u, m] /
(z - e_m).

Water/6-31G RHF on the NO CAS(4,4) of Task 8, its singlet ground state K:
at the full depth (numerics.downfolding.delta_lanczos_depth set to the
sector dimension) both sides are the dense resolvent of the sector built
from the amplitudes <M|a_u|K> and <M|a+_u|K> of the dense Delta methods
(`EnergyDependentSolver.sector`); at depth two the poles and weights
reproduce the moments <a_t K|H^n|a_u K>, n = 0..3, of the dense sector
matrix; alpha and beta give one G for the singlet; an orbital that is
exactly empty in K (its integrals with every other orbital zeroed) has a
start vector of norm 1e-32, which gets zero weights, without NaN, while
the other orbitals keep the dense resolvent. The kernel on a random
symmetric matrix is its dense resolvent. The two settings are validated,
an engine without sector operations is refused, and a sector that does
not exist gives no poles."""
import math

import numpy as np
import pytest

from src.Base.numerics.numerics import Numerics
from src.EffectiveHamiltonian.Downfolding import (build_effective_hamiltonian,
                                                  run_low_level)
from src.EffectiveHamiltonian.solvers.backends import (
    ExactDiagonalizationSolver, PyscfFCISolver)
from src.EffectiveHamiltonian.solvers.dynamic_backends import (
    LowdinSolver, attachment_spectrum, block_lanczos, removal_spectrum)
from src.EffectiveHamiltonian.solvers.solve import SolveRequest, static_states
from tests.no_downfolding import no_space, water

#: a depth above every sector dimension of the CAS(4,4) (24 for one
#: electron less or more): the Krylov space is exhausted first
FULL = Numerics(downfolding={'delta_lanczos_depth': 100})


@pytest.fixture(scope='module')
def state():
    """The water NO CAS(4,4) of Task 8 and its singlet ground state K:
    (h1, eri, engine, K), h1 and eri as the static solve diagonalized
    them."""
    low = run_low_level(water('hf'), method='rpa', mode='casida',
                        natural_orbitals=True)
    h = build_effective_hamiltonian(low, no_space(low, 2, 2),
                                    screening='crpa')
    st = static_states('pyscf_fci', SolveRequest(
        h.g_eff, h.space['n_orbs'], h.space['n_elec_active'],
        spin_free=True, nroots=1, spin=0))
    assert st.sector == (2, 2) and st.norb == 4
    return st.h1, st.eri, st.engine, st.vectors[0]


def _green(poles, weights, z):
    """G_tu(z) = sum_m weights[t, m] weights[u, m] / (z - poles[m])."""
    return (weights / (z - poles)) @ weights.T


def _dense_green(engine, h1, eri, k, kind, sigma, z_values):
    """[G(z)] of the side `kind` from the sector eigenvectors: the
    amplitudes <M|a_u|K> (particle) or <M|a+_u|K> (hole) of every orbital,
    as the dense Delta methods build them (`EnergyDependentSolver.sector`,
    every row of b coupling), and the spectrum e_M of the sector."""
    target = [2, 2]
    target[sigma] += -1 if kind == 'particle' else 1
    solver = LowdinSolver(engine, [])
    _, e_m, amp, _, _ = solver.sector(h1, eri, 4, (2, 2), kind, sigma,
                                      tuple(target), np.zeros(4), np.eye(4))
    a = np.array([amp(u) @ k.ravel() for u in range(4)])     # (4, dim M)
    return [_green(e_m, a, z) for z in z_values], e_m


def _z_values(e_m):
    """Three z: below the spectrum, inside it off the real axis, above it."""
    return (e_m[0] - 0.7, 0.5 * (e_m[0] + e_m[-1]) + 0.3j, e_m[-1] + 0.9)


def _full_depth_case(state, spectrum, kind):
    h1, eri, engine, k = state
    worst = 0.0
    for sigma in (0, 1):
        poles, weights = spectrum(engine, k, h1, eri, 4, (2, 2), range(4),
                                  sigma, FULL)
        _, e_m = _dense_green(engine, h1, eri, k, kind, sigma, [0.0])
        # the Krylov space is exhausted before the depth of 100
        assert weights.shape == (4, len(poles)) and len(poles) <= len(e_m)
        z_values = _z_values(e_m)
        want, _ = _dense_green(engine, h1, eri, k, kind, sigma, z_values)
        for z, g in zip(z_values, want):
            worst = max(worst, np.abs(_green(poles, weights, z) - g).max())
    return worst


def test_full_depth_is_the_dense_resolvent(state):
    worst = _full_depth_case(state, removal_spectrum, 'particle')
    print(f'removal G at full depth against the dense resolvent: {worst:.1e}')
    assert worst < 1e-10


def test_full_depth_attachment_is_the_dense_resolvent(state):
    worst = _full_depth_case(state, attachment_spectrum, 'hole')
    print(f'attachment G at full depth against the dense resolvent: '
          f'{worst:.1e}')
    assert worst < 1e-10


def test_kernel_on_a_plain_matrix():
    rng = np.random.default_rng(7)
    n, p = 30, 3
    a = rng.normal(size=(n, n))
    a = 0.5 * (a + a.T)
    start = rng.normal(size=(n, p))
    e, v = np.linalg.eigh(a)
    # also with a fourth start vector that is a combination of two others,
    # a dropped column whose weights are those of the combination
    dependent = np.column_stack([start, start[:, 0] - 2.0 * start[:, 2]])
    worst = 0.0
    for s in (start, dependent):
        poles, weights = block_lanczos(lambda x: a @ x, s, FULL)
        assert weights.shape == (s.shape[1], len(poles)) and len(poles) <= n
        for z in (e[0] - 1.0, 0.1 + 0.4j, e[-1] + 2.0):
            want = s.T @ v @ np.diag(1.0 / (z - e)) @ v.T @ s
            worst = max(worst, np.abs(_green(poles, weights, z) - want).max())
    print(f'kernel at full depth against the dense resolvent of a random '
          f'symmetric matrix: {worst:.1e}')
    assert worst < 1e-10


def test_depth_two_reproduces_four_moments(state):
    h1, eri, engine, k = state
    two = Numerics(downfolding={'delta_lanczos_depth': 2})
    worst = 0.0
    for sigma, sector in ((0, (1, 2)), (1, (2, 1))):
        poles, weights = removal_spectrum(engine, k, h1, eri, 4, (2, 2),
                                          range(4), sigma, two)
        assert len(poles) == 8                          # two blocks of four
        h = engine.sector_matrix(h1, eri, 4, sector)
        s = np.column_stack([engine.annihilate(k, 4, (2, 2), u, sigma).ravel()
                             for u in range(4)])
        hn_s = s
        for n in range(5):
            got = (weights * poles ** n) @ weights.T
            want = s.T @ hn_s
            error = np.abs(got - want).max()
            print(f'sigma {sigma}, n = {n}: largest entry '
                  f'{np.abs(want).max():.2f}, error {error:.1e}')
            if n < 4:
                worst = max(worst, error)
            else:
                # the fifth moment is beyond two blocks: the depth is real
                assert error > 1e-6
            hn_s = h @ hn_s
    print(f'moments n = 0..3 at depth 2 against <a_t K|H^n|a_u K>: '
          f'{worst:.1e}')
    assert worst < 1e-9


def test_rank_deficient_start_block(state):
    """Orbital 3 made exactly empty in K: its integrals with every other
    orbital zeroed and its own level raised to 5 Ha, so H conserves its
    occupation and the ground state leaves it empty."""
    h1, eri, engine, _ = state
    h1, eri = h1.copy(), eri.copy()
    e = 3
    h1[e, :] = h1[:, e] = 0.0
    h1[e, e] = 5.0
    eri[e], eri[:, e], eri[:, :, e], eri[:, :, :, e] = 0.0, 0.0, 0.0, 0.0
    _, vectors = engine.roots(h1, eri, 4, (2, 2), nroots=1, spin=0)
    k = vectors[0]
    dm1, _ = engine.rdms(k, 4, (2, 2), spin=0)
    norm = np.linalg.norm(engine.annihilate(k, 4, (2, 2), e, 0))
    print(f'occupation of orbital {e}: {dm1[e, e]:.1e}, |a_{e} K| = '
          f'{norm:.1e}')
    assert dm1[e, e] < 1e-12 and dm1[2, 2] > 1e-4
    for numerics in (None, FULL):
        poles, weights = removal_spectrum(engine, k, h1, eri, 4, (2, 2),
                                          range(4), 0, numerics)
        assert np.all(np.isfinite(poles)) and np.all(np.isfinite(weights))
        assert weights.shape == (4, len(poles))
        assert np.array_equal(weights[e], np.zeros(len(poles)))
    # the other orbitals keep the dense resolvent at the full depth
    _, e_m = _dense_green(engine, h1, eri, k, 'particle', 0, [0.0])
    z_values = _z_values(e_m)
    want, _ = _dense_green(engine, h1, eri, k, 'particle', 0, z_values)
    worst = max(np.abs(_green(poles, weights, z) - g)[:e, :e].max()
                for z, g in zip(z_values, want))
    print(f'the kept orbitals against the dense resolvent: {worst:.1e}')
    assert worst < 1e-10


def test_spin_channels_agree_for_a_singlet(state):
    h1, eri, engine, k = state
    worst = 0.0
    for spectrum in (removal_spectrum, attachment_spectrum):
        alpha = spectrum(engine, k, h1, eri, 4, (2, 2), range(4), 0, None)
        beta = spectrum(engine, k, h1, eri, 4, (2, 2), range(4), 1, None)
        lo = min(alpha[0].min(), beta[0].min())
        hi = max(alpha[0].max(), beta[0].max())
        for z in (lo - 0.5, 0.5 * (lo + hi) + 0.2j, hi + 0.5):
            worst = max(worst, np.abs(_green(*alpha, z)
                                      - _green(*beta, z)).max())
    print(f'sigma = 0 against sigma = 1 at the default depth: {worst:.1e}')
    assert worst < 1e-9


def test_settings_refusals_and_empty_sectors(state):
    h1, eri, engine, k = state
    for bad in (0, -1, 2.0, True):
        with pytest.raises(ValueError, match='delta_lanczos_depth'):
            Numerics(downfolding={'delta_lanczos_depth': bad})
    for bad in (0.0, -1e-8, float('inf'), float('nan')):
        with pytest.raises(ValueError, match='delta_rank_tol'):
            Numerics(downfolding={'delta_rank_tol': bad})
    for spectrum in (removal_spectrum, attachment_spectrum):
        with pytest.raises(ValueError, match="'static_sa'"):
            spectrum(ExactDiagonalizationSolver(), k, h1, eri, 4, (2, 2),
                     range(4), 0, None)
    # no alpha electron to remove, the alpha spin full: no sector, no pole
    empty = np.ones((1, math.comb(4, 2)))
    poles, weights = removal_spectrum(engine, empty, h1, eri, 4, (0, 2),
                                      range(4), 0, None)
    assert poles.shape == (0,) and weights.shape == (4, 0)
    poles, weights = attachment_spectrum(engine, empty, h1, eri, 4, (4, 2),
                                         [1, 2], 0, None)
    assert poles.shape == (0,) and weights.shape == (2, 0)
    # an empty start block
    poles, weights = block_lanczos(lambda v: v, np.zeros((5, 0)), None)
    assert poles.shape == (0,) and weights.shape == (0, 0)
