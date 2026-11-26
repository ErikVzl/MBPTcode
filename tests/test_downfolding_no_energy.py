"""The RPA correlation energy of the hybridized active space, E_c^{A,Delta}
(`hybridized_correlation_energy`, spec 6b), and the E_0 it enters as its own
term, `e_c_hybridization` = E_c^{A,Delta} - E_c^A.

casida against the unsymmetric dense RPA (`dense_drpa_ec` in
`tests/no_downfolding.py`) on all canonical pairs with K contracted here;
the two kernels with two different spin channels on a model, and with
beta = alpha against the one-channel g = 2 value; imagfrequency (water) and
space-time (NH3, G1) against casida on one space and one v_eff; zero for
canonical orbitals (not computed), with every orbital active and without an
active ph pair; the E_c terms cancel with every orbital active on coarse
grids (nfreq_rpa = 6); `e0` holds every term, `e0_without_delta` and
g_eff[0] are the constant of the decoupled model, and with Delta off the
three are one number."""
import warnings

import numpy as np
import pytest

from src.Base.numerics.numerics import Numerics
from src.EffectiveHamiltonian.Downfolding import (build_effective_hamiltonian,
                                                  run_low_level)
from src.EffectiveHamiltonian.Downfolding.hybridization import DeltaOffWarning
from src.EffectiveHamiltonian.Downfolding.methods import rpa
from src.EffectiveHamiltonian.Downfolding.methods.rpa import (
    active_correlation_energy, active_pair_factors,
    hybridized_correlation_energy, hybridized_ec_plasmon,
    hybridized_ec_quadrature)
from src.EffectiveHamiltonian.active_space import ActiveSpace
from src.EffectiveHamiltonian.orbital_basis import (OrbitalBasis,
                                                    orbitals_source,
                                                    semicanonical_rotation)
from tests.no_downfolding import (G1, ammonia, dense_drpa_ec, no_space,
                                  water)

#: the coarse grids of the cancellation test: 6 frequencies (and 10 tau
#: points on space-time, the fewest whose transform fit does not warn)
COARSE = {'imagfrequency': Numerics(quadrature={'nfreq_rpa': 6}),
          'space-time': Numerics(isdf={'level': 'G1'},
                                 quadrature={'ntau_rpa': 10,
                                             'nfreq_rpa': 6})}


@pytest.fixture(scope='module')
def mf():
    return water('hf')


@pytest.fixture(scope='module')
def low(mf):
    return run_low_level(mf, method='rpa', mode='casida',
                         natural_orbitals=True)


@pytest.fixture(scope='module')
def low_if(mf):
    return run_low_level(mf, method='rpa', mode='imagfrequency')


@pytest.fixture(scope='module')
def nh3():
    return ammonia('hf')


@pytest.fixture(scope='module')
def nh3_cas(nh3):
    return run_low_level(nh3, method='rpa', mode='casida',
                         natural_orbitals=True)


@pytest.fixture(scope='module')
def nh3_st(nh3):
    return run_low_level(nh3, method='rpa', mode='space-time', numerics=G1)


def _bare_v(low, space):
    """The fitted v of the run on the active orbitals of `space`, chemist
    (tu|vw) on space.act: v_eff of 'bare'."""
    b = active_pair_factors(low, space)
    n = space.n_act
    b = b.reshape(len(b), n * n)
    return (b.T @ b).reshape((n,) * 4)


def _permutation_space(mf, low):
    """An active space on a basis that permutes the canonical orbitals (each
    block reversed): canonical active orbitals, with an environment."""
    nocc, norb = low.nocc, len(low.eps)
    order = list(range(nocc))[::-1] + list(range(nocc, norb))[::-1]
    occupations = np.concatenate([np.linspace(1.99, 1.95, nocc),
                                  np.linspace(0.05, 0.001, norb - nocc)])
    basis = OrbitalBasis(np.eye(norb)[:, order], occupations, nocc,
                         orbitals_source(mf.mo_coeff))
    return ActiveSpace.from_indices(norb, nocc, [0, 1],
                                    [norb - 3, norb - 2, norb - 1],
                                    basis=basis)


def _model_channel(rng, nocc, nvirt, n_ao, n_av):
    """(delta, m) of one model spin channel: canonical pair energies and
    m[(t,u),(i,a)] = O_ti U_ua for random orthonormal active rows."""
    eps_o = np.sort(rng.uniform(-2.0, -0.5, nocc))
    eps_v = np.sort(rng.uniform(0.4, 3.0, nvirt))
    o = np.linalg.qr(rng.normal(size=(nocc, nocc)))[0][:, :n_ao].T
    u = np.linalg.qr(rng.normal(size=(nvirt, nvirt)))[0][:, :n_av].T
    m = np.einsum('ti,ua->tuia', o, u).reshape(n_ao * n_av, nocc * nvirt)
    return (eps_v[None, :] - eps_o[:, None]).ravel(), m


def _model_grid(n=400, cut=2000.0):
    """Gauss-Legendre on (0, inf) by w = x / (1 - x): (points, weights),
    without the points above `cut` Ha. There ln det(1 - Pi K) + tr(Pi K)
    (of order w^-4) is below the rounding of its two terms, which the
    weights (up to 1e5) would lift to 2e-12 (measured, beta = alpha); the
    tail left out is 1.5e-10 of the model's energy (measured)."""
    x, w = np.polynomial.legendre.leggauss(n)
    x, w = 0.5 * (x + 1.0), 0.5 * w
    points, weights = x / (1.0 - x), w / (1.0 - x) ** 2
    keep = points < cut
    return points[keep], weights[keep]


def test_zero_for_canonical_and_for_everything_active(mf, low, monkeypatch):
    """Exactly 0.0, and not computed, for canonical orbitals and with Delta
    off; to rounding with canonical active orbitals on a basis (a
    permutation) and with every orbital active."""
    # canonical active orbitals in a rotated basis: the inactive pairs
    # carry no K, so E_c^{A,Delta} is E_c^A
    space = _permutation_space(mf, low)
    v = _bare_v(low, space)
    diff = (hybridized_correlation_energy(low, space, v)
            - active_correlation_energy(low, space, v))
    print(f'permutation basis: E_c^(A,Delta) - E_c^A = {diff:.1e}')
    assert abs(diff) < 1e-12
    nocc, norb = low.nocc, len(low.eps)
    every = no_space(low, nocc, norb - nocc)
    h = build_effective_hamiltonian(low, every)
    print(f'every orbital active: e_c_hybridization = '
          f'{h.energies.e_c_hybridization:.1e}')
    assert abs(h.energies.e_c_hybridization) < 1e-12

    def refuse(*args):
        raise AssertionError('hybridized_correlation_energy was called')

    monkeypatch.setattr(rpa, 'hybridized_correlation_energy', refuse)
    canonical = ActiveSpace.from_indices(norb, nocc, [3, 4], [5, 6, 7])
    for delta in (True, False):
        h = build_effective_hamiltonian(low, canonical, delta=delta)
        assert h.energies.e_c_hybridization == 0.0
    with pytest.warns(DeltaOffWarning):
        h = build_effective_hamiltonian(low, no_space(low, 2, 3),
                                        delta=False)
    assert h.energies.e_c_hybridization == 0.0
    terms = rpa.downfold(low, no_space(low, 2, 3))
    assert terms.e_c_hybridization == 0.0
    low_hf = run_low_level(mf, method='hf')
    h = build_effective_hamiltonian(low_hf, no_space(low, 2, 3))
    assert h.hybridization is not None
    assert h.energies.e_c_hybridization == 0.0


@pytest.mark.parametrize('rotate_occupied', [False, True])
def test_casida_equals_independent_dense(low, rotate_occupied):
    """The plasmon form over all canonical pairs against the unsymmetric
    dense RPA, with K = O U v_eff O U contracted here from the rotation."""
    space = no_space(low, 2, 3, rotate_occupied=rotate_occupied)
    v_eff = _bare_v(low, space)
    T, _ = semicanonical_rotation(space.basis, space, low.eps)
    nocc, k = low.nocc, space.n_act_occ
    o = T[:nocc, space.occ_active].T
    u = T[nocc:, space.virt_active].T
    K = np.einsum('ti,ua,tusv,sj,vb->iajb', o, u, v_eff[:k, k:, :k, k:],
                  o, u)
    npair = nocc * (len(low.eps) - nocc)
    delta = (low.eps[nocc:][None, :] - low.eps[:nocc][:, None]).ravel()
    want = dense_drpa_ec(delta, K.reshape(npair, npair))
    got = hybridized_correlation_energy(low, space, v_eff)
    e_a = active_correlation_energy(low, space, v_eff)
    print(f'rotate_occupied={rotate_occupied}: E_c^(A,Delta) {got:.10f}, '
          f'dense {want:.10f}, diff {abs(got - want):.1e}; '
          f'E_c^A {e_a:.10f}, e_c_hybridization {got - e_a:+.3e}')
    assert abs(got - want) < 1e-10
    assert abs(got - e_a) > 1e-6


def test_kernels_with_two_spin_channels():
    """Two channels that differ in every count, g = 1: the plasmon form
    over the stacked pairs is the imaginary-frequency form; with beta =
    alpha both are the one-channel g = 2 value."""
    rng = np.random.default_rng(11)
    (da, ma), (db, mb) = (_model_channel(rng, 4, 6, 2, 2),
                          _model_channel(rng, 3, 7, 1, 3))
    nph = len(ma) + len(mb)
    x = rng.normal(size=(nph, 5))
    k_ph = 0.06 * x @ x.T
    points, weights = _model_grid()
    plasmon = hybridized_ec_plasmon([da, db], [ma, mb], k_ph, 1)
    quadrature = hybridized_ec_quadrature([da, db], [ma, mb], k_ph, 1,
                                          points, weights)
    print(f'two channels: plasmon {plasmon:.12f}, quadrature '
          f'{quadrature:.12f}, diff {abs(plasmon - quadrature):.1e}')
    assert abs(plasmon - quadrature) < 1e-8
    kk = k_ph[:len(ma), :len(ma)]
    k2 = np.block([[kk, kk], [kk, kk]])
    one = hybridized_ec_plasmon([da], [ma], kk, 2)
    for name, two, single in (
            ('plasmon', hybridized_ec_plasmon([da, da], [ma, ma], k2, 1),
             one),
            ('quadrature',
             hybridized_ec_quadrature([da, da], [ma, ma], k2, 1, points,
                                      weights),
             hybridized_ec_quadrature([da], [ma], kk, 2, points, weights))):
        print(f'beta = alpha, {name}: two channels {two:.14f}, one '
              f'{single:.14f}, diff {abs(two - single):.1e}')
        assert abs(two - single) < 1e-12
    assert abs(one - dense_drpa_ec(da, ma.T @ kk @ ma)) < 1e-12


@pytest.mark.parametrize('route', ['imagfrequency', 'space-time'])
def test_quadrature_routes_follow_casida(route, low, low_if, nh3_cas,
                                         nh3_st):
    """One space (the casida run's NOs) and one v_eff on the route and on
    casida: E_c^{A,Delta} and e_c_hybridization differ by the quadrature
    alone. imagfrequency on water, 1e-8; space-time on NH3 (G1), 1e-5."""
    cas, other, counts, bar = {
        'imagfrequency': (low, low_if, (2, 3), 1e-8),
        'space-time': (nh3_cas, nh3_st, (3, 3), 1e-5)}[route]
    space = no_space(cas, *counts)
    v_eff = _bare_v(cas, space)
    want = hybridized_correlation_energy(cas, space, v_eff)
    got = hybridized_correlation_energy(other, space, v_eff)
    d_want = want - active_correlation_energy(cas, space, v_eff)
    d_got = got - active_correlation_energy(other, space, v_eff)
    print(f'{route}: E_c^(A,Delta) {got:.10f}, casida {want:.10f}, diff '
          f'{abs(got - want):.1e}; e_c_hybridization {d_got:+.6e}, casida '
          f'{d_want:+.6e}, diff {abs(d_got - d_want):.1e}')
    assert abs(got - want) < bar
    assert abs(d_got - d_want) < bar


@pytest.mark.parametrize('route', ['casida', 'imagfrequency', 'space-time'])
def test_cancels_with_every_orbital_active_on_any_grid(route, mf, low, nh3,
                                                       nh3_cas):
    """Every orbital active on a basis, with Delta on, on the coarse grids
    (nfreq_rpa = 6): E_c^{A,Delta} is built on the full E_c's own grid and
    transform, so e_c_full - e_c_active - e_c_hybridization vanishes."""
    if route == 'casida':
        run, basis_low = low, low
    elif route == 'imagfrequency':
        run = run_low_level(mf, method='rpa', mode=route,
                            numerics=COARSE[route])
        assert len(run.cache['grid'][0]) == 6
        basis_low = low
    else:
        run = run_low_level(nh3, method='rpa', mode=route,
                            numerics=COARSE[route])
        assert run.cache['energy_grid'].nfreq == 6
        basis_low = nh3_cas
    nocc, norb = basis_low.nocc, len(basis_low.eps)
    h = build_effective_hamiltonian(run, no_space(basis_low, nocc,
                                                  norb - nocc))
    en = h.energies
    rest = en.e_c_full - en.e_c_active - en.e_c_hybridization
    print(f'{route}: e_c_full {en.e_c_full:.10f}, e_c_hybridization '
          f'{en.e_c_hybridization:.1e}, e_c_full - e_c_active - '
          f'e_c_hybridization {rest:.1e}')
    assert h.hybridization is not None
    assert abs(rest) < 1e-10


@pytest.mark.parametrize('counts', [(0, 3), (2, 0)])
def test_no_active_pair_gives_zero(low, counts):
    """No active virtual or no active occupied orbital: no active ph pair,
    nothing to hybridize, 0.0."""
    space = no_space(low, *counts)
    v = _bare_v(low, space)
    assert hybridized_correlation_energy(low, space, v) == 0.0
    h = build_effective_hamiltonian(low, space)
    assert h.energies.e_c_hybridization == 0.0
    assert h.energies.e0 == h.energies.e0_without_delta == h.g_eff[0]


def test_e0_holds_every_term_and_g_eff_the_static_constant(low):
    """With Delta on, e0 has every term and g_eff[0] is e0_without_delta,
    the constant of the decoupled model; with canonical orbitals the three
    are one number, exactly."""
    h = build_effective_hamiltonian(low, no_space(low, 2, 3))
    en = h.energies
    print(f'water casida NO (2, 3), crpa: e_c_hybridization '
          f'{en.e_c_hybridization:+.6e}, e0 - e0_without_delta '
          f'{en.e0 - en.e0_without_delta:+.6e}')
    assert h.g_eff[0] == en.e0_without_delta
    assert en.e_c_hybridization != 0.0
    assert abs((en.e0 - en.e0_without_delta) + en.e_c_hybridization) < 1e-12
    assert en.ec == en.e_c_full - en.e_c_active - en.e_c_hybridization
    assert en.e0_without_delta == (en.e_nuc + en.e_core + en.eh + en.ex
                                   + (en.e_c_full - en.e_c_active)
                                   - en.e_sigma_trace)
    assert en.e0 == (en.e_nuc + en.e_core + en.eh + en.ex + en.ec
                     - en.e_sigma_trace)
    nocc, norb = low.nocc, len(low.eps)
    canonical = ActiveSpace.from_indices(norb, nocc, [3, 4], [5, 6, 7])
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeltaOffWarning)
        h = build_effective_hamiltonian(low, canonical)
    en = h.energies
    assert en.e_c_hybridization == 0.0
    assert en.e0 == en.e0_without_delta == h.g_eff[0]
