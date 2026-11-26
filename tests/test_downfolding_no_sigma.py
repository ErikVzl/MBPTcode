"""The environment self-energy Sigma_bar^E of an active space on a rotated
orbital basis: the active orbitals at their semicanonical energies eps_sc,
the environment's propagator at eps_sc[env], the residues of W between the
rotated active and environment orbitals with the canonical excitations.

On casida: 'srg' and 'averaged' against the canonical residues over every
orbital rotated by T on both orbital indices and reduced at eps_sc (the
public `srg_self_energy` / `averaged_self_energy`). On imagfrequency (water)
and space-time (NH3, G1): Sigma_bar against casida's at srg_s = None, as the
canonical route test; the samples S^E(mu + i w) at the canonical mid-gap of
the run, against the casida pole sum built here from public functions. A
space with a high virtual NO active: SigmaNearPoleWarning and its flag,
finite energies."""
import dataclasses
import warnings

import numpy as np
import pytest

from src.Base.numerics.numerics import Numerics
from src.Base.orbital_spaces import particle_hole_window
from src.Base.utils.grids.analytic_continuation import pade_nodes
from src.EffectiveHamiltonian.Downfolding import (build_effective_hamiltonian,
                                                  run_low_level)
from src.EffectiveHamiltonian.Downfolding.methods.rpa import (
    SigmaNearPoleWarning, averaged_self_energy, environment_self_energy,
    environment_self_energy_samples, screened_residues, srg_self_energy)
from src.EffectiveHamiltonian.Downfolding.zero_body import Energies
from src.EffectiveHamiltonian.active_space import ActiveSpace
from src.EffectiveHamiltonian.orbital_basis import semicanonical_rotation
from tests.no_downfolding import G1, ammonia, no_space, water

#: the active occupied and virtual counts of the route comparison: water/
#: 6-31G CAS(4,4) of its NOs (with a third virtual NO, at 1.16 Ha, an active
#: energy is within the continuation margin of the window's edge); NH3 on
#: space-time CAS(6,6), as tests/test_downfolding_no_screening.py
COUNTS = {'imagfrequency': (2, 2), 'space-time': (3, 3)}
INFINITE_S = {'srg_s': None}
# the fields of Energies that are records, not energies
RECORDS = ('diagnostics', 'numerical_flags')


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


def _all_active(low):
    """The canonical ActiveSpace with every orbital active."""
    norb, nocc = len(low.eps), low.nocc
    return ActiveSpace.from_indices(norb, nocc, range(nocc),
                                    range(nocc, norb))


def _rotated_residues(low, space):
    """(w, eps_sc): the canonical residues of W over every orbital on both
    indices, rotated by the semicanonical T of `space` on both, the active
    rows and environment columns; and eps_sc."""
    T, eps_sc = semicanonical_rotation(space.basis, space, low.eps)
    every = _all_active(low)
    w = screened_residues(low, every, np.arange(len(low.eps)))
    w = np.einsum('npq,pt,qr->ntr', w, T[:, space.act], T[:, space.env],
                  optimize=True)
    return w, eps_sc


@pytest.mark.parametrize('rotate_occupied', [False, True])
@pytest.mark.parametrize('sigma', ['srg', 'averaged'])
def test_casida_equals_rotated_residues(low, sigma, rotate_occupied):
    """Sigma_bar^E on casida against the canonical residues over all
    orbitals rotated by T on both orbital indices, then `srg_self_energy`
    (the run's srg_s) or `averaged_self_energy` at eps_sc. Target 1e-10;
    measured 2e-18 (virtual NOs) and 1e-17 (both blocks rotated)."""
    space = no_space(low, 2, 3, rotate_occupied)
    w, eps_sc = _rotated_residues(low, space)
    env = space.env
    args = (w, eps_sc[space.act], eps_sc[env], env < low.nocc, low.omega)
    if sigma == 'srg':
        want = srg_self_energy(*args, Numerics().downfolding.srg_s)
    else:
        want = averaged_self_energy(*args)
    got = environment_self_energy(low, space, sigma)
    assert np.abs(want).max() > 1e-3
    err = np.abs(got - want).max()
    print(f'casida {sigma} rotate_occupied={rotate_occupied}: {err:.2e}')
    assert err < 1e-10
    # the residues of a rotated space are those rotated residues
    assert np.abs(screened_residues(low, space) - w).max() < 1e-10


@pytest.mark.parametrize('sigma', ['srg', 'averaged'])
@pytest.mark.parametrize('mode', ['imagfrequency', 'space-time'])
def test_numerical_routes_follow_casida(request, mode, sigma):
    """One space (the casida run's NOs, both blocks rotated) on the
    numerical route and on casida, at srg_s = None, where 'srg' is Eq. 42
    on both: 1e-4 per element, as the canonical route test
    (tests/test_downfolding_routes.py). Measured 6.0e-6 on imagfrequency,
    on the virtual NOs at 0.8 Ha, 0.95 Ha above the mid-gap (|Sigma_bar| up
    to 1.8e-2), and 4.7e-6 on space-time (up to 3.8e-2), equal for both
    forms. On space-time the spread of the even and odd fits is 1.1e-4 on
    two diagonal elements, above the error itself (SigmaContinuationWarning,
    an indicator, not a bound)."""
    if mode == 'imagfrequency':
        cas, num = request.getfixturevalue('low'), \
            request.getfixturevalue('low_if')
        settings = Numerics(downfolding=INFINITE_S)
    else:
        cas, num = request.getfixturevalue('nh3_cas'), \
            request.getfixturevalue('nh3_st')
        settings = Numerics(isdf={'level': 'G1'}, downfolding=INFINITE_S)
    space = no_space(cas, *COUNTS[mode], rotate_occupied=True)
    with warnings.catch_warnings():
        # a space inside the pole-free window by more than the
        # continuation margin
        warnings.simplefilter('error', SigmaNearPoleWarning)
        got = environment_self_energy(num, space, sigma, numerics=settings)
    want = environment_self_energy(cas, space, sigma,
                                   numerics=Numerics(downfolding=INFINITE_S))
    assert np.array_equal(got, got.T)
    assert np.abs(want).max() > 1e-3
    err = np.abs(got - want).max()
    print(f'{mode} {sigma}: {err:.2e} (|Sigma_bar| up to '
          f'{np.abs(want).max():.2e})')
    assert err < 1e-4


def test_canonical_mid_gap_is_used(low, low_if):
    """environment_self_energy_samples on a rotated space samples S^E at the
    run's canonical mid-gap, mu = particle_hole_window(low.eps, low.nocc)[2]:
    against the casida pole sum built here with the residues of the rotated
    space and the environment at eps_sc, at that mu (1e-5, as the canonical
    test of the samples; measured 4.4e-8); at the mid-gap of eps_sc, 0.20 Ha
    higher, the pole sum differs by 2.3e-3."""
    space = no_space(low, 2, 3, rotate_occupied=True)
    nodes = pade_nodes(Numerics().quadrature)
    got = environment_self_energy_samples(low_if, space, nodes)
    _, eps_sc = semicanonical_rotation(space.basis, space, low.eps)
    env = space.env
    sign = np.where(env < low.nocc, -1.0, 1.0)
    poles = eps_sc[env][None, :] + sign[None, :] * low.omega[:, None]
    w = screened_residues(low, space)

    def pole_sum(mu):
        g = 1.0 / ((mu + 1j * nodes)[:, None, None] - poles[None])
        return np.einsum('ntr,nur,qnr->qtu', w, w, g)

    mu = particle_hole_window(low.eps, low.nocc)[2]
    mu_sc = particle_hole_window(eps_sc, low.nocc)[2]
    assert got.shape == (len(nodes), space.n_act, space.n_act)
    err = np.abs(got - pole_sum(mu)).max()
    other = np.abs(got - pole_sum(mu_sc)).max()
    print(f'canonical mid-gap {mu:.6f}: {err:.2e}; semicanonical '
          f'{mu_sc:.6f}: {other:.2e}')
    assert err < 1e-5
    assert other > 1e-4


def test_energies_with_sigma_are_finite_and_flagged_near_poles(low):
    """A space whose active virtuals are the first virtual NO and the one of
    highest energy, <k|F|k> (NO 10, eps_sc 1.41 Ha, beyond the upper edge
    0.97 Ha of the pole-free window of this space): 'averaged' warns
    (SigmaNearPoleWarning, the flag 'sigma_near_pole') naming it at its
    semicanonical energy, and gives finite energies."""
    basis = no_space(low, 2, 2).basis
    norb, nocc = len(low.eps), low.nocc
    u = basis.rotation
    high = nocc + int(np.argmax(np.diag(u.T @ np.diag(low.eps) @ u)[nocc:]))
    assert high > nocc
    space = ActiveSpace.from_indices(norb, nocc, [nocc - 2, nocc - 1],
                                     [nocc, high], basis=basis)
    _, eps_sc = semicanonical_rotation(space.basis, space, low.eps)
    with pytest.warns(SigmaNearPoleWarning) as rec:
        ham = build_effective_hamiltonian(low, space, sigma='averaged')
    text = str([r.message for r in rec
                if r.category is SigmaNearPoleWarning][0])
    assert f'orbital {high} (eps = {eps_sc[high]:.6f}' in text
    assert {'flag': 'sigma_near_pole',
            'message': text} in ham.numerical_flags
    en = ham.energies
    for f in dataclasses.fields(Energies):
        if f.name not in RECORDS:
            assert np.isfinite(getattr(en, f.name)), f.name
    assert np.isfinite(en.e0)
    assert all(np.all(np.isfinite(ham.g_eff[k])) for k in (0, 1, 2))
    assert np.isfinite(ham.solve('pyscf_fci')[0])


def test_entry_points_refuse_by_name_on_a_rotated_space(low, low_if):
    """What Sigma^E does not do on a rotated space is refused by name:
    sigma_energies='qp' (the QP energies are the canonical orbitals'), the
    residues off casida and the samples on casida."""
    space = no_space(low, 2, 2)
    with pytest.raises(NotImplementedError, match="sigma_energies='qp'"):
        environment_self_energy(low, space, 'srg', sigma_energies='qp')
    with pytest.raises(NotImplementedError, match="sigma_energies='qp'"):
        build_effective_hamiltonian(low, space, sigma='srg',
                                    sigma_energies='qp')
    with pytest.raises(NotImplementedError, match="use mode='casida'"):
        screened_residues(low_if, space)
    with pytest.raises(NotImplementedError, match="use mode='imagfrequency'"):
        environment_self_energy_samples(low, space, [1.0])
