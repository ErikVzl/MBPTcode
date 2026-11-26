"""The residual screening of an active space on a rotated orbital basis on
casida: vt = W^r(0) - v from the stored modes (`residual_solve_from_modes`,
spec 6a) against the Cholesky solve on the residual block of A + B in the
rotated pair basis (`brute_force_vt`), on water and on random models with one
and with two spin channels; a permutation of the canonical orbitals is the
canonical route, no active ph pair gives the fully screened W(0) - v, vt is
negative semidefinite and 'crpa_ph' keeps its ph elements; 'mrpa' on a
rotated basis is the canonical 'mrpa' for a permutation, vanishes with every
orbital active and conserves the zeroth moment of the full RPA on the active
pairs."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import block_diag, cho_factor, cho_solve

from src.EffectiveHamiltonian.Downfolding import (build_effective_hamiltonian,
                                                  run_low_level)
from src.EffectiveHamiltonian.Downfolding.methods.rpa import (
    active_pair_projection, residual_solve_from_modes, screening_correction)
from src.EffectiveHamiltonian.active_space import (ActiveSpace,
                                                   active_pair_mask,
                                                   ph_element_mask)
from src.EffectiveHamiltonian.orbital_basis import (OrbitalBasis,
                                                    orbitals_source,
                                                    semicanonical_rotation)
from tests.no_downfolding import brute_force_vt, no_space, water


@pytest.fixture(scope='module')
def mf():
    return water('hf')


@pytest.fixture(scope='module')
def low(mf):
    return run_low_level(mf, method='rpa', mode='casida',
                         natural_orbitals=True)


def _rotated_factors(low, space):
    """B_Q,tu (naux, n_act^2) of the active pairs in the semicanonical
    orbitals of `space`, and those orbitals' T and energies."""
    T, eps_sc = semicanonical_rotation(space.basis, space, low.eps)
    act = space.act
    b = np.einsum('Qpq,pt,qu->Qtu', low.factors, T[:, act], T[:, act])
    return b.reshape(len(b), -1), T, eps_sc


def _permutation_spaces(mf, low):
    """(rotated, canonical, signs): an active space on a basis that is a
    permutation of the canonical orbitals (each block reversed), the
    canonical space of the same orbitals, and the sign of each of its
    semicanonical orbitals against the canonical one."""
    nocc, norb = low.nocc, len(low.eps)
    order = list(range(nocc))[::-1] + list(range(nocc, norb))[::-1]
    occupations = np.concatenate([np.linspace(1.99, 1.95, nocc),
                                  np.linspace(0.05, 0.001, norb - nocc)])
    basis = OrbitalBasis(np.eye(norb)[:, order], occupations, nocc,
                         orbitals_source(mf.mo_coeff))
    rotated = ActiveSpace.from_indices(norb, nocc, [0, 1],
                                       [norb - 3, norb - 2, norb - 1],
                                       basis=basis)
    canonical = ActiveSpace.from_indices(norb, nocc, [nocc - 2, nocc - 1],
                                         [nocc, nocc + 1, nocc + 2])
    T, _ = semicanonical_rotation(basis, rotated, low.eps)
    signs = T[canonical.act, rotated.act]
    assert np.array_equal(np.abs(signs), np.ones(len(signs)))
    return rotated, canonical, signs


def _signed(vt, signs):
    return np.einsum('tuvw,t,u,v,w->tuvw', vt, signs, signs, signs, signs)


def _modes(apb, delta):
    """(omega, X+Y) of the RPA with A + B = apb and A - B = diag(delta), so
    that X+Y diag(1/omega) (X+Y)^T = apb^-1."""
    s = np.sqrt(delta)
    w2, z = np.linalg.eigh(s[:, None] * apb * s[None, :])
    omega = np.sqrt(w2)
    return omega, s[:, None] * z / np.sqrt(omega)[None, :]


def _model(rng, nocc, nvirt, naux):
    """Random orbital energies (a gap at zero) and symmetric factors
    B[Q, p, q] of one spin channel."""
    eps = np.concatenate([np.sort(rng.uniform(-2.0, -0.5, nocc)),
                          np.sort(rng.uniform(0.4, 3.0, nvirt))])
    b = 0.3 * rng.normal(size=(naux, nocc + nvirt, nocc + nvirt))
    return eps, b + b.transpose(0, 2, 1)


def _block_rotation(rng, nocc, nvirt, rotate_occupied):
    """A random orthogonal T, block diagonal in the occupied and the
    virtual orbitals; the occupied block the identity unless rotated."""
    T = np.eye(nocc + nvirt)
    T[nocc:, nocc:] = np.linalg.qr(rng.normal(size=(nvirt, nvirt)))[0]
    if rotate_occupied:
        T[:nocc, :nocc] = np.linalg.qr(rng.normal(size=(nocc, nocc)))[0]
    return T


def _pair_rotation(T, nocc):
    """The (i, a) pairs of the rotated orbitals on the canonical pairs."""
    R = np.einsum('ij,ab->iajb', T[:nocc, :nocc], T[nocc:, nocc:])
    return R.reshape(R.shape[0] * R.shape[1], -1)


@pytest.mark.parametrize('rotate_occupied', [False, True])
def test_casida_identity_equals_dense_residual_solve(low, rotate_occupied):
    """The check of decision 2 on water/6-31G (HF, CAS(4,5) of NOs): the
    identity of spec 6a on the stored modes against one Cholesky solve on
    the residual block of A + B in the rotated pair basis. Measured
    2026-11-26, largest |identity - Cholesky|: 2.7e-15 (virtual
    rotation only), 2.4e-15 (both blocks rotated), on a largest |vt| of
    0.116 and 0.105."""
    space = no_space(low, 2, 3, rotate_occupied)
    T, _ = semicanonical_rotation(space.basis, space, low.eps)
    vt = screening_correction(low, space, 'crpa')
    want = brute_force_vt(low.factors, low.eps, low.nocc, T, space)
    assert np.abs(vt - want).max() < 1e-10
    assert np.abs(want).max() > 1e-2


@pytest.mark.parametrize('rotate_occupied', [False, True])
def test_casida_identity_on_a_random_model(rotate_occupied):
    rng = np.random.default_rng(11)
    nocc, nvirt, naux = 4, 7, 9
    eps, b = _model(rng, nocc, nvirt, naux)
    b_ov = b[:, :nocc, nocc:].reshape(naux, -1)
    delta = (eps[nocc:][None, :] - eps[:nocc][:, None]).ravel()
    omega, xpy = _modes(np.diag(delta) + 4.0 * b_ov.T @ b_ov, delta)
    T = _block_rotation(rng, nocc, nvirt, rotate_occupied)
    space = ActiveSpace.from_indices(nocc + nvirt, nocc, [2, 3],
                                     [nocc, nocc + 1, nocc + 2])
    e = active_pair_projection(
        SimpleNamespace(rotation=T, nocc=nocc, active_space=space))
    act = space.act
    b_act = np.einsum('Qpq,pt,qu->Qtu', b, T[:, act], T[:, act])
    v = b_ov.T @ b_act.reshape(naux, -1)
    vt = -4.0 * v.T @ residual_solve_from_modes(xpy, omega, e, v)
    want = brute_force_vt(b, eps, nocc, T, space)
    assert np.abs(vt.reshape(want.shape) - want).max() < 1e-10


def test_casida_kernel_with_two_spin_channels():
    """Two different spin channels on one auxiliary basis, g = 1: the pairs
    of both stacked, A + B = Lambda + 2K coupling them; the blocks aa, ab,
    bb of -2 V_s^T [(A+B)_rr]^-1 V_t from the stored modes against one
    Cholesky solve on the stacked residual block in the rotated pairs."""
    rng = np.random.default_rng(5)
    naux, norb = 10, 9
    counts = ((3, [1, 2], [3, 4]), (2, [1], [2, 3, 4]))
    models = [_model(rng, nocc, norb - nocc, naux) for nocc, _, _ in counts]
    rotations = [_block_rotation(rng, nocc, norb - nocc, True)
                 for nocc, _, _ in counts]
    spaces = [ActiveSpace.from_indices(norb, nocc, occ, virt)
              for nocc, occ, virt in counts]
    b_ov = np.hstack([b[:, :nocc, nocc:].reshape(naux, -1)
                      for (_, b), (nocc, _, _) in zip(models, counts)])
    delta = np.concatenate([(eps[nocc:][None, :]
                             - eps[:nocc][:, None]).ravel()
                            for (eps, _), (nocc, _, _) in zip(models,
                                                               counts)])
    apb = np.diag(delta) + 2.0 * b_ov.T @ b_ov
    omega, xpy = _modes(apb, delta)
    e = block_diag(*[active_pair_projection(
        SimpleNamespace(rotation=T, nocc=nocc, active_space=space))
        for T, (nocc, _, _), space in zip(rotations, counts, spaces)])
    b_act = [np.einsum('Qpq,pt,qu->Qtu', b, T[:, s.act],
                       T[:, s.act]).reshape(naux, -1)
             for (_, b), T, s in zip(models, rotations, spaces)]
    v = [b_ov.T @ b for b in b_act]
    # the brute force: the residual block of A + B in the rotated pairs
    R = block_diag(*[_pair_rotation(T, nocc)
                     for T, (nocc, _, _) in zip(rotations, counts)])
    r = ~np.concatenate([active_pair_mask(s) for s in spaces])
    v_r = [(b_ov @ R)[:, r].T @ b for b in b_act]
    factor = cho_factor((R.T @ apb @ R)[np.ix_(r, r)])
    for s, t in ((0, 0), (0, 1), (1, 1)):
        vt = -2.0 * v[s].T @ residual_solve_from_modes(xpy, omega, e, v[t])
        want = -2.0 * v_r[s].T @ cho_solve(factor, v_r[t])
        assert np.abs(want).max() > 1e-2
        assert np.abs(vt - want).max() < 1e-10


def test_casida_permutation_basis_is_the_canonical_route(mf, low):
    rotated, canonical, signs = _permutation_spaces(mf, low)
    vt = screening_correction(low, rotated, 'crpa')
    want = _signed(screening_correction(low, canonical, 'crpa'), signs)
    assert np.abs(vt - want).max() < 1e-10


@pytest.mark.parametrize('n_occ,n_virt', [(2, 0), (0, 2)])
def test_casida_no_active_pair(low, n_occ, n_virt):
    """No active virtual (the brief's n_virt = 0) or no active occupied
    orbital: nothing is deleted, and vt is the fully screened
    -4 V^T (A+B)^-1 V of the rotated active orbitals."""
    space = no_space(low, n_occ, n_virt, rotate_occupied=True)
    vt = screening_correction(low, space, 'crpa')
    b_act, _, _ = _rotated_factors(low, space)
    nocc = low.nocc
    b_ov = low.factors[:, :nocc, nocc:].reshape(len(low.factors), -1)
    delta = (low.eps[nocc:][None, :] - low.eps[:nocc][:, None]).ravel()
    v = b_ov.T @ b_act
    want = -4.0 * v.T @ np.linalg.solve(np.diag(delta) + 4.0 * b_ov.T @ b_ov,
                                        v)
    assert np.abs(want).max() > 1e-2
    assert np.abs(vt - want.reshape(vt.shape)).max() < 1e-10


def test_casida_vt_is_negative_semidefinite(low):
    space = no_space(low, 2, 3, rotate_occupied=True)
    vt = screening_correction(low, space, 'crpa')
    n = space.n_act
    assert np.linalg.eigvalsh(vt.reshape(n * n, n * n)).max() <= 1e-12


def test_mrpa_permutation_basis_is_the_canonical_mrpa(mf, low):
    rotated, canonical, signs = _permutation_spaces(mf, low)
    vt = screening_correction(low, rotated, 'mrpa')
    want = _signed(screening_correction(low, canonical, 'mrpa'), signs)
    assert np.abs(want).max() > 1e-3
    assert np.abs(vt - want).max() < 1e-10


def test_mrpa_all_active_vanishes(low):
    nocc, nvirt = low.nocc, len(low.eps) - low.nocc
    space = no_space(low, nocc, nvirt, rotate_occupied=True)
    assert np.abs(screening_correction(low, space, 'mrpa')).max() < 1e-10


def test_mrpa_conserves_the_zeroth_moment(low):
    """The singlet RPA of the decoupled active pairs, at the semicanonical
    energies with A + B = delta~_A + 4 (ia|jb)_eff and A - B = delta~_A,
    has the zeroth moment (X+Y)(X+Y)^T of the full RPA on the active pairs
    of the rotated basis."""
    space = no_space(low, 2, 3, rotate_occupied=True)
    vt = screening_correction(low, space, 'mrpa')
    b_act, T, eps_sc = _rotated_factors(low, space)
    n, k = space.n_act, space.n_act_occ
    v_eff = (b_act.T @ b_act).reshape((n,) * 4) + vt
    k_eff = v_eff[:k, k:, :k, k:].reshape(k * (n - k), -1)
    eps_a = eps_sc[space.act]
    delta = (eps_a[k:][None, :] - eps_a[:k][:, None]).ravel()
    omega, xpy_a = _modes(np.diag(delta) + 4.0 * k_eff, delta)
    xpy = _pair_rotation(T, low.nocc).T @ low.xpy
    want = (xpy @ xpy.T)[np.ix_(active_pair_mask(space),
                                active_pair_mask(space))]
    assert np.abs(xpy_a @ xpy_a.T - want).max() < 1e-10


def test_casida_crpa_ph_keeps_the_ph_elements(low):
    space = no_space(low, 2, 3, rotate_occupied=True)
    vt = screening_correction(low, space, 'crpa')
    vt_ph = screening_correction(low, space, 'crpa_ph')
    assert np.array_equal(vt_ph, vt * ph_element_mask(space))


@pytest.mark.parametrize('screening', ['crpa', 'crpa_ph', 'mrpa'])
def test_casida_builds_every_screening_on_a_rotated_space(low, screening):
    h = build_effective_hamiltonian(low, no_space(low, 2, 2),
                                    screening=screening)
    assert np.isfinite(h.energies.e_c_active)
    assert np.all(np.isfinite(h.g_eff[2]))
