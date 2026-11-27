"""The Fock coupling a rotated orbital basis creates between active and
environment orbitals (`fock_coupling`, Downfolding/hybridization.py), the
part of the bare Hamiltonian Delta will fold: zero for canonical orbitals,
the rotated orbital energies in both blocks, the environment levels from the
bare Fock operator, the norms of what is left out (Sigma_x - v_xc on a
reference that is not HF), and the switch `delta` of
`build_effective_hamiltonian`: on with a basis, nothing for canonical
orbitals, `DeltaOffWarning` and its flag when it is turned off, and both
blocks folded for a rotated occupied block (Task 13)."""
import warnings
from dataclasses import replace

import numpy as np
import pytest
from pyscf import dft
from pyscf.scf.hf import get_jk

from src.EffectiveHamiltonian.Downfolding import (build_effective_hamiltonian,
                                                  run_low_level)
from src.EffectiveHamiltonian.Downfolding.hybridization import (
    COUPLING_BLOCKS, DEFAULT_BLOCKS, DeltaOffWarning, fock_coupling)
from src.EffectiveHamiltonian.active_space import (ActiveSpace,
                                                   unrestricted_from_counts)
from src.EffectiveHamiltonian.orbital_basis import (OrbitalBasis,
                                                    orbitals_source,
                                                    semicanonical_rotation)
from tests.downfolding_oracle import OH, unrestricted
from tests.no_downfolding import no_space, water

LEFT_OUT = ('av_ev', 'ao_i', 'ao_ev', 'i_av', 'i_ev')


@pytest.fixture(scope='module')
def mfs():
    return {xc: water(xc) for xc in ('hf', 'pbe')}


@pytest.fixture(scope='module')
def lows(mfs):
    return {xc: run_low_level(mf, method='rpa', mode='casida',
                              natural_orbitals=True)
            for xc, mf in mfs.items()}


@pytest.fixture(scope='module')
def low_hf(mfs):
    return run_low_level(mfs['hf'], method='hf')


@pytest.fixture(scope='module')
def tight_pbe(mfs):
    """PBE water converged until its Fock operator is diagonal in its
    orbitals to 1e-10 (measured 1.3e-10): at the conv_tol 1e-11 of `water` the operator
    of the final density differs from diag(eps) by 1e-7 (the orbitals
    diagonalize the operator of the previous density), which shows in the
    norms of F_HF - F_KS at 1.6e-8."""
    mf = dft.RKS(mfs['pbe'].mol, xc='pbe')
    mf.conv_tol, mf.conv_tol_grad, mf.verbose = 1e-12, 1e-10, 0
    mf.kernel()
    assert mf.converged
    return mf, run_low_level(mf, method='rpa', mode='casida',
                             natural_orbitals=True)


def _coupling(mf, space, blocks=COUPLING_BLOCKS):
    """`fock_coupling` of `space` called directly, with the semicanonical
    rotation of its basis, as the build would call it."""
    T, _ = semicanonical_rotation(space.basis, space, mf.mo_energy)
    return fock_coupling(mf, mf.mo_coeff @ T, T, mf.mo_energy, space,
                         blocks=blocks), T


def _bare_fock(mf, mo):
    """mo.T (hcore + J[D] - K[D]/2) mo, exact four-centre J and K."""
    j, k = get_jk(mf.mol, mf.make_rdm1())
    return mo.T @ (mf.get_hcore() + j - 0.5 * k) @ mo


def _flags(h):
    return [f['flag'] for f in h.numerical_flags]


@pytest.mark.parametrize('xc', ['hf', 'pbe'])
def test_zero_for_canonical_orbitals(mfs, xc):
    mf = mfs[xc]
    norb, nocc = len(mf.mo_energy), int(mf.mo_occ.sum()) // 2
    occupations = np.where(np.arange(norb) < nocc, 2.0, 0.0)
    canonical = ActiveSpace.from_indices(norb, nocc, [2, 3, 4], [5, 6])
    perm = np.concatenate([np.roll(np.arange(nocc), 1),
                           nocc + np.arange(norb - nocc)[::-1]])
    for rotation in (np.eye(norb), np.eye(norb)[:, perm]):
        basis = OrbitalBasis(rotation, occupations, nocc,
                             orbitals_source(mf.mo_coeff))
        c, _ = _coupling(mf, replace(canonical, basis=basis))
        assert c.b_particle.shape == (5, norb - 7)
        assert c.b_hole.shape == (5, 2)
        assert np.all(c.b_particle == 0.0) and np.all(c.b_hole == 0.0)
        assert c.norms == {'av_ev': 0.0, 'ao_i': 0.0}


@pytest.mark.parametrize('xc', ['hf', 'pbe'])
def test_blocks_are_the_rotated_orbital_energies(mfs, lows, xc):
    # the function is called directly, on both blocks
    mf = mfs[xc]
    space = no_space(lows[xc], 2, 3, rotate_occupied=True)
    c, T = _coupling(mf, space)
    nocc, eps = space.nocc, mf.mo_energy
    k = space.n_act_occ
    sides = ((c.b_particle[k:], slice(nocc, None), space.virt_active,
              space.env_virt),
             (c.b_hole[:k], slice(None, nocc), space.occ_active,
              space.env_occ))
    for b, block, act, env in sides:
        e = np.diag(eps[block])
        u_a, u_e = T[block, act], T[block, env]
        want = u_a.T @ e @ u_e @ u_e.T @ e @ u_a
        assert np.abs(b @ b.T - want).max() < 1e-9
    # the other rows of each block are zero
    assert np.all(c.b_particle[:k] == 0.0) and np.all(c.b_hole[k:] == 0.0)
    assert c.norms['av_ev'] == pytest.approx(np.linalg.norm(c.b_particle),
                                             abs=1e-12)
    assert c.norms['ao_i'] == pytest.approx(np.linalg.norm(c.b_hole),
                                            abs=1e-12)
    assert min(c.norms.values()) > 1e-3
    # a block not asked for is zeroed; its norm is reported all the same
    only = fock_coupling(mf, mf.mo_coeff @ T, T, eps, space,
                         blocks=('av_ev',))
    assert np.all(only.b_hole == 0.0)
    assert np.array_equal(only.b_particle, c.b_particle)
    assert only.norms == c.norms and only.blocks == ('av_ev',)
    assert c.blocks == COUPLING_BLOCKS and c.n_act_occ == k


def test_levels_are_those_of_the_bare_fock_operator(mfs, lows, low_hf):
    mf = mfs['pbe']
    space = no_space(lows['pbe'], 2, 3, rotate_occupied=True)
    c, T = _coupling(mf, space)
    F = _bare_fock(mf, mf.mo_coeff @ T)
    ev, i = space.env_virt, space.env_occ
    err_v = np.abs(c.lam_virtual - np.linalg.eigvalsh(F[np.ix_(ev, ev)])).max()
    err_o = np.abs(c.lam_occupied - np.linalg.eigvalsh(F[np.ix_(i, i)])).max()
    err_a = np.abs(c.eps_active
                   - np.diag(F[np.ix_(space.act, space.act)])).max()
    print(f'PBE levels: virtual {err_v:.2e}, occupied {err_o:.2e}, '
          f'active {err_a:.2e}')
    assert max(err_v, err_o, err_a) < 1e-9
    # on HF with exact integrals the bare Fock operator is the reference's own
    mf = mfs['hf']
    space = no_space(lows['hf'], 2, 3)
    h = build_effective_hamiltonian(low_hf, space)
    T, _ = semicanonical_rotation(space.basis, space, mf.mo_energy)
    F_s = T.T @ np.diag(mf.mo_energy) @ T
    ev = space.env_virt
    err = np.abs(h.hybridization.lam_virtual
                 - np.linalg.eigvalsh(F_s[np.ix_(ev, ev)])).max()
    print(f'HF levels against F_s: {err:.2e}')
    assert err < 1e-7


def test_left_out_is_sigma_x_minus_vxc(tight_pbe, lows, low_hf):
    mf, low = tight_pbe
    space = no_space(low, 2, 3, rotate_occupied=True)
    c, T = _coupling(mf, space)
    mo = mf.mo_coeff @ T
    X = _bare_fock(mf, mo) - mo.T @ mf.get_fock() @ mo
    i, ao, av, ev = (space.env_occ, space.occ_active, space.virt_active,
                     space.env_virt)
    pairs = {'av_ev': (av, ev), 'ao_i': (ao, i), 'ao_ev': (ao, ev),
             'i_av': (i, av), 'i_ev': (i, ev)}
    assert set(c.left_out) == set(LEFT_OUT)
    for name, (r, s) in pairs.items():
        want = np.linalg.norm(X[np.ix_(r, s)])
        print(f'PBE left_out {name}: {c.left_out[name]:.6e} '
              f'(F_HF - F_KS {want:.6e}, diff '
              f'{abs(c.left_out[name] - want):.1e})')
        assert abs(c.left_out[name] - want) < 1e-9
        assert want > 1e-3
    # on HF with exact integrals nothing is left out
    h = build_effective_hamiltonian(low_hf, no_space(lows['hf'], 2, 3))
    left = h.diagnostics['fock_coupling']['left_out']
    print('HF exact left_out:', {k: f'{v:.1e}' for k, v in left.items()})
    assert set(left) == set(LEFT_OUT)
    assert max(left.values()) < 1e-7


def test_everything_active_has_no_environment(lows, low_hf):
    low = lows['hf']
    nocc, norb = low.nocc, len(low.eps)
    h = build_effective_hamiltonian(low_hf, no_space(low, nocc, norb - nocc))
    c = h.hybridization
    assert c.b_particle.shape == (norb, 0) and c.lam_virtual.shape == (0,)
    assert c.b_hole.shape == (norb, 0) and c.lam_occupied.shape == (0,)
    assert c.eps_active.shape == (norb,)
    assert c.norms == {'av_ev': 0.0, 'ao_i': 0.0}
    assert c.left_out == {name: 0.0 for name in LEFT_OUT}


def test_on_with_a_basis_and_nothing_for_canonical(lows):
    low = lows['hf']
    space = no_space(low, 2, 2)
    h = build_effective_hamiltonian(low, space, screening='bare')
    assert h.hybridization is not None
    assert h.hybridization.blocks == DEFAULT_BLOCKS
    assert np.all(h.hybridization.b_hole == 0.0)
    assert h.diagnostics['fock_coupling'] == {
        'folded': h.hybridization.norms, 'left_out': h.hybridization.left_out}
    canonical = ActiveSpace.from_indices(space.norb, space.nocc, [3, 4],
                                         [5, 6])
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeltaOffWarning)
        on = build_effective_hamiltonian(low, canonical, screening='bare')
        off = build_effective_hamiltonian(low, canonical, screening='bare',
                                          delta=False)
    for h in (on, off):
        assert h.hybridization is None
        assert 'fock_coupling' not in h.diagnostics
        assert 'delta_off' not in _flags(h)
    assert on.g_eff[0] == off.g_eff[0]
    for k in (1, 2):
        assert np.array_equal(on.g_eff[k], off.g_eff[k])
    assert set(on.diagnostics) == set(off.diagnostics)
    # the test hook selects the blocks
    assert DEFAULT_BLOCKS == COUPLING_BLOCKS
    particle = build_effective_hamiltonian(low, space, screening='bare',
                                           _blocks=('av_ev',))
    assert particle.hybridization.blocks == ('av_ev',)


def test_delta_false_is_flagged(lows):
    low = lows['hf']
    with pytest.warns(DeltaOffWarning, match='delta=False'):
        h = build_effective_hamiltonian(low, no_space(low, 2, 2),
                                        screening='bare', delta=False)
    assert h.hybridization is None
    assert 'fock_coupling' not in h.diagnostics
    assert 'delta_off' in _flags(h)


def test_rotated_occupied_block_folds_both_sides(mfs, lows, low_hf):
    """A rotated occupied block with Delta on (refused before Task 13)
    builds, for the RPA and the HF method: its coupling folds both blocks,
    the hole block is the one `fock_coupling` gives, and sides() holds the
    particle and the hole entry. delta=False still leaves it out."""
    mf = mfs['hf']
    space = no_space(lows['hf'], 2, 2, rotate_occupied=True)
    want, _ = _coupling(mf, space)
    for low in (lows['hf'], low_hf):
        h = build_effective_hamiltonian(low, space)
        c = h.hybridization
        assert c.blocks == COUPLING_BLOCKS
        assert np.abs(c.b_hole - want.b_hole).max() < 1e-12
        assert np.abs(c.b_hole[:2]).max() > 1e-3
        assert [kind for kind, _, _, _ in c.sides()] == ['particle', 'hole']
        assert h.diagnostics['fock_coupling']['folded'] == c.norms
    with pytest.warns(DeltaOffWarning):
        h = build_effective_hamiltonian(low_hf, space, delta=False)
    assert h.hybridization is None


def test_canonical_unrestricted_is_untouched():
    mf = unrestricted(OH, 1)
    eps = tuple(mf.mo_energy)
    nocc = tuple(int(n) for n in mf.mo_occ.sum(axis=1))
    pair = unrestricted_from_counts(eps, nocc, 3, (2, 1))
    low = run_low_level(mf, method='rpa', mode='casida')
    want = build_effective_hamiltonian(low, pair)
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeltaOffWarning)
        for delta in (True, False):
            h = build_effective_hamiltonian(low, pair, delta=delta)
            assert h.hybridization is None
            assert 'fock_coupling' not in h.diagnostics
            assert h.g_eff[0] == want.g_eff[0]
            for k in (1, 2):
                for got, ref in zip(h.g_eff[k], want.g_eff[k]):
                    assert np.array_equal(got, ref)
