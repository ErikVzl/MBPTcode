"""The static downfolding on a rotated orbital basis with two spin channels
(an unrestricted reference, g = 1 each; spec 6c), on the OH radical of
tests/downfolding_oracle.py with a random rotation per spin
(`random_basis`): what Tasks 2 to 7 built per channel and tested on one.

Method 'hf' is UCASCI on the rotated orbitals; an identity basis per spin is
the canonical route; the bare RPA takes the semicanonical active pairs of
both spins, stacked; the residual screening on casida against the Cholesky
solve on the stacked residual pairs (`brute_force_vt`), on imagfrequency and
space-time P - P^A against the per-spin dense solve (`brute_force_pr`), and
imagfrequency against casida; Sigma^E of each spin against the rotated
canonical residues; the Fock coupling of each spin from that spin's bare
Fock operator; E_c^{A,Delta} against the dense RPA of the stacked canonical
pairs; 'mrpa' keeps the zeroth moment of the stacked active pairs; and a
closed shell run as UHF with the same rotation on both spins is the
restricted result. Delta for two channels does not exist: the default
delta=True is refused, and every build here passes delta=False."""
import dataclasses
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from pyscf import gto, mcscf, scf
from pyscf.scf import addons
from pyscf.scf.hf import get_jk
from scipy.linalg import block_diag

from src.Base.numerics.constants import DOWNFOLDING_NO_CUT_WARN
from src.Base.numerics.numerics import Numerics
from src.EffectiveHamiltonian.Downfolding import (build_effective_hamiltonian,
                                                  run_low_level)
from src.EffectiveHamiltonian.Downfolding.hybridization import (
    COUPLING_BLOCKS, DeltaOffWarning, fock_coupling)
from src.EffectiveHamiltonian.Downfolding.methods.rpa import (
    SigmaNearPoleWarning, active_correlation_energy, active_pair_factors,
    active_pair_projection, active_polarizability_static,
    averaged_self_energy,
    environment_self_energy, hybridized_correlation_energy,
    screened_residues, screening_correction, srg_self_energy)
from src.EffectiveHamiltonian.Downfolding.zero_body import Energies
from src.EffectiveHamiltonian.active_space import (ActiveSpace,
                                                   active_pair_mask)
from src.EffectiveHamiltonian.orbital_basis import (OrbitalBasis,
                                                    natural_orbital_basis,
                                                    orbitals_source,
                                                    semicanonical_rotation)
from tests.downfolding_oracle import NH, OH, unrestricted
from tests.no_downfolding import (G1, brute_force_pr, brute_force_vt,
                                  dense_drpa_ec, no_space, random_basis,
                                  water)

#: the active orbitals of each spin of OH/6-31G (5 alpha and 4 beta
#: occupied of 11): three core orbitals per spin, four active per spin,
#: (N_alpha, N_beta) = (2, 1) active electrons
ACTIVE = (([3, 4], [5, 6]), ([3], [4, 5, 6]))
#: those of triplet NH/6-31G (5 alpha and 3 beta occupied), two core
#: orbitals per spin, four active, (3, 1) active electrons
ACTIVE_NH = (([2, 3, 4], [5]), ([2], [3, 4, 5]))
# the fields of Energies that are records, not energies
RECORDS = ('diagnostics', 'numerical_flags')


@pytest.fixture(scope='module')
def mf():
    return unrestricted(OH, 1)


@pytest.fixture(scope='module')
def low(mf):
    return run_low_level(mf, method='rpa', mode='casida')


@pytest.fixture(scope='module')
def low_hf(mf):
    return run_low_level(mf, method='hf')


@pytest.fixture(scope='module')
def low_if(mf):
    return run_low_level(mf, method='rpa', mode='imagfrequency')


@pytest.fixture(scope='module')
def mf_st():
    """OH/cc-pVDZ, UHF density fitted on cc-pvdz-ri, the mean field of the
    canonical unrestricted space-time tests (6-31G has no RI basis of its
    own for the ISDF factors)."""
    mol = gto.M(atom=OH, basis='cc-pvdz', spin=1, cart=False, verbose=0)
    mf = scf.UHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.conv_tol, mf.verbose = 1e-10, 0
    mf.kernel()
    assert mf.converged
    return mf


@pytest.fixture(scope='module')
def low_st(mf_st):
    return run_low_level(mf_st, method='rpa', mode='space-time', numerics=G1)


@pytest.fixture(scope='module')
def pair(low):
    return _spaces(low.mf, random_basis(low))


@pytest.fixture(scope='module')
def tight_uhf():
    """OH UHF converged until its Fock operators are diagonal in their
    orbitals to 2.8e-9 (measured): at the conv_tol 1e-10 of `unrestricted`
    they differ from diag(eps) by 1.2e-7, which is the left_out of a UHF."""
    mf = unrestricted(OH, 1)
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-8
    mf.kernel(dm0=mf.make_rdm1())
    assert mf.converged
    return mf


@pytest.fixture(scope='module')
def uks():
    return unrestricted(OH, 1, xc='pbe0')


@pytest.fixture(scope='module')
def uks_nh():
    """Triplet NH, UKS-PBE0, converged until its Fock operators are
    diagonal in their orbitals to 1.3e-10 (measured), as tight_pbe of
    tests/test_downfolding_no_coupling.py for the restricted reference. OH
    cannot be: its UKS-PBE0 stalls with an occupied-virtual gradient of
    3e-7 (measured, DIIS and second-order alike), which shows in the norms
    of F_HF - F_KS at 1.7e-7."""
    mf = unrestricted(NH, 2, xc='pbe0')
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-10
    mf.kernel(dm0=mf.make_rdm1())
    assert mf.converged
    return mf


def _nocc(mf):
    return tuple(int(np.count_nonzero(occ)) for occ in mf.mo_occ)


def _spaces(mf, bases, active=ACTIVE):
    """The pair of ActiveSpaces of `active` on `mf`, each on its basis
    (None: canonical)."""
    norb = mf.mo_coeff.shape[-1]
    return tuple(ActiveSpace.from_indices(norb, n, occ, virt, basis=b)
                 for n, (occ, virt), b in zip(_nocc(mf), active, bases))


def _rotations(mf, spaces):
    """(T, eps_sc) of each spin: the semicanonical rotation of its basis."""
    return [semicanonical_rotation(s.basis, s, e)
            for s, e in zip(spaces, mf.mo_energy)]


def _static(low, spaces, **kwargs):
    """build_effective_hamiltonian with delta=False, its DeltaOffWarning
    asserted."""
    with pytest.warns(DeltaOffWarning, match='delta=False'):
        return build_effective_hamiltonian(low, spaces, delta=False, **kwargs)


def _flags(h):
    return [f['flag'] for f in h.numerical_flags]


def _bare_v(low, spaces):
    """The fitted v of the run on the active orbitals of both spins, the
    blocks (aa, ab, bb), chemist: v_eff of 'bare'."""
    b = [x.reshape(len(x), -1) for x in active_pair_factors(low, spaces)]
    n = spaces[0].n_act
    return tuple((b[s].T @ b[t]).reshape((n,) * 4)
                 for s, t in ((0, 0), (0, 1), (1, 1)))


def _stacked_ph(spaces, eps_sc, v):
    """(delta, K) of the active ph pairs of both spins, stacked: the
    semicanonical energy differences and K^{st}_ia,jb = v^{st}[i,a,j,b]."""
    occ = [s.n_act_occ for s in spaces]
    delta = []
    for space, e, k in zip(spaces, eps_sc, occ):
        e = e[space.act]
        delta.append((e[k:][None, :] - e[:k][:, None]).ravel())
    blocks = {(0, 0): v[0], (0, 1): v[1], (1, 1): v[2],
              (1, 0): v[1].transpose(2, 3, 0, 1)}
    k_ph = np.block([[blocks[s, t][:occ[s], occ[s]:, :occ[t], occ[t]:]
                      .reshape(len(delta[s]), len(delta[t]))
                      for t in (0, 1)] for s in (0, 1)])
    return np.concatenate(delta), k_ph


def _modes(apb, delta):
    """(omega, X+Y) of the RPA with A + B = apb and A - B = diag(delta)."""
    s = np.sqrt(delta)
    w2, z = np.linalg.eigh(s[:, None] * apb * s[None, :])
    omega = np.sqrt(w2)
    return omega, s[:, None] * z / np.sqrt(omega)[None, :]


def _pair_rotation(T, nocc):
    """The (i, a) pairs of the rotated orbitals on the canonical pairs."""
    R = np.einsum('ij,ab->iajb', T[:nocc, :nocc], T[nocc:, nocc:])
    return R.reshape(R.shape[0] * R.shape[1], -1)


def _isdf_ov(low, s):
    """B_Q,ia (naux, nocc, nvirt) of spin s on the ISDF factors of a
    space-time result, sum_P D_PQ X^s_Pi X^s_Pa."""
    x, d = low.cache['factors']
    n = low.nocc[s]
    return np.einsum('PQ,Pi,Pa->Qia', d, x[s][:, :n], x[s][:, n:],
                     optimize=True)


def test_a_pair_of_bases(mf, low, pair):
    bases = random_basis(low)
    assert isinstance(bases, tuple) and len(bases) == 2
    for s, (basis, space, eps) in enumerate(zip(bases, pair, mf.mo_energy)):
        nocc = basis.nocc
        assert nocc == _nocc(mf)[s] and space.basis is not None
        assert basis.source == orbitals_source(mf.mo_coeff[s])
        assert basis.rotates_occupied and basis.rotates_virtual
        R = basis.rotation
        assert np.abs(R.T @ R - np.eye(len(R))).max() < 1e-12
        assert not R[:nocc, nocc:].any() and not R[nocc:, :nocc].any()
        for block in (basis.occupations[:nocc], basis.occupations[nocc:]):
            assert np.diff(block).max() < -DOWNFOLDING_NO_CUT_WARN
        T, eps_sc = semicanonical_rotation(space.basis, space, eps)
        F = T.T @ np.diag(eps) @ T
        assert np.abs(np.diag(F) - eps_sc).max() < 1e-12
        for idx in (space.env_occ, space.occ_active, space.virt_active,
                    space.env_virt):
            block = F[np.ix_(idx, idx)]
            off = np.abs(block - np.diag(np.diag(block))).max()
            print(f'spin {s}: largest off-diagonal F in a block {off:.1e}')
            assert off < 1e-10
    # each spin its own rotation; the occupied block kept on request
    assert not np.array_equal(bases[0].rotation[4:, 4:],
                              bases[1].rotation[4:, 4:])
    for basis in random_basis(low, rotate_occupied=False):
        assert not basis.rotates_occupied and basis.rotates_virtual


def test_hf_method_is_ucasci_on_the_rotated_orbitals(mf, low_hf, pair):
    h = _static(low_hf, pair)
    mo = []
    for c, space, (T, _) in zip(mf.mo_coeff, pair, _rotations(mf, pair)):
        order = list(space.env_occ) + list(space.act) + list(space.env_virt)
        mo.append((c @ T)[:, order])
    cas = mcscf.UCASCI(mf, 4, (2, 1))
    cas.verbose = 0
    ref = cas.kernel(tuple(mo))[0]
    err = abs(h.solve('pyscf_fci')[0] - ref)
    print(f'UCASCI on the rotated orbitals: |dE| {err:.1e}')
    assert err < 1e-9
    assert all(b is s.basis for b, s in zip(h.space['basis'], pair))
    for got, (T, _), c in zip(h.space['mo'], _rotations(mf, pair),
                              mf.mo_coeff):
        assert np.array_equal(got, c @ T)


def test_identity_basis_reproduces_canonical(mf, low):
    bases = [OrbitalBasis(np.eye(len(e)), np.asarray(occ, float), n,
                          orbitals_source(c))
             for e, occ, n, c in zip(mf.mo_energy, mf.mo_occ, _nocc(mf),
                                     mf.mo_coeff)]
    want = build_effective_hamiltonian(low, _spaces(mf, (None, None)),
                                       screening='bare')
    got = _static(low, _spaces(mf, bases), screening='bare')
    assert abs(got.g_eff[0] - want.g_eff[0]) < 1e-12
    for k in (1, 2):
        assert len(got.g_eff[k]) == len(want.g_eff[k]) == k + 1
        for ours, ref in zip(got.g_eff[k], want.g_eff[k]):
            assert np.abs(ours - ref).max() < 1e-12
    for field in dataclasses.fields(Energies):
        if field.name not in RECORDS:
            assert abs(getattr(got.energies, field.name)
                       - getattr(want.energies, field.name)) < 1e-12
    assert 'basis' not in want.space and 'mo' not in want.space


def test_bare_rpa_uses_semicanonical_active_pairs(mf, low, pair):
    h = _static(low, pair, screening='bare')
    rotations = _rotations(mf, pair)
    b = [np.einsum('Qpq,pt,qu->Qtu', f, T[:, s.act], T[:, s.act])
         .reshape(len(f), -1)
         for f, s, (T, _) in zip(low.factors, pair, rotations)]
    n = pair[0].n_act
    v = tuple((b[s].T @ b[t]).reshape((n,) * 4)
              for s, t in ((0, 0), (0, 1), (1, 1)))
    delta, k_ph = _stacked_ph(pair, [e for _, e in rotations], v)
    want = dense_drpa_ec(delta, k_ph, g=1)
    err = abs(h.energies.e_c_active - want)
    print(f'bare E_c^A {h.energies.e_c_active:.12f}, dense {want:.12f}, '
          f'diff {err:.1e}')
    assert err < 1e-10
    assert h.energies.e_c_full == low.e_c_full


def test_casida_identity_equals_dense_residual_solve(mf, low, pair):
    Ts = [T for T, _ in _rotations(mf, pair)]
    vt = screening_correction(low, pair, 'crpa')
    want = brute_force_vt(low.factors, low.eps, low.nocc, Ts, pair)
    assert len(vt) == len(want) == 3
    for name, ours, ref in zip(('aa', 'ab', 'bb'), vt, want):
        err = np.abs(ours - ref).max()
        print(f'casida vt {name}: |identity - Cholesky| {err:.1e}, '
              f'largest |vt| {np.abs(ref).max():.3f}')
        assert np.abs(ref).max() > 1e-2
        assert err < 1e-10


@pytest.mark.parametrize('mode', ['imagfrequency', 'space-time'])
def test_identity_equals_brute_force(mf, mf_st, low_if, low_st, mode):
    """P(0) - P^A(0) against the per-spin dense solve over the residual
    pairs, summed with g = 1; on space-time on the route's own ISDF factors,
    with P(0) and P^A(0) on its static tau grid, so the difference is that
    grid's quadrature error."""
    run, ref_mf = {'imagfrequency': (low_if, mf),
                   'space-time': (low_st, mf_st)}[mode]
    spaces = _spaces(ref_mf, random_basis(run))
    pr = run.cache['p0'] - active_polarizability_static(run, spaces)
    ref = 0.0
    for s, (space, (T, _)) in enumerate(zip(spaces,
                                            _rotations(ref_mf, spaces))):
        n = run.nocc[s]
        b_ov = (_isdf_ov(run, s) if mode == 'space-time'
                else run.factors[s][:, :n, n:])
        ref = ref + brute_force_pr(b_ov, run.eps[s], n, T, space, g=1)
    err = np.abs(pr - ref).max()
    print(f'{mode}: |P - P^A - P^r| {err:.1e}, largest |P^r| '
          f'{np.abs(ref).max():.3f}')
    assert np.abs(ref).max() > 1e-2
    assert err < (1e-10 if mode == 'imagfrequency' else 1e-6)


def test_routes_match_casida(low, low_if, pair):
    vt = screening_correction(low_if, pair, 'crpa')
    want = screening_correction(low, pair, 'crpa')
    for name, ours, ref in zip(('aa', 'ab', 'bb'), vt, want):
        err = np.abs(ours - ref).max()
        print(f'imagfrequency vt {name} against casida: {err:.1e}')
        assert np.abs(ref).max() > 1e-2
        assert err < 1e-9


@pytest.mark.parametrize('sigma', ['srg', 'averaged'])
def test_sigma_equals_rotated_residues(mf, low, pair, sigma):
    """Sigma_bar^E of each spin against the canonical residues over every
    orbital of that spin, rotated by its T on both orbital indices, then
    `srg_self_energy` (the run's srg_s) or `averaged_self_energy` at its
    eps_sc and the Omega of the two-spin RPA."""
    norb = mf.mo_coeff.shape[-1]
    every = tuple(ActiveSpace.from_indices(norb, n, range(n), range(n, norb))
                  for n in _nocc(mf))
    canonical = screened_residues(low, every, (np.arange(norb),) * 2)
    if sigma == 'averaged':
        # the random occupied rotation mixes the 1s orbital into an active
        # occupied orbital of each spin (eps_sc -19.3 and -12.3 Ha), far
        # outside the pole-free window
        with pytest.warns(SigmaNearPoleWarning, match='alpha orbital 3'):
            got = environment_self_energy(low, pair, sigma)
    else:
        got = environment_self_energy(low, pair, sigma)
    residues = screened_residues(low, pair)
    assert len(got) == len(residues) == 2
    for s, (space, (T, eps_sc)) in enumerate(zip(pair,
                                                 _rotations(mf, pair))):
        w = np.einsum('npq,pt,qr->ntr', canonical[s], T[:, space.act],
                      T[:, space.env], optimize=True)
        env = space.env
        args = (w, eps_sc[space.act], eps_sc[env], env < space.nocc,
                low.omega)
        if sigma == 'srg':
            want = srg_self_energy(*args, Numerics().downfolding.srg_s)
        else:
            want = averaged_self_energy(*args)
        err = np.abs(got[s] - want).max()
        print(f'{sigma} spin {s}: {err:.1e}, largest |Sigma| '
              f'{np.abs(want).max():.3e}')
        assert np.abs(want).max() > 1e-3
        assert err < 1e-10
        assert np.abs(residues[s] - w).max() < 1e-10


def _coupling(mf, space, s):
    """`fock_coupling` of spin s called directly, with the semicanonical
    rotation of its basis, both blocks."""
    eps = mf.mo_energy[s]
    T, _ = semicanonical_rotation(space.basis, space, eps)
    return fock_coupling(mf, mf.mo_coeff[s] @ T, T, eps, space,
                         blocks=COUPLING_BLOCKS, spin=s), T


def _bare_fock(mf, mo, s):
    """mo.T (hcore + J[D_a + D_b] - K[D_s]) mo of spin s, exact J and K."""
    dm = mf.make_rdm1()
    j, k = get_jk(mf.mol, dm)
    return mo.T @ (mf.get_hcore() + j[0] + j[1] - k[s]) @ mo


def test_coupling_per_spin(tight_uhf, uks, uks_nh):
    """Each spin's coupling from that spin's bare Fock operator, on OH (UHF
    and UKS-PBE0) and on triplet NH (UKS-PBE0), whose Sigma_x - v_xc is
    compared to 1e-9 (see `uks_nh`)."""
    refs = {'UHF': (tight_uhf, ACTIVE), 'UKS-PBE0': (uks, ACTIVE),
            'NH UKS-PBE0': (uks_nh, ACTIVE_NH)}
    for name, (ref, active) in refs.items():
        norb = ref.mo_coeff.shape[-1]
        # canonical: the identity and a permutation inside each block
        for s, n in enumerate(_nocc(ref)):
            occupations = np.where(np.arange(norb) < n, 1.0, 0.0)
            perm = np.concatenate([np.roll(np.arange(n), 1),
                                   n + np.arange(norb - n)[::-1]])
            for rotation in (np.eye(norb), np.eye(norb)[:, perm]):
                basis = OrbitalBasis(rotation, occupations, n,
                                     orbitals_source(ref.mo_coeff[s]))
                space = _spaces(ref, (basis, basis), active)[s]
                c, _ = _coupling(ref, space, s)
                assert np.all(c.b_particle == 0.0)
                assert np.all(c.b_hole == 0.0)
                assert c.norms == {'av_ev': 0.0, 'ao_i': 0.0}
        spaces = _spaces(ref, random_basis(SimpleNamespace(mf=ref)), active)
        for s, space in enumerate(spaces):
            c, T = _coupling(ref, space, s)
            nocc, eps, k = space.nocc, ref.mo_energy[s], space.n_act_occ
            sides = ((c.b_particle[k:], slice(nocc, None), space.virt_active,
                      space.env_virt),
                     (c.b_hole[:k], slice(None, nocc), space.occ_active,
                      space.env_occ))
            for b, block, act, env in sides:
                e = np.diag(eps[block])
                u_a, u_e = T[block, act], T[block, env]
                want = u_a.T @ e @ u_e @ u_e.T @ e @ u_a
                err = np.abs(b @ b.T - want).max()
                print(f'{name} spin {s}: B B^T against the rotated orbital '
                      f'energies {err:.1e}')
                assert err < 1e-9
            assert np.all(c.b_particle[:k] == 0.0)
            assert np.all(c.b_hole[k:] == 0.0)
            assert min(c.norms.values()) > 1e-3
            mo = ref.mo_coeff[s] @ T
            F = _bare_fock(ref, mo, s)
            ev, i = space.env_virt, space.env_occ
            assert np.abs(c.lam_virtual
                          - np.linalg.eigvalsh(F[np.ix_(ev, ev)])).max() < 1e-9
            assert np.abs(c.lam_occupied
                          - np.linalg.eigvalsh(F[np.ix_(i, i)])).max() < 1e-9
            if name == 'UKS-PBE0':
                continue
            if name == 'UHF':
                print(f'UHF spin {s} left_out: '
                      f'{max(c.left_out.values()):.1e}')
                assert max(c.left_out.values()) < 1e-7
                continue
            # NH: Sigma_x - v_xc of spin s
            X = F - mo.T @ ref.get_fock()[s] @ mo
            ao, av = space.occ_active, space.virt_active
            for part, (r, q) in {'av_ev': (av, ev), 'ao_i': (ao, i),
                                 'ao_ev': (ao, ev), 'i_av': (i, av),
                                 'i_ev': (i, ev)}.items():
                want = np.linalg.norm(X[np.ix_(r, q)])
                err = abs(c.left_out[part] - want)
                print(f'NH UKS spin {s} left_out {part}: {c.left_out[part]:.6e}'
                      f' (F_HF - F_KS {want:.6e}, diff {err:.1e})')
                assert want > 1e-3
                assert err < 1e-9
    # one coupling per spin: an unrestricted reference names its spin, a
    # restricted one has none
    mf = tight_uhf
    space = _spaces(mf, random_basis(SimpleNamespace(mf=mf)))[0]
    T, _ = semicanonical_rotation(space.basis, space, mf.mo_energy[0])
    with pytest.raises(ValueError, match='spin=0'):
        fock_coupling(mf, mf.mo_coeff[0] @ T, T, mf.mo_energy[0], space)
    rhf = water('hf')
    norb = len(rhf.mo_energy)
    with pytest.raises(ValueError, match='restricted'):
        fock_coupling(rhf, rhf.mo_coeff, np.eye(norb), rhf.mo_energy,
                      ActiveSpace.from_indices(norb, 5, [4], [5]), spin=0)


def test_hybridized_energy_equals_independent_dense(mf, low, low_if, pair):
    v_eff = _bare_v(low, pair)
    got = hybridized_correlation_energy(low, pair, v_eff)
    occ = [s.n_act_occ for s in pair]
    blocks = {(0, 0): v_eff[0], (0, 1): v_eff[1], (1, 1): v_eff[2],
              (1, 0): v_eff[1].transpose(2, 3, 0, 1)}
    ms, deltas = [], []
    for s, (space, (T, _)) in enumerate(zip(pair, _rotations(mf, pair))):
        nocc, eps = space.nocc, mf.mo_energy[s]
        o, u = T[:nocc, space.occ_active].T, T[nocc:, space.virt_active].T
        ms.append(np.einsum('ti,ua->tuia', o, u).reshape(len(o) * len(u),
                                                         -1))
        deltas.append((eps[nocc:][None, :] - eps[:nocc][:, None]).ravel())
    K = np.block([[ms[s].T @ blocks[s, t][:occ[s], occ[s]:, :occ[t], occ[t]:]
                   .reshape(len(ms[s]), len(ms[t])) @ ms[t]
                   for t in (0, 1)] for s in (0, 1)])
    want = dense_drpa_ec(np.concatenate(deltas), K, g=1)
    e_a = active_correlation_energy(low, pair, v_eff)
    on_if = hybridized_correlation_energy(low_if, pair, v_eff)
    print(f'E_c^(A,Delta) {got:.12f}, dense {want:.12f}, diff '
          f'{abs(got - want):.1e}; imagfrequency {on_if:.12f}, diff '
          f'{abs(on_if - got):.1e}; E_c^A {e_a:.10f}, E_c^(A,Delta) - '
          f'E_c^A {got - e_a:+.3e}')
    assert abs(got - want) < 1e-10
    assert abs(on_if - got) < 1e-8
    assert abs(got - e_a) > 1e-6


def test_default_is_refused(mf, low, low_hf, pair):
    for run in (low, low_hf):
        with pytest.raises(NotImplementedError,
                           match='unrestricted.*delta=False'):
            build_effective_hamiltonian(run, pair)
        h = _static(run, pair)
        assert 'delta_off' in _flags(h)
        assert h.hybridization is None
        assert 'fock_coupling' not in h.diagnostics
        assert h.energies.e_c_hybridization == 0.0
        assert h.energies.e0 == h.energies.e0_without_delta == h.g_eff[0]
        # a basis on one spin only stays refused
        with pytest.raises(ValueError, match='one spin'):
            build_effective_hamiltonian(run, (pair[0], replace(pair[1],
                                                               basis=None)),
                                        delta=False)
    # the natural orbitals of an unrestricted run do not exist
    with pytest.raises(NotImplementedError, match='unrestricted GW density'):
        run_low_level(mf, method='rpa', mode='casida', natural_orbitals=True)
    with pytest.raises(NotImplementedError, match='unrestricted GW density'):
        natural_orbital_basis(low)


def test_mrpa_with_two_channels(mf, low, pair):
    """The RPA of the decoupled active pairs of both spins, stacked, at the
    semicanonical energies with A + B = delta~_A + 2 (ia|jb)_eff and A - B
    = delta~_A (g = 1), has the zeroth moment (X+Y)(X+Y)^T of the two-spin
    RPA on the active pairs of the rotated bases."""
    vt = screening_correction(low, pair, 'mrpa')
    v_eff = tuple(v + t for v, t in zip(_bare_v(low, pair), vt))
    rotations = _rotations(mf, pair)
    delta, k_eff = _stacked_ph(pair, [e for _, e in rotations], v_eff)
    _, xpy_a = _modes(np.diag(delta) + 2.0 * k_eff, delta)
    R = block_diag(*[_pair_rotation(T, s.nocc)
                     for s, (T, _) in zip(pair, rotations)])
    xpy = R.T @ low.xpy
    mask = np.concatenate([active_pair_mask(s) for s in pair])
    want = (xpy @ xpy.T)[np.ix_(mask, mask)]
    err = np.abs(xpy_a @ xpy_a.T - want).max()
    print(f'mrpa zeroth moment of the stacked active pairs: {err:.1e}')
    assert max(np.abs(t).max() for t in vt) > 1e-3
    assert err < 1e-10
    # the projection of the stored modes is the same matrix
    e = block_diag(*[active_pair_projection(
        SimpleNamespace(rotation=T, nocc=s.nocc, active_space=s))
        for s, (T, _) in zip(pair, rotations)])
    assert np.abs((e.T @ low.xpy) @ (e.T @ low.xpy).T - want).max() < 1e-10


def test_closed_shell_limit():
    """Water (RHF) and the same orbitals as a UHF (pyscf's convert_to_uhf,
    bitwise the RHF orbitals in both spins), with the restricted natural
    orbitals' rotation on both spins, each spin's basis with its own
    source: every energy and every spin block (so their spin sums too)
    equal the restricted result, both built with delta=False, and so does
    the FCI energy of H_eff."""
    rhf = water('hf')
    uhf = addons.convert_to_uhf(rhf)
    assert all(np.array_equal(c, rhf.mo_coeff) for c in uhf.mo_coeff)
    low_r = run_low_level(rhf, method='rpa', mode='casida',
                          natural_orbitals=True)
    low_u = run_low_level(uhf, method='rpa', mode='casida')
    space = no_space(low_r, 2, 3, rotate_occupied=True)
    pair = tuple(replace(space, basis=replace(
        space.basis, source=orbitals_source(c))) for c in uhf.mo_coeff)
    assert pair[0].basis.digest == pair[1].basis.digest == space.basis.digest
    for opts in (dict(screening='crpa', sigma='srg'),
                 dict(screening='mrpa')):
        want = _static(low_r, space, **opts)
        got = _static(low_u, pair, **opts)
        errs = {}
        for field in dataclasses.fields(Energies):
            if field.name not in RECORDS:
                errs[field.name] = abs(getattr(got.energies, field.name)
                                       - getattr(want.energies, field.name))
        errs['g_eff[0]'] = abs(got.g_eff[0] - want.g_eff[0])
        errs['h'] = max(np.abs(h - want.g_eff[1]).max() for h in got.g_eff[1])
        errs['v'] = max(np.abs(v - want.g_eff[2]).max() for v in got.g_eff[2])
        e_u, e_r = got.solve('pyscf_fci')[0], want.solve('pyscf_fci')[0]
        errs['fci'] = abs(e_u - e_r)
        print(f'{opts}: largest difference {max(errs.values()):.1e} '
              f'({max(errs, key=errs.get)})')
        for name, err in errs.items():
            assert err < 1e-9, name
