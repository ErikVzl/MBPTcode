"""Mean fields and settings shared by the tests of the natural-orbital
downfolding (tests/test_downfolding_no_*.py), the brute-force checks of
their residual screening (`brute_force_vt`) and residual polarizability
(`brute_force_pr`), and the dense oracle of the RPA correlation energy of a
hybridized active space (`dense_drpa_ec`)."""
import numpy as np
import scipy.linalg
from pyscf import dft, scf
from scipy.linalg import cho_factor, cho_solve

from src.Base.numerics.numerics import Numerics
from src.EffectiveHamiltonian.active_space import ActiveSpace
from src.EffectiveHamiltonian.orbital_basis import natural_orbital_basis
from tests.downfolding_oracle import water as _water_hf
from tests.gw_density_routes import mean_field as _nh3_mean_field

#: the ISDF grid of the space-time cases
G1 = Numerics(isdf={'level': 'G1'})


def water(xc='hf', df=False):
    """Water/6-31G on the geometry of tests/test_downfolding_routes.py: RHF
    ('hf') or RKS with functional `xc`. `df` fits the MEAN FIELD's integrals
    (pyscf's default auxiliary basis); the `df` option of the RPA run is
    another thing."""
    mol = _water_hf().mol
    mf = scf.RHF(mol) if xc == 'hf' else dft.RKS(mol, xc=xc)
    if df:
        mf = mf.density_fit()
    mf.conv_tol = 1e-11
    mf.verbose = 0
    return mf.run()


def ammonia(xc='hf'):
    """NH3/cc-pVDZ, density fitted on cc-pvdz-ri, RHF ('hf') or RKS with
    `xc` ('pbe0'), as the space-time cases of tests/test_downfolding_routes.py."""
    return _nh3_mean_field('HF' if xc == 'hf' else xc.upper())


def no_space(low, n_occ, n_virt, rotate_occupied=False):
    """The ActiveSpace of `low`'s natural orbitals (`natural_orbital_basis`)
    with n_occ active occupied and n_virt active virtual orbitals: the n_occ
    highest canonical occupied orbitals (with rotate_occupied, the n_occ
    occupied NOs with the largest holes) and the n_virt virtual NOs of
    largest occupation. Each rotated block is in order of descending
    occupation, so both are the last n_occ occupied and the first n_virt
    virtual orbitals of the basis."""
    basis = natural_orbital_basis(low, rotate_occupied=rotate_occupied)
    nocc, norb = basis.nocc, len(basis.occupations)
    return ActiveSpace.from_indices(norb, nocc, range(nocc - n_occ, nocc),
                                    range(nocc, nocc + n_virt), basis=basis)


def brute_force_vt(b, eps, nocc, T, space):
    """vt = W^r(0) - v by its definition: the active ph pairs deleted from A + B in the
    rotated pair basis (non-diagonal one-body part), one Cholesky. b (naux, nmo, nmo) canonical."""
    nvirt = len(eps) - nocc
    b_ov = b[:, :nocc, nocc:].reshape(len(b), -1)
    delta = (eps[nocc:][None, :] - eps[:nocc][:, None]).ravel()
    R = np.einsum('ij,ab->iajb', T[:nocc, :nocc], T[nocc:, nocc:]).reshape(nocc * nvirt, -1)
    apb = R.T @ (np.diag(delta) + 4.0 * b_ov.T @ b_ov) @ R
    active = np.zeros((nocc, nvirt), bool)
    active[np.ix_(space.occ_active, space.virt_active - nocc)] = True
    r = ~active.ravel()
    b_rot = np.einsum('Qpq,pi,qj->Qij', b, T, T)
    b_act = b_rot[:, space.act][:, :, space.act].reshape(len(b), -1)
    v_r = b_rot[:, :nocc, nocc:].reshape(len(b), -1)[:, r].T @ b_act
    vt = -4.0 * v_r.T @ cho_solve(cho_factor(apb[np.ix_(r, r)]), v_r)
    return vt.reshape((len(space.act),) * 4)


def brute_force_pr(b_ov, eps, nocc, T, space, g=2):
    """P^r(0) = -2g B_r^T Lambda_rr^-1 B_r of one spin channel (two channels: g = 1 each,
    summed) by a dense solve over the residual pairs
    of the rotated basis (every pair but active occupied x active virtual)."""
    nvirt = len(eps) - nocc
    F = T.T @ np.diag(eps) @ T
    lam = (np.einsum('ab,ij->iajb', F[nocc:, nocc:], np.eye(nocc))
           - np.einsum('ij,ab->iajb', F[:nocc, :nocc], np.eye(nvirt))).reshape(nocc * nvirt, -1)
    b_rot = np.einsum('Qjb,ji,ba->Qia', b_ov, T[:nocc, :nocc], T[nocc:, nocc:]).reshape(len(b_ov), -1)
    active = np.zeros((nocc, nvirt), bool)
    active[np.ix_(space.occ_active, space.virt_active - nocc)] = True
    r = ~active.ravel()
    return -2.0 * g * b_rot[:, r] @ np.linalg.solve(lam[np.ix_(r, r)], b_rot[:, r].T)


def dense_drpa_ec(delta, k, g=2):
    """E_c of the direct RPA with A = diag(delta) + g k, B = g k (g = 2: one closed-shell
    channel; g = 1: the pairs of both spins stacked), from the positive
    eigenvalues of [[A, B], [-B, -A]]: (sum omega - tr A) / 2."""
    a, b = np.diag(delta) + g * k, g * k
    w = scipy.linalg.eig(np.block([[a, b], [-b, -a]]), right=False)
    assert np.abs(w.imag).max() < 1e-10
    return 0.5 * (np.sort(w.real)[len(delta):].sum() - np.trace(a))
