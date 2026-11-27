"""Mean fields and settings shared by the tests of the natural-orbital
downfolding (tests/test_downfolding_no_*.py), a random rotated basis that
stands in for natural orbitals where there are none (`random_basis`, e.g.
an unrestricted reference), the brute-force checks of their residual
screening (`brute_force_vt`) and residual polarizability (`brute_force_pr`),
and the dense oracle of the RPA correlation energy of a hybridized active
space (`dense_drpa_ec`), N2 (`nitrogen`), whose pi and pi* orbitals give
degenerate states, benzene and its pi NO CAS(6,6) (`benzene`,
`benzene_pi_space`), and the explicit P + Q matrix of the Delta solvers
(`pq_spectrum`)."""
import math

import numpy as np
import scipy.linalg
from pyscf import dft, fci, gto, scf
from scipy.linalg import block_diag, cho_factor, cho_solve

from src.Base.numerics.constants import DOWNFOLDING_NO_CUT_WARN
from src.Base.numerics.numerics import Numerics
from src.EffectiveHamiltonian.active_space import ActiveSpace
from src.EffectiveHamiltonian.orbital_basis import (OrbitalBasis,
                                                    natural_orbital_basis,
                                                    orbitals_source)
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


def random_basis(low, seed=0, rotate_occupied=True):
    """An `OrbitalBasis` of the mean field of `low` that is a fixed random
    rotation inside the occupied and inside the virtual orbitals (the
    occupied block the identity unless rotate_occupied), for a test that
    needs a rotated basis and has no natural orbitals; the pair (alpha,
    beta), each on that spin's orbitals and with its own rotation, for an
    unrestricted reference. The occupations are a strictly descending ramp
    in each block, g - k s in the occupied and (nvirt - k) s in the virtual
    block (g = 2 restricted, 1 per spin), with the step s = 2.5
    DOWNFOLDING_NO_CUT_WARN, so no cut through either block comes near the
    warning of the cut check."""
    mf, rng = low.mf, np.random.default_rng(seed)
    if isinstance(mf, scf.uhf.UHF):
        return tuple(_random_basis_of(rng, c, occ, 1.0, rotate_occupied)
                     for c, occ in zip(mf.mo_coeff, mf.mo_occ))
    return _random_basis_of(rng, mf.mo_coeff, mf.mo_occ, 2.0, rotate_occupied)


def _random_basis_of(rng, mo_coeff, mo_occ, g, rotate_occupied):
    """`random_basis` of one spin channel."""
    norb, nocc = len(mo_occ), int(np.count_nonzero(mo_occ))
    step = 2.5 * DOWNFOLDING_NO_CUT_WARN
    rotation = np.eye(norb)
    if rotate_occupied:
        rotation[:nocc, :nocc] = np.linalg.qr(rng.normal(size=(nocc, nocc)))[0]
    rotation[nocc:, nocc:] = np.linalg.qr(
        rng.normal(size=(norb - nocc, norb - nocc)))[0]
    occupations = np.concatenate([g - step * np.arange(1, nocc + 1),
                                  step * np.arange(norb - nocc, 0, -1)])
    return OrbitalBasis(rotation, occupations, nocc,
                        orbitals_source(mo_coeff))


def brute_force_vt(b, eps, nocc, T, space):
    """vt = W^r(0) - v by its definition: the active ph pairs deleted from A + B in the
    rotated pair basis (non-diagonal one-body part), one Cholesky. b (naux, nmo, nmo) canonical.

    One spin channel (restricted, g = 2): b, eps, nocc, T and space of it,
    A + B = R^T (delta + 4K) R, and vt (n_act,)*4. Two (unrestricted, g = 1
    each): every argument the pair (alpha, beta), the pairs of both spins
    stacked, A + B = R^T (delta + 2K) R with K = B_ov^T B_ov coupling them
    and R block diagonal in the spins, and the blocks (aa, ab, bb),
    vt^{st} = -2 V_s^T [(A+B)_rr]^-1 V_t."""
    one = not isinstance(space, (tuple, list))
    chans = list(zip(*[(x,) if one else x for x in (b, eps, nocc, T, space)]))
    g = 2.0 if one else 1.0
    b_ov, delta, rots, res, b_res, b_act = [], [], [], [], [], []
    for b_s, eps_s, n_s, T_s, sp in chans:
        nvirt = len(eps_s) - n_s
        b_ov.append(b_s[:, :n_s, n_s:].reshape(len(b_s), -1))
        delta.append((eps_s[n_s:][None, :] - eps_s[:n_s][:, None]).ravel())
        rots.append(np.einsum('ij,ab->iajb', T_s[:n_s, :n_s],
                              T_s[n_s:, n_s:]).reshape(n_s * nvirt, -1))
        active = np.zeros((n_s, nvirt), bool)
        active[np.ix_(sp.occ_active, sp.virt_active - n_s)] = True
        res.append(~active.ravel())
        b_rot = np.einsum('Qpq,pi,qj->Qij', b_s, T_s, T_s)
        b_act.append(b_rot[:, sp.act][:, :, sp.act].reshape(len(b_s), -1))
        b_res.append(b_rot[:, :n_s, n_s:].reshape(len(b_s), -1)[:, res[-1]])
    b_ov, R, r = np.hstack(b_ov), block_diag(*rots), np.concatenate(res)
    apb = R.T @ (np.diag(np.concatenate(delta)) + 2.0 * g * b_ov.T @ b_ov) @ R
    v_r = [np.hstack(b_res).T @ b for b in b_act]
    factor = cho_factor(apb[np.ix_(r, r)])
    n = len(chans[0][4].act)
    blocks = [(-2.0 * g * v_r[s].T @ cho_solve(factor, v_r[t])).reshape(
        (n,) * 4) for s, t in (((0, 0),) if one else ((0, 0), (0, 1), (1, 1)))]
    return blocks[0] if one else tuple(blocks)


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


def nitrogen():
    """N2/6-31G at 1.098 Angstrom, RHF with density fitting (pyscf's default
    auxiliary basis): a molecule whose pi and pi* orbitals are degenerate
    pairs, so that its active spaces hold doubly degenerate states."""
    mol = gto.M(atom='N 0 0 0; N 0 0 1.098', basis='6-31g', verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-11
    mf.verbose = 0
    return mf.run()


def benzene():
    """Benzene/cc-pVDZ, D6h with r(CC) = 1.397 and r(CH) = 1.084 Angstrom in
    the xy plane, RHF with density fitting (pyscf's default auxiliary
    basis): the molecule of the benzene handoff, whose pi system gives an
    NO CAS(6,6) with degenerate states (`benzene_pi_space`)."""
    atoms = []
    for k in range(6):
        a = math.pi / 3 * k
        atoms.append(('C', (1.397 * math.cos(a), 1.397 * math.sin(a), 0.0)))
        r = 1.397 + 1.084
        atoms.append(('H', (r * math.cos(a), r * math.sin(a), 0.0)))
    mol = gto.M(atom=atoms, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-11
    mf.verbose = 0
    return mf.run()


def benzene_pi_space(low, rotate_occupied=False):
    """The NO CAS(6,6) of the pi system of benzene (`benzene`): the three
    occupied orbitals of the basis with a weight above 0.9 on the pz, dxz
    and dyz functions (with rotate_occupied, occupied NOs) and the three
    virtual NOs of largest occupation, which are the pi* ones."""
    basis = natural_orbital_basis(low, rotate_occupied=rotate_occupied)
    mf, nocc = low.mf, basis.nocc
    s = mf.mol.intor('int1e_ovlp')
    pz = np.array([('pz' in lab) or ('dxz' in lab) or ('dyz' in lab)
                   for lab in mf.mol.ao_labels()])
    c = mf.mo_coeff @ basis.rotation
    weight = np.einsum('pi,pi->i', c[pz], (s @ c)[pz])
    occ_pi = [i for i in range(nocc) if weight[i] > 0.9]
    return ActiveSpace.from_indices(len(basis.occupations), nocc, occ_pi,
                                    range(nocc, nocc + 3), basis=basis)


#: the pyscf operator that takes an electron of spin sigma out (particle
#: side) or puts one in (hole side)
_OPERATORS = {('particle', 0): fci.addons.des_a,
              ('particle', 1): fci.addons.des_b,
              ('hole', 0): fci.addons.cre_a, ('hole', 1): fci.addons.cre_b}


def _fci_matrix(h1, eri, norb, nelec):
    """The dense FCI matrix of the sector nelec = (n_alpha, n_beta), pyscf's
    determinant order, column by column from direct_spin1.contract_2e."""
    shape = (math.comb(norb, nelec[0]), math.comb(norb, nelec[1]))
    n = shape[0] * shape[1]
    h2e = fci.direct_spin1.absorb_h1e(h1, eri, norb, nelec, 0.5)
    out = np.empty((n, n))
    for k in range(n):
        unit = np.zeros(n)
        unit[k] = 1.0
        out[:, k] = fci.direct_spin1.contract_2e(
            h2e, unit.reshape(shape), norb, nelec).ravel()
    return out


def _operator_matrix(op, norb, nelec, u):
    """The matrix of one pyscf creation or annihilation operator on orbital
    u, from the sector nelec to the one it reaches."""
    shape = (math.comb(norb, nelec[0]), math.comb(norb, nelec[1]))
    n = shape[0] * shape[1]
    cols = []
    for k in range(n):
        unit = np.zeros(n)
        unit[k] = 1.0
        cols.append(op(unit.reshape(shape), norb, nelec, u).ravel())
    return np.array(cols).T


def pq_spectrum(h1, eri, norb, nelec, sides, with_vectors=False):
    """(energies, p_weights): the eigenvalues of the explicit P + Q matrix
    of an active space coupled to environment levels, and the weight of each
    eigenvector in P; with_vectors, also the P part of each eigenvector,
    (dim P, n) in pyscf's determinant order, unnormalized.

    P is the FCI sector nelec of (h1, eri). `sides` are those of
    `Coupling.sides()`, (kind, sigma, lam, b), sigma None for both spins.
    Each kind, spin and level x adds the block H_M + lam_x (particle: the
    sector with one electron of that spin less) or H_M - lam_x (hole: one
    more), coupled to P by C = sum_u b[u, x] A_u, A_u the matrix of a_u
    (particle) or a+_u (hole):

        [[H_N, C^T], [C, blockdiag(H_M +- lam_x)]]

    Built with pyscf directly, nothing of the Delta solvers."""
    h_n = _fci_matrix(h1, eri, norb, nelec)
    couplings, blocks = [], []
    for kind, sigma, lam, b in sides:
        for s in ((0, 1) if sigma is None else (sigma,)):
            target = list(nelec)
            target[s] += -1 if kind == 'particle' else 1
            if not 0 <= target[s] <= norb:
                continue
            ops = [_operator_matrix(_OPERATORS[kind, s], norb, nelec, u)
                   for u in range(norb)]
            h_m = _fci_matrix(h1, eri, norb, tuple(target))
            sign = 1.0 if kind == 'particle' else -1.0
            for x in range(len(lam)):
                couplings.append(sum(b[u, x] * ops[u] for u in range(norb)))
                blocks.append(h_m + sign * lam[x] * np.eye(len(h_m)))
    c = np.vstack(couplings)
    full = np.block([[h_n, c.T], [c, block_diag(*blocks)]])
    energies, vectors = np.linalg.eigh(full)
    p_parts = vectors[:len(h_n)]
    if with_vectors:
        return energies, (p_parts ** 2).sum(0), p_parts
    return energies, (p_parts ** 2).sum(0)
