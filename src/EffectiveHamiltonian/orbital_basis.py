"""Orbital bases of the downfolding that are not the canonical orbitals: the
blocks of the GW density that natural orbitals are built from, the basis of
natural orbitals itself, and the semicanonical rotation that makes the Fock
matrix diagonal again inside each block of an active space.
"""
import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NaturalDensity:
    """The occupied-occupied and virtual-virtual blocks of the spatial RPA
    (GW) density in the canonical MO basis of the mean field, and the size of
    the occupied-virtual block, which natural orbitals leave out.

    gamma_oo: (nocc, nocc), with the reference's 2 on the diagonal.
    gamma_vv: (nvirt, nvirt).
    ov_max, ov_fro: the largest element and the Frobenius norm of the
        occupied-virtual block.
    """
    gamma_oo: np.ndarray
    gamma_vv: np.ndarray
    ov_max: float
    ov_fro: float


def _sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


@dataclass(frozen=True)
class OrbitalBasis:
    """An orbital basis of one spin channel that is a rotation of the
    canonical MOs inside the occupied block and inside the virtual block.

    rotation: (norb, norb); column k is orbital k in the canonical MO basis.
        The occupied block sits in columns [0, nocc), and the matrix is block
        diagonal (nothing mixes an occupied with a virtual orbital).
    occupations: (norb,), the occupation of each orbital.
    nocc: the number of occupied orbitals.
    source: sha256 of the bytes of the `mo_coeff` of this spin channel, the
        orbitals `rotation` refers to.
    """
    rotation: np.ndarray
    occupations: np.ndarray
    nocc: int
    source: str

    @property
    def digest(self):
        """sha256 of the bytes of `rotation` and of `nocc`."""
        h = hashlib.sha256(np.ascontiguousarray(self.rotation).tobytes())
        h.update(str(int(self.nocc)).encode())
        return h.hexdigest()

    def table(self):
        """One line per orbital: index (0-based), molden number (index + 1),
        block, occupation."""
        lines = ['index  molden  block  occupation']
        for k, n in enumerate(self.occupations):
            block = 'occ' if k < self.nocc else 'virt'
            lines.append(f'{k:5d}  {k + 1:6d}  {block:5s}  {n:10.6f}')
        return '\n'.join(lines)


def natural_orbital_basis(low, rotate_occupied=False):
    """The `OrbitalBasis` of the natural orbitals of the GW density stored in
    `low.natural`: each rotated block in order of descending occupation.

    With rotate_occupied=False the occupied block is the identity, so the
    occupied orbitals stay canonical and in canonical order, and their
    occupations are the diagonal of gamma_oo, unsorted.

    NotImplementedError for an unrestricted run (there is no unrestricted GW
    density); ValueError when the run was made without natural_orbitals=True.
    """
    if low.unrestricted:
        raise NotImplementedError(
            "natural orbitals need the GW density of a restricted reference; "
            "there is no unrestricted GW density, so the natural orbitals of "
            "a UHF/UKS run do not exist.")
    if low.natural is None:
        raise ValueError("the low-level run holds no GW density: run it with "
                         "natural_orbitals=True.")
    nat, nocc = low.natural, int(low.nocc)
    norb = nocc + nat.gamma_vv.shape[0]
    rotation = np.zeros((norb, norb))
    occupations = np.zeros(norb)

    def descending(gamma):
        n, v = np.linalg.eigh(gamma)
        return n[::-1], v[:, ::-1]

    if rotate_occupied:
        occupations[:nocc], rotation[:nocc, :nocc] = descending(nat.gamma_oo)
    else:
        occupations[:nocc] = np.diag(nat.gamma_oo)
        rotation[:nocc, :nocc] = np.eye(nocc)
    occupations[nocc:], rotation[nocc:, nocc:] = descending(nat.gamma_vv)
    return OrbitalBasis(rotation, occupations, nocc,
                        _sha256(low.mf.mo_coeff))


def semicanonical_rotation(basis, active_space, eps):
    """(T, eps_sc): the rotation that is semicanonical in every index set of
    the active space.

    For each of env_occ, occ_active, virt_active and env_virt the columns of
    T at those indices span the same space as the columns of basis.rotation
    there, and diagonalize diag(eps) in it, ascending. T (norb, norb) is
    orthogonal; eps_sc[k] = (T.T @ diag(eps) @ T)[k, k].
    """
    rotation, eps = basis.rotation, np.asarray(eps)
    T = np.zeros_like(rotation)
    for idx in (active_space.env_occ, active_space.occ_active,
                active_space.virt_active, active_space.env_virt):
        if len(idx) == 0:
            continue
        block = rotation[:, idx]
        _, u = np.linalg.eigh(block.T @ (eps[:, None] * block))
        T[:, idx] = block @ u
    eps_sc = np.einsum('pk,p,pk->k', T, eps, T)
    return T, eps_sc
