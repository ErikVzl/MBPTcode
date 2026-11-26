"""The orbital basis of natural orbitals and the ActiveSpace that indexes it:
the rotation is block diagonal and orthogonal, each rotated block is ordered
by occupation, the semicanonical rotation diagonalizes the orbital energies
inside every index set, and the basis enters the key of an ActiveSpace."""
import numpy as np
import pytest
from pyscf import gto, scf

from src.EffectiveHamiltonian.Downfolding import run_low_level
from src.EffectiveHamiltonian.active_space import ActiveSpace
from src.EffectiveHamiltonian.orbital_basis import (
    natural_orbital_basis, semicanonical_rotation)
from tests.no_downfolding import water


@pytest.fixture(scope='module')
def low():
    return run_low_level(water('hf', df=True), method='rpa', mode='casida',
                         natural_orbitals=True)


@pytest.fixture(scope='module')
def space(low):
    nocc, norb = low.nocc, len(low.eps)
    return ActiveSpace.from_indices(norb, nocc, [nocc - 2, nocc - 1],
                                    [nocc, nocc + 1, nocc + 2])


def test_rotation_is_block_diagonal_and_orthogonal(low):
    for rot in (False, True):
        basis = natural_orbital_basis(low, rotate_occupied=rot)
        r, nocc = basis.rotation, basis.nocc
        assert nocc == low.nocc
        assert np.abs(r.T @ r - np.eye(len(r))).max() < 1e-12
        assert np.all(r[:nocc, nocc:] == 0) and np.all(r[nocc:, :nocc] == 0)


def test_occupations_descend_in_each_rotated_block(low):
    nocc = low.nocc
    nat = low.natural
    full = natural_orbital_basis(low, rotate_occupied=True)
    assert np.all(np.diff(full.occupations[:nocc]) <= 0)
    assert np.all(np.diff(full.occupations[nocc:]) <= 0)
    part = natural_orbital_basis(low)
    assert np.all(np.diff(part.occupations[nocc:]) <= 0)
    assert np.array_equal(part.occupations[:nocc], np.diag(nat.gamma_oo))
    # the occupied diagonal is in canonical order, not sorted
    assert np.any(np.diff(part.occupations[:nocc]) > 0)


def test_occupied_block_is_identity_unless_rotated(low):
    nocc = low.nocc
    part = natural_orbital_basis(low)
    assert np.array_equal(part.rotation[:nocc, :nocc], np.eye(nocc))
    full = natural_orbital_basis(low, rotate_occupied=True)
    assert not np.array_equal(full.rotation[:nocc, :nocc], np.eye(nocc))


def test_source_and_digest(low):
    a, b = natural_orbital_basis(low), natural_orbital_basis(low)
    assert a.source == b.source and a.digest == b.digest
    assert a.digest != natural_orbital_basis(low, rotate_occupied=True).digest
    assert len(a.source) == 64


def test_table_lists_every_orbital(low):
    lines = natural_orbital_basis(low).table().splitlines()
    assert len(lines) >= len(low.eps)


def test_semicanonical_blocks_are_diagonal(low, space):
    basis = natural_orbital_basis(low)
    sp = ActiveSpace.from_indices(space.norb, space.nocc, space.occ_active,
                                  space.virt_active, basis=basis)
    T, e = semicanonical_rotation(basis, sp, low.eps)
    assert np.abs(T.T @ T - np.eye(len(T))).max() < 1e-12
    F = T.T @ np.diag(low.eps) @ T
    for idx in sp.as_tuple():
        off = F[np.ix_(idx, idx)] - np.diag(e[idx])
        assert np.abs(off).max() < 1e-10
        assert np.all(np.diff(e[idx]) >= 0)
    assert np.abs(e - np.diag(F)).max() < 1e-12
    Pno = basis.rotation[:, sp.virt_active] @ basis.rotation[:, sp.virt_active].T
    Psc = T[:, sp.virt_active] @ T[:, sp.virt_active].T
    assert np.abs(Pno - Psc).max() < 1e-10


def test_equality_includes_the_basis(low, space):
    basis = natural_orbital_basis(low)
    sp = ActiveSpace.from_indices(space.norb, space.nocc, space.occ_active,
                                  space.virt_active, basis=basis)
    same = ActiveSpace(space.norb, space.nocc, space.occ_active,
                       space.virt_active, basis)
    assert sp != space and hash(sp) != hash(space)
    assert sp == same and hash(sp) == hash(same)
    assert space._key()[-1] is None and sp._key()[-1] == basis.digest


def test_missing_density_names_the_option():
    low = run_low_level(water('hf', df=True), method='rpa', mode='casida')
    with pytest.raises(ValueError, match='natural_orbitals=True'):
        natural_orbital_basis(low)


def test_natural_orbitals_of_an_unrestricted_run_are_refused():
    mol = gto.M(atom='O 0 0 0; H 0 0.75 0.6', basis='6-31g', spin=1,
                verbose=0)
    low = run_low_level(scf.UHF(mol).run(), method='rpa', mode='casida')
    with pytest.raises(NotImplementedError, match='unrestricted GW density'):
        natural_orbital_basis(low)
