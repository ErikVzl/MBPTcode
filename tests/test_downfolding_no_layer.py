"""A rotated orbital basis in the general layer of the downfolding: method
'hf' is CASCI on the rotated orbitals, the bare RPA takes the semicanonical
active pairs, an identity basis is the canonical route, the cut of an active
space is checked on the occupations of a rotated block (refused below
QP_DEGENERACY_TOL, warned below no_cut_warn) and on the orbital energies of
an unrotated one, and a basis of other orbitals, exact integrals, the QP
energies and QDPT's permuted mean field are refused; what a later task
builds is refused by its name. An unrestricted reference on a rotated basis
is tests/test_downfolding_no_two_channels.py."""
import warnings
from dataclasses import replace

import numpy as np
import pytest
from pyscf import gto, mcscf, scf

from src.Base.numerics.constants import (DOWNFOLDING_NO_CUT_WARN,
                                         QP_DEGENERACY_TOL)
from src.Base.numerics.numerics import Numerics, NumericsMismatch
from src.EffectiveHamiltonian.Downfolding import (build_effective_hamiltonian,
                                                  run_low_level)
from src.EffectiveHamiltonian.Downfolding.hybridization import (
    DeltaOffWarning)
from src.EffectiveHamiltonian.Downfolding.methods import rpa
from src.EffectiveHamiltonian.active_space import (ActiveSpace,
                                                   NearDegenerateCutWarning,
                                                   ReorderedMF)
from src.EffectiveHamiltonian.orbital_basis import (OrbitalBasis,
                                                    natural_orbital_basis,
                                                    orbitals_source,
                                                    semicanonical_rotation)
from tests.downfolding_oracle import rpa_energy
from tests.no_downfolding import no_space, water


@pytest.fixture(scope='module')
def mf():
    return water('hf')


@pytest.fixture(scope='module')
def low(mf):
    return run_low_level(mf, method='rpa', mode='casida',
                         natural_orbitals=True)


@pytest.fixture(scope='module')
def low_hf(mf):
    return run_low_level(mf, method='hf')


def _flags(h):
    return [f['flag'] for f in h.numerical_flags]


def _with_occupations(space, changes):
    """`space` on its basis with the occupations of `changes` ({index:
    occupation}) replaced: the rotation, and so every orbital, unchanged."""
    occ = space.basis.occupations.copy()
    for k, n in changes.items():
        occ[k] = n
    return replace(space, basis=replace(space.basis, occupations=occ))


def test_hf_method_is_casci_on_the_rotated_orbitals(mf, low, low_hf):
    # a rotated basis gets Delta by default: CASCI is the static Hamiltonian
    space = no_space(low, 2, 2)
    with pytest.warns(DeltaOffWarning):
        h = build_effective_hamiltonian(low_hf, space, delta=False)
    T, _ = semicanonical_rotation(space.basis, space, low.eps)
    order = list(space.env_occ) + list(space.act) + list(space.env_virt)
    ref = mcscf.CASCI(mf, 4, 4).kernel((mf.mo_coeff @ T)[:, order])[0]
    assert abs(h.solve('pyscf_fci')[0] - ref) < 1e-9
    assert h.space['basis'] is space.basis
    assert np.array_equal(h.space['mo'], mf.mo_coeff @ T)


def test_bare_rpa_uses_semicanonical_active_pairs(low):
    space = no_space(low, 2, 2)
    h = build_effective_hamiltonian(low, space, screening='bare')
    T, eps_sc = semicanonical_rotation(space.basis, space, low.eps)
    act = space.act
    b = np.einsum('Qpq,pt,qu->Qtu', low.factors, T[:, act], T[:, act])
    v = np.einsum('Qtu,Qvw->tuvw', b, b)
    assert abs(h.energies.e_c_active
               - rpa_energy(v, eps_sc[act], space.n_act_occ)) < 1e-10
    assert h.energies.e_c_full == low.e_c_full


def test_rotated_channel_is_built_once_and_kept(low):
    space = no_space(low, 2, 3)
    (chan,) = rpa._channels(low, space)
    T, eps_sc = semicanonical_rotation(space.basis, space, low.eps)
    assert chan.factors is None and chan.canonical.factors is low.factors
    assert np.array_equal(chan.rotation, T) and np.array_equal(chan.eps,
                                                               eps_sc)
    assert chan.rotated_factors.shape == (low.factors.shape[0], space.n_act,
                                          len(low.eps))
    want = np.einsum('Qpq,pt,qj->Qtj', low.factors, T[:, space.act], T)
    assert np.abs(chan.rotated_factors - want).max() < 1e-12
    assert rpa._channels(low, space)[0] is chan
    # an equal space, built again, finds the same channel
    again = ActiveSpace.from_indices(space.norb, space.nocc, space.occ_active,
                                     space.virt_active, basis=space.basis)
    assert rpa._channels(low, again)[0] is chan
    assert rpa._channels(low, no_space(low, 2, 2))[0] is not chan


def test_identity_basis_reproduces_canonical(mf, low):
    nocc, norb = low.nocc, len(low.eps)
    occupations = np.where(np.arange(norb) < nocc, 2.0, 0.0)
    basis = OrbitalBasis(np.eye(norb), occupations, nocc,
                         orbitals_source(mf.mo_coeff))
    canonical = ActiveSpace.from_indices(norb, nocc, [3, 4], [5, 6])
    identity = replace(canonical, basis=basis)
    want = build_effective_hamiltonian(low, canonical, screening='bare')
    got = build_effective_hamiltonian(low, identity, screening='bare')
    assert abs(got.g_eff[0] - want.g_eff[0]) < 1e-12
    for k in (1, 2):
        assert np.abs(got.g_eff[k] - want.g_eff[k]).max() < 1e-12
    for name in ('e_core', 'e_hartree', 'e_exchange', 'e_c_full',
                 'e_c_active'):
        assert abs(getattr(got.energies, name)
                   - getattr(want.energies, name)) < 1e-12
    assert 'basis' not in want.space and 'mo' not in want.space


def test_degenerate_occupation_cut_is_refused(low, low_hf):
    nocc = low.nocc
    space = _with_occupations(no_space(low, 2, 2), {nocc + 1: 0.0123,
                                                    nocc + 2: 0.0123})
    with pytest.raises(ValueError, match=r'0\.0123.*0\.0123'):
        build_effective_hamiltonian(low_hf, space)
    # just below the refusal and on the occupied side of a rotated block
    nearly = 0.9 * QP_DEGENERACY_TOL
    space = _with_occupations(no_space(low, 2, 2, rotate_occupied=True),
                              {nocc - 2: 1.96, nocc - 3: 1.96 + nearly})
    with pytest.raises(ValueError, match='occupation'):
        build_effective_hamiltonian(low_hf, space)


@pytest.mark.filterwarnings(
    'ignore::src.EffectiveHamiltonian.Downfolding.hybridization.'
    'DeltaOffWarning')
def test_near_degenerate_cut_warns_and_runs(low):
    # a rotated occupied block is refused with Delta on until Task 13: those
    # spaces are built with delta=False
    nocc = low.nocc
    virt = no_space(low, 2, 2)
    occ = no_space(low, 2, 2, rotate_occupied=True)
    near = [_with_occupations(virt, {nocc + 1: 0.0411, nocc + 2: 0.0399}),
            _with_occupations(occ, {nocc - 2: 1.9589, nocc - 3: 1.9601})]
    apart = [_with_occupations(virt, {nocc + 1: 0.0499, nocc + 2: 0.0400}),
             _with_occupations(occ, {nocc - 2: 1.9501, nocc - 3: 1.9600})]
    for space, (a, b) in zip(near, (('0.0411', '0.0399'),
                                    ('1.9589', '1.9601'))):
        with pytest.warns(NearDegenerateCutWarning,
                          match=f'{a}.*{b}'):
            h = build_effective_hamiltonian(
                low, space, screening='bare',
                delta=not space.basis.rotates_occupied)
        assert 'no_cut_near_degenerate' in _flags(h)
        assert np.isfinite(h.g_eff[0])
        # a smaller no_cut_warn lets the same cut through without a word
        quiet = Numerics(downfolding={'no_cut_warn': 1e-3})
        with warnings.catch_warnings():
            warnings.simplefilter('error', NearDegenerateCutWarning)
            h = build_effective_hamiltonian(
                low, space, screening='bare', numerics=quiet,
                delta=not space.basis.rotates_occupied)
        assert 'no_cut_near_degenerate' not in _flags(h)
    for space in apart:
        with warnings.catch_warnings():
            warnings.simplefilter('error', NearDegenerateCutWarning)
            h = build_effective_hamiltonian(
                low, space, screening='bare',
                delta=not space.basis.rotates_occupied)
        assert 'no_cut_near_degenerate' not in _flags(h)
    assert Numerics().downfolding.no_cut_warn == DOWNFOLDING_NO_CUT_WARN


def test_unrotated_occupied_block_keeps_the_energy_check(low, low_hf):
    nocc = low.nocc
    space = no_space(low, 2, 2)
    assert not space.basis.rotates_occupied
    assert no_space(low, 2, 2, rotate_occupied=True).basis.rotates_occupied
    # equal diagonal elements of gamma_oo across the occupied cut: not
    # compared, the occupied orbitals are canonical
    equal = _with_occupations(space, {nocc - 2: 1.97, nocc - 3: 1.97})
    with warnings.catch_warnings():
        warnings.simplefilter('error', NearDegenerateCutWarning)
        build_effective_hamiltonian(low_hf, equal)
    # the orbital energies are: a degenerate pair across the occupied cut
    eps = low.eps.copy()
    eps[nocc - 3] = eps[nocc - 2]
    with pytest.raises(ValueError, match='eps ='):
        space.require_no_degenerate_cut(eps)


def test_foreign_basis_is_refused(low):
    mol = water('hf').mol.copy()
    mol.set_geom_(mol.atom_coords() * 1.05, unit='Bohr')
    stretched = scf.RHF(mol)
    stretched.conv_tol, stretched.verbose = 1e-11, 0
    stretched.run()
    space = no_space(low, 2, 2)
    with pytest.raises(ValueError, match='other orbitals'):
        build_effective_hamiltonian(run_low_level(stretched, method='hf'),
                                    space)
    norb, nocc, basis = space.norb, space.nocc, space.basis
    fewer = replace(basis, nocc=nocc - 1)
    with pytest.raises(ValueError, match=f'{nocc - 1}.*{nocc}'):
        build_effective_hamiltonian(run_low_level(low.mf, method='hf'),
                                    replace(space, basis=fewer))
    small = replace(basis, rotation=basis.rotation[:-1, :-1])
    with pytest.raises(ValueError, match=f'{norb - 1}.*{norb}'):
        build_effective_hamiltonian(run_low_level(low.mf, method='hf'),
                                    replace(space, basis=small))


@pytest.mark.parametrize('kwargs,error,match', [
    (dict(sigma='srg', sigma_energies='qp'), NotImplementedError,
     "sigma_energies='qp'"),
])
def test_refused_with_a_basis(low, kwargs, error, match):
    with pytest.raises(error, match=match):
        build_effective_hamiltonian(low, no_space(low, 2, 2), **kwargs)


def test_screening_of_the_imaginary_axis_with_a_basis(mf):
    """crpa and crpa_ph build on imagfrequency with a basis; 'mrpa' stays
    on the Casida eigenvectors."""
    low = run_low_level(mf, method='rpa', mode='imagfrequency',
                        natural_orbitals=True)
    for screening in ('crpa', 'crpa_ph'):
        h = build_effective_hamiltonian(low, no_space(low, 2, 2),
                                        screening=screening)
        assert np.isfinite(h.energies.e_c_active)
        assert np.all(np.isfinite(h.g_eff[2]))
    with pytest.raises(NotImplementedError, match="mode='casida'"):
        build_effective_hamiltonian(low, no_space(low, 2, 2),
                                    screening='mrpa')


def test_active_polarizability_of_a_rotated_space(mf, low):
    """P^A of the rotated active pairs from the canonical factors: the
    same on casida as on imagfrequency, on the same factors, and not the
    canonical space's of the same indices."""
    space = no_space(low, 2, 2)
    p_a = rpa.active_polarizability_static(low, space)
    low_if = run_low_level(mf, method='rpa', mode='imagfrequency')
    assert np.abs(rpa.active_polarizability_static(low_if, space)
                  - p_a).max() < 1e-12
    canonical = replace(space, basis=None)
    assert np.abs(rpa.active_polarizability_static(low, canonical)
                  - p_a).max() > 1e-3


def test_exact_integrals_refused(mf, low):
    exact = run_low_level(mf, method='rpa', mode='casida', df=False)
    space = no_space(low, 2, 2)
    with pytest.raises(NotImplementedError, match='df=False'):
        build_effective_hamiltonian(exact, space, screening='bare')
    with pytest.raises(NotImplementedError, match='df=False'):
        rpa.active_correlation_energy(exact, space, np.zeros((4,) * 4))


def test_permuted_mf_refuses_a_basis(mf, low):
    space = no_space(low, 2, 2)
    with pytest.raises(ValueError, match='canonical'):
        space.permuted_mf(mf)
    canonical = replace(space, basis=None)
    assert isinstance(canonical.permuted_mf(mf), ReorderedMF)


def test_qp_energies_diagnostic_is_left_out_with_a_basis(mf):
    low = run_low_level(mf, method='rpa', mode='casida', qp=True,
                        natural_orbitals=True)
    space = no_space(low, 2, 2)
    canonical = replace(space, basis=None)
    h = build_effective_hamiltonian(low, canonical, screening='bare')
    assert 'qp_energies' in h.diagnostics
    h = build_effective_hamiltonian(low, space, screening='bare')
    assert 'qp_energies' not in h.diagnostics


def test_settings_of_the_build(low, low_hf):
    """`settings`: the numerics a build uses; the RPA record lets the
    downfolding and the layout section differ from the run's, refuses any
    other difference; the HF record has none to take."""
    assert low.settings(None) == Numerics()
    own = Numerics(downfolding={'no_cut_warn': 1e-3},
                   layout={'tile_gb': 0.5})
    assert low.settings(own) is own
    with pytest.raises(NumericsMismatch, match='numerics.downfolding'):
        low.settings(Numerics(quadrature={'npade': 12}))
    assert low_hf.settings(None) == Numerics()
    with pytest.raises(TypeError, match='takes no options'):
        low_hf.settings(own)

