"""Mean fields and settings shared by the tests of the natural-orbital
downfolding (tests/test_downfolding_no_*.py)."""
from pyscf import dft, scf

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
