"""Mean fields and settings shared by the tests of the natural-orbital
downfolding (tests/test_downfolding_no_*.py)."""
from pyscf import dft, scf

from src.Base.numerics.numerics import Numerics
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
