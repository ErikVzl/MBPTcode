"""Downfolding onto natural orbitals: the pi space of ethylene in cc-pVDZ on
an active space of RPA natural orbitals (NOs), with Delta, the folding of the
coupling the rotation creates between the active and the environment
orbitals (see "Natural-orbital active spaces" in
docs/methods/downfolding.md).

Steps: the RPA run keeps the occupied-occupied and virtual-virtual blocks of
its GW density (natural_orbitals=True); `natural_orbital_basis` turns them
into an orbital basis, whose `table()` lists every orbital with its 0-based
index (the one an ActiveSpace takes), its molden number (index + 1), its
block and its occupation. The occupied orbitals stay canonical
(rotate_occupied=False, the default); the virtual ones are NOs, in order of
descending occupation. The CAS(2,2) is the canonical occupied pi orbital and
the virtual NO of largest occupation that is pi, both picked by character
(a p_z weight above 0.9).

The build folds the coupling by default (delta=True): `h.hybridization`
holds it, and E_0 gains one term, e_c_hybridization. So there are two
constants ("Which constant goes with which energy" in
docs/methods/downfolding.md): `h.energies.e0`, every term, the constant of the
active space coupled to its environment; and `h.energies.e0_without_delta`,
without that term, the constant of the decoupled active space, which is
`h.g_eff[0]`. `h.solve` adds the constant that belongs to the active energy
it computed, and reports both parts: out['e0'] (the constant added) and
out['e_active'] (the active energy of each root), whose sum is the total.

Two Delta methods: 'linearized' (the default; it reports Z and an estimate
of its own error per root) and 'sa_remainder' (a static h + D_SA, built by
`prepare_delta` from the states declared, plus a first-order correction per
state). For each the two lowest singlets (spin=0) and the lowest triplet
(spin=2), and the excitation energies S1 - S0 and T1 - S0. Any named flag a
solve raises is printed with its message. A flagged root is to be checked
with 'lowdin', exact Loewdin, the reference: it is solved last, for every
state, so the two methods can be compared with it."""
import math
import os
import sys
import warnings

import numpy as np
from pyscf import gto, scf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Base.numerics.constants import HARTREE_TO_EV
from src.EffectiveHamiltonian.active_space import ActiveSpace
from src.EffectiveHamiltonian.Downfolding import (build_effective_hamiltonian,
                                                  run_low_level)
from src.EffectiveHamiltonian.orbital_basis import natural_orbital_basis

# ethylene, planar in the xy plane (the pi orbitals are the carbon p_z):
# C=C 1.339 A, C-H 1.087 A, HCH 117.6 degrees
R_CC, R_CH, HCH = 1.339, 1.087, 117.6


def ethylene():
    half = math.radians(HCH / 2.0)
    x, y = R_CH * math.sin(half), R_CC / 2.0 + R_CH * math.cos(half)
    atoms = [('C', (0.0, R_CC / 2.0, 0.0)), ('C', (0.0, -R_CC / 2.0, 0.0))]
    atoms += [('H', (sx * x, sy * y, 0.0)) for sy in (1, -1) for sx in (1, -1)]
    return gto.M(atom=atoms, basis='cc-pvdz', verbose=0)


def pz_weight(mol, c):
    """The Mulliken-like weight of each column of `c` (AO coefficients) on
    the p_z (and d_xz, d_yz) functions: 1 for a pure pi orbital."""
    s = mol.intor('int1e_ovlp')
    pi = np.array([any(t in lab for t in ('pz', 'dxz', 'dyz'))
                   for lab in mol.ao_labels()])
    return np.einsum('pi,pi->i', c[pi], (s @ c)[pi])


def solve(h, label, method):
    """The two lowest singlets and the lowest triplet of `h` with
    delta_method=`method`; prints every root's total, out['e0'] and
    out['e_active'], and any named flag; returns {state: total}."""
    totals = {}
    for nroots, spin in ((2, 0), (1, 2)):
        out = {}
        with warnings.catch_warnings():
            # the flags are printed below from out['numerical_flags']
            warnings.simplefilter('ignore')
            energies = h.solve('pyscf_fci', delta_method=method,
                               nroots=nroots, spin=spin, out=out)
        for k, e in enumerate(energies):
            name = f"{'ST'[spin // 2]}{k + (spin // 2)}"
            totals[name] = e
            print(f'  {label:12s} {name}: total {e:.8f} = out[e0] '
                  f'{out["e0"]:.8f} + out[e_active] {out["e_active"][k]:.8f}'
                  f'   Z {out["z"][k]:.4f}')
        if 'linearization_error' in out:
            print(f'  {"":12s} estimated linearization error per root (Ha): '
                  + ', '.join(f'{x:.1e}' for x in out['linearization_error']))
        for f in out['numerical_flags']:
            print(f"  {'':12s} flag {f['flag']!r}: {f['message']}")
    return totals


def main():
    mol = ethylene()
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.run()
    low = run_low_level(mf, method='rpa', mode='casida', natural_orbitals=True)
    basis = natural_orbital_basis(low)          # rotate_occupied=False
    nocc, norb = basis.nocc, len(basis.occupations)

    print('ethylene/cc-pVDZ RHF, E_HF = %.8f' % mf.e_tot)
    print('natural-orbital basis, the frontier of basis.table() '
          '(it lists all %d orbitals):' % norb)
    rows = basis.table().splitlines()
    print('\n'.join([rows[0]] + rows[1 + nocc - 3:1 + nocc + 4]))

    # the pi orbitals by character: the occupied canonical one, and the
    # first virtual NO (the largest occupation) that is pi
    weight = pz_weight(mol, mf.mo_coeff @ basis.rotation)
    occ_pi = [k for k in range(nocc) if weight[k] > 0.9]
    vir_pi = [next(k for k in range(nocc, norb) if weight[k] > 0.9)]
    print(f'active: occupied {occ_pi} (p_z weight {weight[occ_pi[0]]:.3f}), '
          f'virtual NO {vir_pi} (p_z weight {weight[vir_pi[0]]:.3f}, '
          f'occupation {basis.occupations[vir_pi[0]]:.5f})')
    active = ActiveSpace.from_indices(norb, nocc, occ_pi, vir_pi, basis=basis)

    h = build_effective_hamiltonian(low, active, screening='crpa')
    en = h.energies
    print(f'\nthe constants: energies.e0 {en.e0:.8f} (with Delta), '
          f'g_eff[0] {h.g_eff[0]:.8f} = energies.e0_without_delta '
          f'{en.e0_without_delta:.8f}; e_c_hybridization '
          f'{en.e_c_hybridization:+.2e}')
    print(f'the coupling Delta folds: folded block norms '
          f"{h.diagnostics['fock_coupling']['folded']}")

    print('\ntotal = out[e0] + out[e_active] for every root:')
    lin = solve(h, 'linearized', 'linearized')
    hs = h.prepare_delta(((2, 0), (1, 2)))   # D_SA from the states solved
    rem = solve(hs, 'sa_remainder', 'sa_remainder')
    # a flagged root is checked with exact Loewdin, the reference
    exact = solve(h, 'lowdin', 'lowdin')

    print('\nexcitation energies (eV):   S1 - S0   T1 - S0')
    for label, t in (('linearized', lin), ('sa_remainder', rem),
                     ('lowdin (the reference)', exact)):
        print(f'  {label:22s} {(t["S1"] - t["S0"]) * HARTREE_TO_EV:8.4f}  '
              f'{(t["T1"] - t["S0"]) * HARTREE_TO_EV:8.4f}')


if __name__ == '__main__':
    main()
