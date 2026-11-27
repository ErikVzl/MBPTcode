# Downfolding to an active space (HF and RPA)

`src.EffectiveHamiltonian.Downfolding` builds a static effective Hamiltonian
for an active space from a low-level method on the whole molecule: the orbitals
outside the active space enter as a frozen core (HF) and, for the RPA, as a
screening of the active-space Coulomb interaction. The scheme is that of the
RPA-downfolding paper (cRPA, cRPA_ph and mRPA screening of an active-space
Hamiltonian with a constant E_0), on canonical orbitals, or on an active
space of natural orbitals of the RPA run, whose coupling to the environment
is then folded back in by an energy-dependent term, Delta (see
[Natural-orbital active spaces](#natural-orbital-active-spaces)). The
result goes to any active-space solver in
`src.EffectiveHamiltonian.solvers.solve`.

Available now:

- `method='hf'`: the HF limit, which is CASCI.
- `method='rpa'`, `mode='casida'` (dense Casida, every pole),
  `mode='imagfrequency'` (density fitted, imaginary frequency, N^4) or
  `mode='space-time'` (separable ISDF factors, imaginary time, N^3), with
  `screening` `'bare'`, `'crpa'` or `'crpa_ph'`, and on `mode='casida'`
  also `'mrpa'` (the numerical routes refuse it and point to casida); and
  optionally (every route) `sigma='srg'` or `'averaged'`: the static
  environment self-energy Sigma^E that the paper drops (see below). The
  default, `sigma=None`, is the paper's scheme; see
  [Known difficulties](#known-difficulties) before switching Sigma^E on.
- E_0 always, term by term.
- Restricted closed-shell (RHF/RKS) and unrestricted (UHF/UKS) references,
  on every route. An unrestricted one gives an H_eff with different alpha
  and beta blocks; see [Unrestricted references](#unrestricted-references).
- For a restricted reference, on every route: an active space of natural
  orbitals of the RPA (GW) density, with Delta, the folding of the coupling
  the rotation creates between active and environment orbitals, solved by
  `h.solve`; see [Natural-orbital active spaces](#natural-orbital-active-spaces).
  On an unrestricted reference a rotated basis gives the static Hamiltonian
  only ([below](#unrestricted-references-and-rotated-orbitals)).

## Calling it

```python
from pyscf import gto, scf

from src.EffectiveHamiltonian.Downfolding import (build_effective_hamiltonian,
                                                  run_low_level)
from src.EffectiveHamiltonian.active_space import ActiveSpace

mol = gto.M(atom='C 0 0.6695 0; C 0 -0.6695 0; H 0.9289 1.2321 0; '
                 'H -0.9289 1.2321 0; H 0.9289 -1.2321 0; H -0.9289 -1.2321 0',
            basis='cc-pvdz')   # ethylene; the snippet's pi space is on atoms 0, 1
mf = scf.RHF(mol).run()
active = ActiveSpace.from_population(mf, atoms=[0, 1], ao_type='p',
                                     n_occ=1, n_virt=1)

low = run_low_level(mf, method='rpa', mode='casida', df=True, numerics=None)
h = build_effective_hamiltonian(low, active, screening='crpa', sigma=None,
                                spin_free=True)
roots = h.solve('pyscf_fci', nroots=3)
```

`screening='crpa'` and `sigma=None` are the defaults and could be left out:
the paper's scheme, without Sigma^E and with the full double counting.
`sigma='srg'` adds Sigma^E in the SRG form at the mean-field energies; with
`crpa` on transition-metal active spaces read
[Known difficulties](#known-difficulties) first.

`run_low_level` solves the low-level method on the whole system once and does
not know the active space; `build_effective_hamiltonian` then makes the
effective Hamiltonian of one active space. One `low` serves any number of
active spaces and screenings, so a comparison loop pays for the RPA once.
It belongs to the mean field as it was at the run: when the same `mf`
object is converged again (another geometry in a scan), a build on the old
`low` is refused (`ValueError`), and `run_low_level` has to be called again.
`method='hf'` takes no options: `run_low_level(mf, method='hf')`, and
`build_effective_hamiltonian(low_hf, active)` is CASCI.

**The two limits.** A (0,0) active space,
`ActiveSpace.from_indices(norb, nocc, [], [])`, leaves everything in the
environment: `h.energies.e0` is then the total energy, E_HF for `method='hf'`
and E_HF + E_RPA,corr for `method='rpa'` (the paper's all-environment rows).
`h.solve` accepts it too: it returns one root, the constant alone (E_act =
0), and warns with `EmptyActiveSpaceWarning`, named flag
`'empty_active_space'`. With nothing active there is nothing to fold, so
that constant is `h.energies.e0`, which is then also `g_eff[0]` (see
[Which constant goes with which energy](#which-constant-goes-with-which-energy)
for the case where the two differ). Every orbital
active, `ActiveSpace.from_indices(norb, nocc, range(nocc), range(nocc, norb))`,
is FCI with either method: the screening correction vanishes and the two RPA
correlation energies cancel exactly.

**The routes.** `'casida'` diagonalizes the full Casida problem once and
keeps every excitation energy and amplitude. `'imagfrequency'` diagonalizes
nothing: it keeps the full RPA energy of the existing imaginary-axis code,
its minimax frequency grid and the static polarizability P(0) of the whole
system in the fitting basis. Its screening correction is P - P^A (P^A the
active particle-hole pairs' part) and one Dyson step, which on the same
three-index object is the Casida route's deletion to rounding. E_c^A[v_eff]
is integrated on the full energy's grid (`numerics.quadrature.nfreq_rpa`), so
the E_c pair is one quadrature and cancels to rounding with every orbital
active, whatever the grid; the dense-Casida E_c^A is the
`'ec_active_casida'` diagnostic. Both routes agree on every term of H_eff
and E_0 to the quadrature error of E_c[v] (below 1e-11 Eh at the default
grid on water).

`'space-time'` is the same in imaginary time on separable (ISDF) factors
(X, D), built on the grid `numerics.isdf` names (there is no default grid:
pass e.g. `Numerics(isdf={'level': 'G1'})`). It keeps the factors, P(0) on
the static tau grid (`quadrature.ntau_w`) and the full RPA energy with its
time-frequency grid (`quadrature.ntau_rpa`, `nfreq_rpa`). The same
`nfreq_rpa` counts Gauss-Legendre points here (default 40) but minimax
points on imagfrequency (default 20, tables up to 34). P^A is built as P
was, from the active columns of X on the same grid, and E_c^A takes the
active pairs through the energy's own tau -> omega transform, so with every
orbital active both cancel to rounding on any grid. Against casida on the
mean field's RI factors the route differs by the ISDF fit (NH3/cc-pVDZ, G1
grid: v~ 5e-6 per element, the FCI energy of H_eff 1e-6 Eh, E_0 3e-5 Eh
through E_HF,ee[v~], which cancels against h_eff in the energy).

`'imagfrequency'` and `'space-time'` build their frequency grids on the
occupied-virtual transitions, so a reference without a virtual (or an
occupied) orbital, such as He in a one-function basis, is refused there
(`NotImplementedError`, pointing to casida). `'casida'` takes it: there is
no excitation, E_c = 0 and nothing is screened.

Options of the RPA:

| call | option | values |
|---|---|---|
| `run_low_level` | `mode` | `'casida'`, `'imagfrequency'`, `'space-time'` |
| | `df` | `True`: the screening from the mean field's density fit (an exact eigendecomposition of the MO integrals if it has none; ISDF factors on `'space-time'`); `False`: exact 4-centre integrals, `'casida'` only |
| | `numerics` | a `Numerics`, or `None` for the defaults (`'space-time'` needs one naming the ISDF grid) |
| | `qp` | `False` (default); `True`: also the G0W0 QP spectrum of every orbital, `low.qp` (below) |
| | `natural_orbitals` | `False` (default); `True`: also the occupied-occupied and virtual-virtual blocks of the RPA (GW) density, `low.natural`, from which `natural_orbital_basis` makes natural orbitals ([Natural-orbital active spaces](#natural-orbital-active-spaces)); restricted references only |
| `build_effective_hamiltonian` | `active_space` | one `ActiveSpace` for a restricted reference; the pair `(space_alpha, space_beta)` for UHF/UKS. An `ActiveSpace` with a `basis` (natural orbitals) indexes the orbitals of that basis |
| | `delta` | `True` (default): on a rotated basis, build the coupling between active and environment orbitals that Delta folds (`h.hybridization`) and the E_0 term that goes with it; `False`: leave it out, flagged `'delta_off'`. Nothing to build for canonical orbitals, whatever it is |
| | `spin_free` | `True` (default): spatial-orbital `g_eff`, for UHF/UKS the per-spin blocks `(h_alpha, h_beta)` and `(aa, ab, bb)`; `False`: the 2 n_act interleaved spin orbitals |
| | `screening` | `'bare'`, `'crpa'` (default), `'crpa_ph'`; `'mrpa'` on `'casida'` only |
| | `sigma` | `None` (default, the paper's scheme), `'srg'` or `'averaged'` (Sigma^E, below). On `'casida'` 'srg' is damped at `srg_s`, on `'imagfrequency'` and `'space-time'` it is s = inf, continued from the imaginary axis. Not the default: with `crpa` it can be unreliable, see [Known difficulties](#known-difficulties) |
| | `sigma_energies` | `'mf'` (default): Sigma^E at the mean-field energies of the active orbitals; `'qp'`: at their QP energies (`low.qp`, needs `qp=True`; refused with `sigma=None`) |
| | `diagnostics` | a tuple of names, default all; see below |
| | `numerics` | `None` (default): the settings of the run; a `Numerics` whose `downfolding` section (the Sigma^E settings `srg_s`, `pole_margin`, `continuation_margin`; the cut check of a rotated basis, `no_cut_warn`; the Delta settings `delta_pole_warn`, `delta_conv`, `delta_max_iter`, `delta_weight_floor`, `delta_degenerate`, `delta_linearization_warn`, `delta_occupation_floor`, `delta_lanczos_depth`, `delta_rank_tol`, `delta_remainder_warn`) and `layout` section (the memory of the dense Delta methods, `delta_dense_gb`) are used for this Hamiltonian instead, the rest equal to the run's. The record keeps them as `h.numerics`, and `h.solve` hands them to the Delta solvers ([Settings](#the-settings-of-delta)) |

What the two calls return is in [What you get back](#what-you-get-back).

A restricted reference is one spin channel throughout: the general layer
builds the exact J and K of the environment once and transforms the active
integrals once, and the result goes to the restricted (spin-free) solver.
An unrestricted reference is two channels, alpha and beta, through the same
code; a restricted calculation computes nothing twice and gives the numbers
it gave before the second channel existed.

## What you get back

`run_low_level` and `build_effective_hamiltonian` each return a record. The
low-level one holds the calculation of the whole system, independent of the
active space, and can be used for more than building H_eff; the effective
Hamiltonian holds H_eff, every term of its energy, and the checks.

### From `run_low_level`

`method='hf'` gives `low.mf` and `low.method` only. `method='rpa'` gives an
`RPALowLevel` (the docstring in `methods/rpa.py` has the conventions); for a
UHF/UKS reference `eps`, `nocc`, `factors` and `qp` hold one entry per spin
and `eri` the blocks (aa, ab, bb):

| field | what it holds | routes |
|---|---|---|
| `mf`, `mode`, `df`, `numerics`, `method` | the mean field and the settings of the run | all |
| `eps`, `nocc` | the orbital energies and the occupied count of `mf` | all |
| `e_c_full` | the RPA correlation energy of the whole system, E_c[v] (so E_RPA = E_HF[mf orbitals] + `e_c_full`) | all |
| `omega` | every RPA excitation energy, ascending | `'casida'` |
| `xpy` | X+Y of every excitation, the (i, a) pairs over all occupied and virtual orbitals in C order; with `omega` the static density response chi(0) = -4 xpy diag(1/omega) xpy^T and the transition densities | `'casida'` |
| `factors` | the three-index tensor B[Q, p, q] of the MOs, (pq\|rs) = sum_Q B_Q,pq B_Q,rs | `df=True`, not `'space-time'` |
| `eri` | the exact MO integrals (pq\|rs) | `'casida'`, `df=False` |
| `qp` | the one-shot G0W0 spectrum of every orbital: `energies`, `z`, `converged`, `quasiparticle` ([below](#sigmae-at-the-qp-energies-qptrue-sigma_energiesqp)) | `qp=True` |
| `cache` | what the run and later builds keep: on `'imagfrequency'` the minimax grid and P(0) = chi0(0) in the fitting basis; on `'space-time'` the ISDF factors, grids and P(0); after a Sigma^E build also the lowest excitation energy and W on the grid. Internal: names and contents may change | numerical routes |
| `numerical_flags` | the named flags raised during the run | all |
| `natural` | a `NaturalDensity`: the occupied-occupied and virtual-virtual blocks of the unrelaxed RPA (GW) density of the run's own route in the canonical MO basis, `gamma_oo` (with the reference's 2 on the diagonal) and `gamma_vv`, and the largest element and the Frobenius norm of the occupied-virtual block, `ov_max` and `ov_fro`, which natural orbitals leave out | `natural_orbitals=True`, restricted references |

Not kept: the occupied-virtual block of the density and the relaxed
density; the natural orbitals themselves, which `natural_orbital_basis(low)`
makes from `low.natural` when asked; and on the numerical routes W away from
their grids (on `'casida'` `omega` and `xpy` give it at any frequency).

### From `build_effective_hamiltonian`

`build_effective_hamiltonian` returns one `DownfoldedHamiltonian` `h`, with
`h.g_eff`, `h.space`, `h.energies`, `h.diagnostics`, `h.numerical_flags`,
`h.hybridization` and `h.numerics`; `h.solve(solver, ...)` solves it.
`g_eff = {0: e0_without_delta, 1: h_eff, 2: v_eff}` and
`space` follow the QDPT `build_effective_hamiltonian` contract (`n_orbs`, `n_elec_active`, `act_idx`,
`act_occ_idx`, `act_vir_idx`, `spin_free`, `active_space`; on a rotated
basis also `basis` and `mo`, the rotated orbitals); for a UHF/UKS
reference `g_eff[1]` and `g_eff[2]` hold the per-spin blocks and the entries
of `space` are pairs (see
[Unrestricted references](#unrestricted-references)). `h.hybridization` is
the coupling Delta folds, `None` for canonical orbitals and with
`delta=False`; `h.g_eff_is_static_part` is True exactly when there is one,
and `g_eff` is then the static part of the Hamiltonian only. `h.numerics` is
the `Numerics` the build used, which `h.solve` hands to the Delta solvers.

`h.solve` returns **total energies**: the active energy of each root plus
the constant of the same model. Without a coupling (canonical orbitals,
every run before natural orbitals) that constant is `g_eff[0]`, which then
equals `h.energies.e0`. With a coupling it is `h.energies.e0` for the
default solve, and not `g_eff[0]`, which is the constant of the model
without Delta; which constant goes with which energy, and how a script
gets each, is in
[Which constant goes with which energy](#which-constant-goes-with-which-energy).
`h.solve(solver, nroots=n)` gives the lowest `n` states at the active Sz, of
any spin; `spin=2S` gives the lowest `n` of total spin S exactly
(`'pyscf_fci'` with a spin penalty and an <S^2> check, `'dmrg'` as its SU2
target; refused by the other backends and for an unrestricted reference).
Excitation energies are differences of these roots, e.g. S0 and S1 from
`h.solve('pyscf_fci', nroots=2, spin=0)` and T1 from
`h.solve('pyscf_fci', nroots=1, spin=2)` (examples/22_rpa_downfolding.py).
The `ActiveSpace` is the user's choice (`from_indices`, `from_counts`,
`from_irreps`, `from_population`; on natural orbitals `from_indices` with a
`basis`); nothing here selects orbitals.

`energies` is a frozen record holding every term of E_0 on its own, and `e0`
for their sum:

| field | term | sign in E_0 |
|---|---|---|
| `e_nuc` | nuclear repulsion | + |
| `e_core` | one-electron energy of the environment (the inactive occupied orbitals) | + |
| `e_hartree`, `e_exchange` | Hartree and exchange energy of the environment | + |
| `e_hartree_dc`, `e_exchange_dc` | Hartree and exchange halves of E_HF,ee[v~]: the double-counting correction of the mean-field energy. Zero unless v~ has occupied-occupied elements, so nonzero for `crpa` only | + |
| `e_c_full` | correlation energy of the whole system, E_c[v] | + |
| `e_c_active` | correlation energy of the active space, E_c^A[v_eff]: the double-counting correction of the correlation energy | - |
| `e_exchange_term` | the exchange term that comes with Sigma^E (-2 `e_exchange_dc`); zero with `sigma=None` | + |
| `e_sigma_trace` | Tr[Sigma_bar gamma^A,0]; zero with `sigma=None` | - |
| `e_c_hybridization` | dE_c^A = E_c^{A,Delta}[v_eff] - E_c^A[v_eff], the change of the active space's RPA correlation energy when its orbitals are dressed by the coupling Delta folds ([E_0 with Delta](#e_0-with-delta)); 0.0, not computed, without a coupling (canonical orbitals, `delta=False`) and for `method='hf'` | - (in `e0` only) |

Three properties give the sums per kind, `eh = e_hartree + e_hartree_dc`,
`ex = e_exchange + e_exchange_dc + e_exchange_term` and
`ec = e_c_full - e_c_active - e_c_hybridization`, so that
`e0 = e_nuc + e_core + eh + ex + ec - e_sigma_trace`. A fifth,
`e0_without_delta`, is the same sum without `e_c_hybridization`: the
constant of the active space without Delta, the one `g_eff[0]` holds. With
Delta off the two are one number; for canonical orbitals it is, to the bit,
the E_0 of every run before natural orbitals. These are then the `Enuc`,
`Ecore`, `Eh`, `Ex`, `Ec` columns of the paper's supporting data. In
general neither is a total energy (with nothing active, `e0` is the total,
see the two limits near the top of this page), and with a coupling they
differ: [Which constant goes with which
energy](#which-constant-goes-with-which-energy) says which one a total
uses.

The diagnostics (`h.diagnostics`, keyed by name: `'static_defect'`,
`'ec_active_casida'`, `'sigma_poles'`, `'sigma_continuation'`,
`'qp_energies'`, `'fock_coupling'`, each where it applies) and the named flags
(`h.numerical_flags`) are described in
[Checks that are always on](#checks-that-are-always-on).

### From the solver

`h.solve(...)` returns the total energies of the roots. With `out=` a dict
it also reports `out['e0']`, the constant it added, and `out['e_active']`,
the active energy of each root, so that the energies are
`out['e0'] + out['e_active']`; `out['numerical_flags']`, the named flags of
the call; `out['spin_square']`, <S^2> of each root (for `'pyscf_fci'`; for
an unrestricted reference from the record's alpha-beta overlaps); and with
Delta the per-root entries of [Delta in `h.solve`](#delta-in-hsolve).

It returns no wavefunctions or density matrices. For a record **without a
coupling** (`h.g_eff_is_static_part` False), call the backend on `h.g_eff`
directly: `PyscfFCISolver().roots(h1, eri, n_orbs, (n_alpha, n_beta),
nroots=n)` of `solvers/backends.py` returns the active energies, without
any constant, and the CI vectors (for a spin-free restricted `h`:
`h1 = h.g_eff[1]`, `eri = h.g_eff[2].transpose(0, 2, 1, 3)`, chemist
notation), from which pyscf's `fci.direct_spin1.make_rdm1` gives the active
one-particle density; `h.g_eff[0]` plus those energies are the totals. For
a record **with a coupling** the same call on `h.g_eff` gives the states of
the decoupled active space, without Delta, and `h.g_eff[0]` plus their
energies is the energy of that model, not the total of `h.solve` (rule 4
of [Which constant goes with which energy](#which-constant-goes-with-which-energy)).
A solver outside `h.solve` takes `h.static_hamiltonian('static_sa')`, after
`h.prepare_delta(...)` has set D_SA (without it the call raises
`ValueError`), or `h.static_hamiltonian('static_mf')` instead, a dict in the layout of `g_eff` whose constant
matches its one-body part; the states that the dynamic Delta methods refine
are not returned.

## What is built

H_eff = E_0 + sum h_eff,tu a†_t a_u + 1/2 sum (v + v~)_tu,vw a†_t a†_v a_w a_u,
with the active-space orbitals t, u, v, w:

- **v** is the bare Coulomb interaction on the active orbitals, always the
  exact 4-centre integral, never a density-fitted one.
- **v~** is the screening correction (zero for `hf` and `bare`). For `crpa` it
  is the static (omega = 0) screening of the *residual* polarizability, the
  one with the active particle-hole pairs removed, so that v + v~ is the
  partially screened W^r. `crpa_ph` keeps only the elements of v~ with a
  particle-hole pair in them, those that enter as (ia|jb). `mrpa`
  (Scott and Booth, PRL 132, 076401 (2024), Eq. 5; `mode='casida'` only)
  replaces the particle-hole elements by the static interaction whose RPA
  on the active particle-hole pairs alone has the full RPA's zeroth moment
  eta0 = (X+Y)(X+Y)^T on those pairs (and its first, A - B = Delta, by
  construction): (ia|jb)_eff = 1/4 [eta0_S^-1 Delta_A eta0_S^-1 - Delta_A],
  v~ = (ia|jb)_eff - (ia|jb), with eta0_S the active block of
  (X+Y)(X+Y)^T of the full singlet Casida solve, normalized so that
  (X+Y) Omega^-1 (X+Y)^T = (Delta + 4K)^-1, with no further factor: in spin
  orbitals a singlet mode puts (X+Y)/sqrt(2) on each spin and a triplet
  mode of the direct RPA has X+Y = 1 (A + B = A - B = Delta), so the
  same-spin block of eta0 is (eta0_S + 1)/2 and the opposite-spin block
  (eta0_S - 1)/2, and Eq. 5 gives 1/4 [eta0_S^-1 Delta eta0_S^-1 - Delta]
  in both blocks and nothing in the triplet channel. v~ is zero outside
  the particle-hole elements and zero with every orbital active. Only
  omega = 0 is implemented.
- **h_eff = f - t^DC.** f is the Fock operator of the inactive core, built
  from h^(0) and the bare v. t^DC removes what v~ would otherwise count twice:
  the Hartree and exchange of v~ with the active reference density. For
  `crpa_ph` this is the same general formula with v~ zero outside the
  particle-hole block; its exchange-like part is what survives, and it
  changes the virtual-virtual block of the one-body term. Nothing is
  special-cased for the ph screenings.

E_0 = E_nuc + E_HF,core + E_HF,ee + E_HF,ee[v~] + E_c[v] - E_c^A[v_eff]
(on a rotated basis with Delta, `e0` also subtracts dE_c^A, see
[E_0 with Delta](#e_0-with-delta)):

- E_HF,core and E_HF,ee are the HF energy of the inactive core alone; the
  couplings of core and active space are in f.
- E_HF,ee[v~] = 1/2 Tr[t^DC rho^A,0] is the HF energy of v~ in the active
  reference determinant, entering with a plus sign, so that the reference
  determinant of H_eff keeps the energy E_HF + E_c[v] - E_c^A.
- E_c[v] is the RPA correlation energy of the whole system, E_c^A[v_eff] that
  of the active space alone, both by the plasmon formula and **on the same v
  as the full RPA**. With every orbital active the pair cancels to rounding,
  and with none active E_0 = E_HF + E_c[v]. `bare` has v~ = 0, so it keeps
  the difference E_c[v] - E_c^A[v] (a mechanical embedding). With `hf` both
  E_c terms are zero.

Every formula has a spin-orbital and a restricted form (spin-summed, spatial
orbitals); the code is written per spin channel and contracts the two
identical channels of a restricted reference at the end. For the restricted
forms, with K_ia,jb = (ia|jb), I the inactive and A_o the active occupied
orbitals, `Delta` the orbital-energy differences:

| quantity | restricted |
|---|---|
| f_tu | h_tu + sum over I of [2 (tu\|ii) - (ti\|iu)] |
| t^DC_tu | sum over A_o of [2 v~_tu,kk - v~_tk,ku] |
| E_HF,ee[v~] | sum over k, l in A_o of [2 v~_kk,ll - v~_kl,lk] |
| E_RPA,corr | 1/2 sum Omega - 1/2 sum_ia (Delta_ia + 2 K_ia,ia); Omega from Delta^1/2 (Delta + 4K) Delta^1/2, singlet modes only |
| cRPA v~_tu,vw | -4 (tu\|ia) [(Delta + 4K)_rr^-1]_ia,jb (jb\|vw), "rr" the residual pairs |
| mRPA (ia\|jb)_eff | 1/4 [eta0_S^-1 Delta_A eta0_S^-1 - Delta_A]; spin orbitals: 1/2 [eta0_A^-1 Delta_A eta0_A^-1 - Delta_A] |

## Natural-orbital active spaces

An active space need not be a set of canonical orbitals. It can be a set of
natural orbitals (NOs) of the RPA (GW) density of the low-level run: the
virtual NOs of largest occupation and, optionally, the occupied NOs with the
largest holes. An NO basis is a rotation of the canonical orbitals inside the
virtual block (and, with `rotate_occupied=True`, inside the occupied block);
it never mixes an occupied with a virtual orbital. Unlike canonical orbitals,
rotated ones are coupled to the environment by the reference's own Fock
operator. Delta folds that coupling back into the active space as an
energy-dependent one-body term, which `h.solve` applies. (In this section
Delta is that term; the orbital-energy differences of the formulas above are
written d_ia here.)

Without Delta a rotated basis is not a usable method: on benzene/cc-pVDZ
(HF, the pi CAS(6,6) with virtual NOs) the excitation energies without
Delta lie 1.24 to 1.37 eV above those of exact Loewdin (measured on
2026-11-27 at commit 7c4bf1b), as the prototype of this method found (1.3 to
1.45 eV). Delta is therefore on by default, and leaving it out is flagged.

### Getting the natural orbitals

```python
from src.EffectiveHamiltonian.Downfolding import (build_effective_hamiltonian,
                                                  run_low_level)
from src.EffectiveHamiltonian.active_space import ActiveSpace
from src.EffectiveHamiltonian.orbital_basis import natural_orbital_basis

low = run_low_level(mf, method='rpa', mode='casida', natural_orbitals=True)
basis = natural_orbital_basis(low)              # rotate_occupied=False
print(basis.table())                            # index, molden, block, occupation
nocc, norb = basis.nocc, len(basis.occupations)
# the highest occupied orbital and the virtual NO of largest occupation;
# check their character before trusting the indices
active = ActiveSpace.from_indices(norb, nocc, [nocc - 1], [nocc], basis=basis)
h = build_effective_hamiltonian(low, active)    # delta=True, screening='crpa'
out = {}
energies = h.solve('pyscf_fci', nroots=2, spin=0, out=out)   # 'linearized'
```

`run_low_level(..., natural_orbitals=True)` keeps the occupied-occupied and
the virtual-virtual blocks of the unrelaxed RPA density of the run's own
route as `low.natural` (on casida from the run's own eigenvectors; also on
imagfrequency and space-time), with the size of the occupied-virtual block,
which NOs leave out. It works for restricted references only.

`natural_orbital_basis(low, rotate_occupied=False)` returns an
`OrbitalBasis`: `rotation` (norb x norb; column k is orbital k in the
canonical MOs), `occupations`, `nocc` and `source`, a fingerprint of the
mean field's orbitals. The virtual block holds the eigenvectors of the
virtual-virtual density, in order of descending occupation. With
`rotate_occupied=False` the occupied block is the identity: the occupied
orbitals stay canonical and in canonical order, and their occupations are
the diagonal of the occupied-occupied density, unsorted. With
`rotate_occupied=True` it holds the occupied NOs, also in order of
descending occupation, so the orbitals with the largest holes come last.

**Indices** are 0-based everywhere, as for canonical orbitals. The indices
of an `ActiveSpace` with a `basis` number the orbitals of that basis, not
the canonical ones. Molden files number from 1, so molden NO 45 is index
44; `basis.table()` prints each orbital's index, its molden number
(index + 1), its block (`occ` or `virt`) and its occupation. Choosing the
orbitals is the user's: `ActiveSpace.from_indices(norb, nocc, occ, virt,
basis=basis)`. The other constructors select canonical orbitals and take
no basis. The order of the occupations is not the order of character: look
at the orbitals before trusting an index (`h.space['mo']` holds the rotated
orbitals of a build, for a Molden file), and recompute and check NO indices
read from another run. `examples/24_natural_orbital_downfolding.py` picks
the pi orbitals of ethylene by their p_z weight.

**The cut.** Two NOs whose occupations are equal (symmetry partners, such
as a degenerate pi pair) are an arbitrary pair of vectors of one
eigenspace. A cut between the active and the environment orbitals of a
rotated block is therefore checked on the occupations, by their
difference, the same way for occupied and virtual NOs: below 1e-4
(`QP_DEGENERACY_TOL`) it is refused, naming both occupations; below
`numerics.downfolding.no_cut_warn` (2e-3) it is made, with a
`NearDegenerateCutWarning` and the flag `'no_cut_near_degenerate'`. A block
the basis leaves as the identity (the occupied one with
`rotate_occupied=False`) keeps the check on the orbital energies.

### What the build does on a rotated basis

- **The orbitals.** In each index set of the active space (inactive
  occupied, active occupied, active virtual, environment virtual) the
  orbitals of the basis are made semicanonical: the reference's orbital
  energies are diagonalized inside the span of that set
  (`semicanonical_rotation`). The span of the active NOs is kept, their
  rotation inside it is not. h_eff, v, v~ and E_0 are built on these
  orbitals; `h.space['mo']` holds the orbital coefficients of the semicanonical
  orbitals, and `h.space['basis']` the natural-orbital `OrbitalBasis` the
  active space was made on, not the semicanonical orbitals.
- **The screening.** With rotated active orbitals the one-body part of the
  RPA couples the active particle-hole pairs to the others, so the residual
  polarizability is not "P minus the active bubble". It is built exactly by
  a block inverse: P^r(0) = P(0) + 4 Y^T M^-1 Y, with M on the active pairs
  and Y from the canonical pairs (M[(t,u),(t',u')] = sum_ia O_ti O_t'i U_ua
  U_u'a / d_ia, Y[(t,u), Q] = sum_ia O_ti U_ua B_Q,ia / d_ia, O and U the
  overlaps of the active occupied and virtual orbitals with the canonical
  ones). It depends on the span of the active orbitals only, and for
  canonical ones it is the P - P^A of before, to rounding. All three routes
  and every screening take it (`'mrpa'` on casida only, as before; see
  [mRPA on a rotated basis](#mrpa-on-a-rotated-basis)). The
  `'static_defect'` diagnostic is not zero for `crpa` on NOs: the active
  space holds the bubble of the decoupled active pairs, which differs from
  the folded one in P^r by what Delta restores.
- **E_c^A and Sigma^E.** E_c^A[v_eff] is the RPA of the semicanonical
  active pairs. Sigma^E is built on the rotated active and environment
  orbitals at their semicanonical energies, and its pole checks use those
  energies; `sigma_energies='qp'` is refused (the QP energies belong to the
  canonical orbitals).
- **The coupling** (`delta=True`, the default). The reference's own
  operator in the rotated basis, F_s = T^T diag(eps) T (T the rotation, eps
  the orbital energies), is not diagonal between active and environment
  orbitals of one block. That coupling, and nothing else (see
  [References that are not HF](#references-that-are-not-hf)), is
  `h.hybridization`, a `Coupling`:

      F     = C^T (h0 + J[D] - K[D]/2) C      the bare Fock operator, exact J, K
      lam_E, U_E = eigh F(E_v, E_v);  lam_I, U_I = eigh F(I, I)
      b_particle = F_s(A_v, E_v) U_E          an electron leaves the active space
      b_hole     = F_s(A_o, I) U_I            an electron enters it

  with A_v, A_o the active virtual and occupied orbitals, E_v the
  environment virtual and I the inactive occupied ones. The particle side
  exists with virtual NOs; the hole side with `rotate_occupied=True` and at
  least one active occupied orbital (for canonical occupied orbitals
  b_hole is exactly zero, and there is no hole side). `h.diagnostics
  ['fock_coupling']` holds `'folded'`, the norm of each block, and
  `'left_out'`, the norms of what Delta does not fold. For canonical
  orbitals the coupling is identically zero on any reference: nothing is
  built, `h.hybridization` is None, and the route is bit for bit the one
  before natural orbitals, whatever `delta` is.

With M the eigenstates of the static active Hamiltonian with one electron
less (particle side) or more (hole side), Delta and the Hamiltonian it
gives are

    Delta_tu(w)  = sum_e b_te b_ue / (w - lam_e)                 particle
    H_eff(E)     = H_static + sum_sigma sum_M sum_tu a+_t|M><M|a_u Delta_tu(E - E_M)
                 + sum_sigma sum_M sum_tu a_t|M><M|a+_u sum_i b_ti b_ui / (E + lam_i - E_M)

and the energies with Delta are the solutions of E = an eigenvalue of
H_eff(E). Both sides enter one H_eff(E).

**`rotate_occupied`.** `False`, the default, rotates the virtual orbitals
only: the occupied ones stay canonical and there is no hole side. `True`
rotates both blocks and folds both sides. The hole side is the smaller one
on the cases measured: on water/6-31G (the NO CAS(4,4) of the tests) it
moves the ground state by -0.33 mEh and the two excited states of the tests
by -28 and -21 mEh, on N2/6-31G (NO CAS(6,6)) by -0.04 to -0.07 mEh
(measured with the tests of `tests/test_downfolding_no_hole.py` at commit
964df7c). Occupied NOs chosen by
their holes alone can pick a dynamically correlated orbital: on N2 at 1.1 A
the occupied NO with the largest hole was 2sigma_u, not the 3sigma_g of a
CASSCF(6,6) valence space (RPA occupations 1.9806 and 1.9824; a comparison
with CASSCF run on 2026-11-27, outside the tests); from 1.5 A on the
occupation picked the CASSCF orbitals.

### E_0 with Delta

E_0 subtracts the RPA correlation energy of the model whose active energy is
added to it, so that nothing is counted twice. Without Delta that model is
the decoupled active space, E_c^A[v_eff] on the semicanonical active pairs.
With Delta the model also holds the hopping to the environment levels,
which E_c[v] of the whole molecule already contains at the RPA level, so
E_0 subtracts it as well:

| field | term | sign in E_0 |
|---|---|---|
| `e_c_full` | E_c[v], the whole molecule | + |
| `e_c_active` | E_c^A[v_eff], the decoupled active space | - |
| `e_c_hybridization` | dE_c^A = E_c^{A,Delta}[v_eff] - E_c^A[v_eff] | - |

E_c^{A,Delta} is the same ring sum as E_c^A with the active orbitals
expanded in all canonical orbitals of their block (O, U as above):

    Pi^Delta[(t,u),(t',v)](i w) = -2 sum_i sum_a O_ti O_t'i U_ua U_va 2 d_ia / (d_ia^2 + w^2)
    E_c^{A,Delta} = 1/(2 pi) int dw Tr[ln(1 - Pi^Delta v_eff) + Pi^Delta v_eff]

on casida in the plasmon form over all canonical pairs (as large as the
full Casida problem), on imagfrequency as a frequency sum on the grid of the
full RPA energy, and on space-time on that energy's grid through its own
tau -> omega transform, so that with every orbital active the E_c terms
cancel on every route. dE_c^A is zero for canonical active orbitals
and with every orbital active; it is a constant, so excitation energies do
not see it; it holds both sides; and it is there before any Delta solve, as
the RPA estimate of what the coupling does to the ground state. It is the
RPA-level consistency condition, the logic of the existing -E_c^A, not an
exact result.

`h.energies.e0` holds every term; `h.energies.e0_without_delta` all but
`e_c_hybridization`, and `g_eff[0]` is `e0_without_delta`. The sign of
dE_c^A is not fixed; on the spaces measured it is negative, so `e0` lies
above `e0_without_delta`: -0.45 mEh on water/6-31G (NO CAS(4,5), `no_space(low, 2, 3)`) and
-1.56 mEh on NH3/cc-pVDZ (NO CAS(6,6), `no_space(nh3, 3, 3)`) (with the tests of
`tests/test_downfolding_no_energy.py` at commit 70e8ef7), -1.31 to -2.75
mEh on N2/6-31G at 1.1 and 1.6 A (NO CAS(6,5) to CAS(6,8); a run at commit
7c4bf1b), -0.44 mEh on ethylene/cc-pVDZ (NO CAS(2,2),
`examples/24_natural_orbital_downfolding.py`). On that N2 series against a
canonical CAS(10,16) (cRPA, `sigma=None`), leaving the term out did not
make the dependence of the total energy on the size of the NO space flatter:
the spread over the series was 2.13 mEh with it and 2.36 without at 1.1 A,
4.62 and 5.66 at 1.6 A. It stays in E_0.

`delta=False` builds no coupling, so `e_c_hybridization` is 0.0, the two
constants are one number, and `h.solve` solves the static Hamiltonian only;
the build warns (`DeltaOffWarning`, flag `'delta_off'`). With canonical
orbitals `delta` changes nothing.

### W^r is static, and why the formula stays at omega = 0

The downfolding takes the static W^r(0). The block-inverse identity behind
P^r holds at any frequency, but its extension is not a change of weights:
chi0(i w) weights a pair by d / (d^2 + w^2) = 1/2 [1/(d - i w) + 1/(d + i
w)], a sum of two resolvents, and the block-inverse step holds for one
resolvent at a time (the inverse of a sum is not the sum of the inverses).
The frequency-dependent form needs complex M(i w) = E^T (Lambda - i w)^-1 E
and Y(i w) = E^T (Lambda - i w)^-1 B, P^r(i w) = P(i w) + 4 Re[Y^T M^-1 Y],
one small complex solve per frequency (E the active pairs on the canonical
pairs, Lambda the one-body part of the pair space). Replacing 1/d by
d / (d^2 + w^2) in the static M and Y is wrong (on a random model off by
0.2 to 0.3 on a scale of 70 at w = 0.7 and 3.0, where the complex form
agrees with a brute-force solve to 1e-14). In imaginary time the form needs
the cosine and the sine transform of the tau-loop quantities. On casida the
extension would go through the full RPA resolvent of the stored modes; that
is not worked out. Nothing here needs W^r away from omega = 0, and none is
built.

### The Delta methods

The method is chosen at solve time, `h.solve(solver, delta_method=...)`,
because the dynamic forms need the solver's states; the two static forms go
through the same option:

| `delta_method` | what | cost, limit | accuracy against exact Loewdin |
|---|---|---|---|
| `None` = `'linearized'` (default) | linearized Loewdin: per state a Z estimate E0 of its energy, then one generalized eigenproblem (H_N + D0 - E0 D1) c = E (1 - D1) c, D0 = H_eff(E0) - H_N, D1 = dH_eff/dE(E0) | dense: every eigenvector of the sectors with one electron less (and more); `'pyscf_fci'` | exact to first order in E - E0: below 2e-7 eV on benzene, 7.1e-9 Ha on N2; 1.6e-3 and 1.7e-2 Ha on the strongly coupled excited states of water, which it flags ([below](#choosing-the-delta-method)) |
| `'lowdin'` | exact Loewdin: per state, E iterated to self-consistency, each step the eigenvalue of H_eff(E) whose vector overlaps most with the previous one | dense, as `'linearized'`; `'pyscf_fci'` | the reference: the eigenvalues of the active space coupled to the environment levels (the explicit P + Q matrix, within 3.3e-9 Ha in the tests). Next to a pole the iteration can stop unconverged (`'delta_not_converged'`) |
| `'sa_remainder'` | the static h + D_SA solved once, plus a first-order dynamic correction per state, <K\|Delta(E_K)\|K> - (tr(D dm1_K) + c), Z-renormalized, from the removal and attachment Green's functions of K by block Lanczos | one static solve and a short block Lanczos per state and side, no full sector spectrum; needs `prepare_delta`; `'pyscf_fci'` | 1.1e-4 Ha on benzene, 1.9e-4 Ha on N2 (the bar is 0.01 eV = 3.7e-4 Ha); its limit is strong coupling, flagged `'delta_remainder_large'` (water) |
| `'static_sa'` | D_SA: D_tu = 1/2 [Delta_tu(w_t) + Delta_tu(w_u)] at the removal energies of the active virtual (attachment energies of the active occupied) orbitals, averaged over a declared set of states, added to h | one static solve per declared request for the density matrices (`prepare_delta`), then a plain static Hamiltonian; any solver | 0.100 eV on benzene (the prototype: 0.09 to 0.13 eV) |
| `'static_mf'` | D at the mean-field energies of the active orbitals (the diagonal of the bare Fock operator on the active block) | nothing solved first; any solver; warns on every use | 0.82 eV on benzene (the prototype: 0.8 eV, 1.1 eV at QZ) |
| `False` | no Delta | flagged `'delta_off'` | 1.24 to 1.37 eV too high on benzene (the prototype: 1.3 to 1.45 eV) |

Where each number was measured: benzene/cc-pVDZ HF, the pi CAS(6,6) with
virtual NOs, the largest deviation from `'lowdin'` over six excitation
energies (a run on 2026-11-27 at commit 7c4bf1b; the remainder's
1.1e-4 Ha is on the totals, 8.5e-5 Ha on the excitation energies,
`tests/test_downfolding_no_remainder.py`); N2/6-31G NO CAS(6,6), the eight
lowest states (`tests/test_downfolding_no_delta_dense.py`, commit b54a8ca)
and for `'sa_remainder'` both sides (`tests/test_downfolding_no_remainder.py`,
2026-11-27); water/6-31G NO CAS(4,4) (`tests/test_downfolding_no_delta_dense.py`).
The prototype's numbers were for benzene CAS(6,6) with its own residual
polarizability, which is not the exact one built here.

The static forms change h and run with every solver; their hole part (with
`rotate_occupied=True`) is a one-body matrix with the sign of the particle
part plus a constant, c = -2 tr(D_hole), which is added to the energy. The
hole part of `'static_mf'` has poles at the occupied levels and is even less
trustworthy than its particle part. The three dynamic methods refine the
states of a static solve and run on the FCI engine (`'pyscf_fci'`), the
only backend with the operations on the sectors with one electron less or
more that they need; any other solver is refused before anything is solved,
naming the static forms.

**`prepare_delta`.** D_SA is built once, from a declared set of states:

```python
hs = h.prepare_delta(((2, 0), (1, 2)))         # two singlets, one triplet
e = hs.solve('pyscf_fci', delta_method='sa_remainder', nroots=2, spin=0)
hd = h.prepare_delta(((3, 0),), solver='dmrg')  # D_SA from DMRG states
e = hd.solve('dmrg', delta_method='static_sa', nroots=3, spin=0)
```

Each pair is a request as `solve` takes it, (number of states, spin): spin
None, the lowest states at the active Sz of any spin (then no other
request), or 2S. The ground state is added when no request contains it,
and every state enters with equal weight. `prepare_delta` returns a copy
of the record that holds D_SA (`hs.hybridization.d_sa`, the pair (D, c));
`'static_sa'` and `'sa_remainder'` refuse a record without it, naming
`prepare_delta`. D_SA depends on the set of states, so two solves that are
compared (another `nroots`, another `spin`) share one prepared record, and
for `'sa_remainder'` the states declared should be those solved. The
density matrices come from the backend named, `'pyscf_fci'` (default) or
`'dmrg'` (which takes a spin per request; the agreement of the two is
checked by `tests/dmrg_prepare_delta_check.py` where block2 is installed);
the final solve of `'static_sa'` can use any backend.

### Delta in `h.solve`

**The order of the roots.** Delta can move two states past each other. With
Delta the energies are returned in ascending order, `out['static_index']`
gives for each returned root the static state it came from, and every
per-root entry of `out` is permuted with them. Two flags of
`DeltaRootsCrossedWarning` report what the order shows:

- `'delta_roots_reordered'`: sorting changed the order of returned roots
  that are not degenerate partners. Nothing is wrong with the energies; a
  script that reads root k as the refinement of static state k uses
  `out['static_index']`.
- `'delta_roots_missing'`: more solutions of the Delta problem lie below a
  returned root than returned roots do, so a state the static solve did
  not compute went below it, and more roots are needed (how many is not
  known in advance; the message gives the least `nroots` that can reach
  it). On water/6-31G NO CAS(4,4), `h.solve('pyscf_fci', nroots=2)` raises
  it, correctly: in the explicit P + Q matrix the refinements of static
  states 2 and 5 lie below a returned root, so `nroots=6` is needed to
  compute them (checked against that matrix on 2026-11-27, commit
  68aed94).

The second check counts the eigenvalues of H_eff(E_k) below the root's own,
which is the number of solutions below E_k only below the lowest pole of
the coupling, min(E_M + lam) (E_M - lam on the hole side). So it has limits.
A root flagged `'delta_near_pole'` and a root above the lowest pole are not
checked; `out['crossing_unchecked']` (a boolean per returned root) marks
them, and such a root can have states missing below it without a flag. For
a `'linearized'` root flagged `'delta_linearization'` the count is made, but
its energy is off by about the linearization error, so with another root
within that error the count can be wrong either way; use `'lowdin'` there.
`'sa_remainder'` never forms H_eff(E) as a matrix and counts nothing: all
its roots are unchecked, and `'delta_roots_missing'` is never raised for it
(`'delta_roots_reordered'` is). **Silence means that no missing state was
found among the checked roots, not that none is missing.**

**What `out` reports.** Every solve: `out['e0']`, the constant added, and
`out['e_active']`, the active energy of each root (their sum is the
energies returned), and `out['numerical_flags']`. With a dynamic method,
per returned root: `'static'` (the static energy, without constant; for
`'sa_remainder'` with c), `'correction'` (`e_active - static`), `'z'`,
`'min_denominator'` (the smallest |E - E_M -+ lam| over the poles the root
sees), `'converged'`, `'static_index'`, `'crossing_unchecked'`,
`'spin_square'`, and the method's own entries: `'solutions_below'`,
`'near_pole'` and `'above_pole'` for the dense methods,
`'linearization_error'` and `'linearization_point'` for `'linearized'`,
`'near_pole'` and `'lanczos_change'` for `'sa_remainder'`. With `False` or a
static form, on a record with a coupling, `'static'` is `out['e_active']`,
`'correction'` zero and `'z'` one. A reused `out` loses the Delta entries
of its earlier solve.

**The flags of a Delta solve**, each a named flag with its row in
`docs/warnings.md`: `'delta_near_pole'` (a root, or the linearization point
of `'linearized'`, within `delta_pole_warn` of a pole it sees; the energy is
finite, never NaN, but moves fast with E), `'delta_satellite'` (Z below 0.5,
`QP_WINDOW_Z_MIN`: the root is mostly environment), both
`DeltaNearPoleWarning`; `'delta_linearization'` (`DeltaLinearizationWarning`),
`'delta_not_converged'` (`DeltaNotConverged`), `'delta_remainder_large'`
(`DeltaRemainderWarning`), the two crossing flags above, `'delta_static_mf'`
(`StaticDeltaWarning`, every use of `'static_mf'`) and `'delta_off'`
(`DeltaOffWarning`).

### Which constant goes with which energy

There are two models, and each has its own constant and its own active
energy. A total energy is right only when both come from the same model.

| model | constant | one-body, two-body | active energy E^A from |
|---|---|---|---|
| decoupled active space (static, no Delta) | `e0_without_delta` | h, v | a plain solve of (h, v) |
| active space coupled to the environment (Delta) | `e0` | h, v and the coupling | a Delta method |

`e0 = e0_without_delta - e_c_hybridization` ([E_0 with Delta](#e_0-with-delta)).
The second model has no static (h, v) of its own, except in the static
forms of Delta, where it is (h + D, v), with the constant c of the hole
part of D added to `e0`.

**What the record holds.**

- `g_eff = {0: g0, 1: h, 2: v}`, a dict with v in physicist notation, is
  the decoupled model, all three parts of it:
  `g0 = g_eff[0] = e0_without_delta`. Nothing in `g_eff` contains Delta.
- `energies` holds every term of E_0, with `e0` (every term) and
  `e0_without_delta`.
- `hybridization` is the coupling, None when Delta is off.
  `g_eff_is_static_part` is True when there is a coupling.

**Two things are called solve.** The solver function
(`solvers.solve.solve(solver, g_eff, ...)`) takes any dict
`{0: g0, 1: h, 2: v}` in the layout of `g_eff` and returns g0 + E^A. It
applies Delta only when the coupling is passed to it (`sides`); without it
E^A is that of (h, v), and it does not know there is a Delta. The method of
the record, `h.solve(...)`, knows the Delta method: it passes the coupling
together with the constant that belongs to it. With c the constant of the
hole part of D (zero without a hole side):

| `h.solve` on | dict passed | E^A from | constant added |
|---|---|---|---|
| a record without a coupling (canonical, Delta off) | `g_eff` | (h, v) | `g_eff[0]`, which is `e0` there |
| a coupling and `delta_method=False` | `g_eff` | (h, v) | `e0_without_delta` (= `g_eff[0]`); flag `'delta_off'` |
| a coupling and `'static_sa'` or `'static_mf'` | `{0: e0 + c, 1: h + D, 2: v}` | (h + D, v) | `e0 + c` |
| a coupling and `'linearized'`, `'lowdin'`, `'sa_remainder'` | `{0: e0, 1: h, 2: v}`, with the coupling | the static states of (h, v), refined by a dynamic backend with the coupling | `e0` |

In the last two rows `g_eff[0]` is not used. For `'sa_remainder'` the
static states are those of (h + D_SA, v), the remainder is taken with h,
and the constant c of D_SA is part of E^A: it is in the static energy of
each state, and the remainder takes the whole of D_SA out again. With `out`
a dict, every solve reports `out['e0']`, the constant it added, and
`out['e_active']`, the E^A of each root: the returned totals are their sum.

**For a solver outside `h.solve`**, `h.static_hamiltonian(method)` returns a
dict in the layout of `g_eff` whose constant already matches its one-body
part: `{0: e0 + c, 1: h + D, 2: v}` for the static forms (`'static_sa'`
after `prepare_delta`, or `'static_mf'`, which warns), that is `e0` plus
the constant of the hole part, and `g_eff` itself,
`{0: e0_without_delta, 1: h, 2: v}`, for `False` (with `'delta_off'` when
the record has a coupling). The total is its constant plus that solver's
energy. The dynamic methods have no static form and are refused there.

**Rules for scripts.**

1. A total energy comes from `h.solve(...)`.
2. E_0 for reporting comes from `h.energies.e0`, or from `out['e0']` when
   the constant of one particular solve is meant.
3. E^A comes from `out['e_active']`, not from a total minus a constant
   chosen by hand.
4. `g_eff[0]` plus a solve of (h, v) done by hand is the energy without
   Delta. With a coupling on the record that is never the intended total.

**An illustration. The numbers are invented**, to show the bookkeeping
only: `e0_without_delta` = -76.2000, `e_c_hybridization` = -0.0017, so
`e0` = -76.1983; E^A = -0.3000 without Delta and -0.3400 with it.

| computed | sum | |
|---|---|---|
| `h.solve(delta_method=False)` | -76.2000 - 0.3000 = -76.5000 | consistent, the model without Delta |
| `h.solve()` with a Delta method | -76.1983 - 0.3400 = -76.5383 | consistent, the model with Delta |
| by hand, `g_eff[0]` + E^A with Delta | -76.5400 | mismatch |
| by hand, `e0` + E^A without Delta | -76.4983 | mismatch |

`h.solve` gives one of the first two rows and nothing else. Example 24
prints, for every root, the total beside `out['e0']` and `out['e_active']`,
and `energies.e0` beside `g_eff[0]`.

**With Delta off** (every canonical run, and every run before natural
orbitals) the one Delta term, `e_c_hybridization`, is zero, and `e0`,
`e0_without_delta` and `g_eff[0]` are one number, to the bit what
`g_eff[0]` was before. Existing scripts and tests are unaffected.

**What was considered and not taken.**

- `g_eff[0] = e0`: the solver function on `g_eff` alone would then return
  the fourth row of the illustration, the energy of no model, because
  without the coupling it applies no Delta, whatever the constant.
- `g_eff[0] = 0`, the constant added elsewhere: it changes the canonical
  route (the bit-for-bit anchor, existing tests, the paper's {E_0, h, v}).
- A function that corrects a total afterwards by exchanging the constant:
  the constant and E^A have to change together; exchanging one alone gives
  one of the mismatched rows.

### Choosing the Delta method

The dense methods differ in cost and in how they treat a strongly coupled
state. Wall seconds on water/6-31G RHF, natural orbitals, cRPA, the
particle side (virtual NOs), measured on one Mac with 2 threads at commit
b54a8ca (2026-11-27): the static solve, what `'lowdin'` and `'linearized'`
add to it, and the largest |`'linearized'` - `'lowdin'`| over the roots:

| active space | roots | static | + `'lowdin'` | + `'linearized'` | max \|lin - lowdin\| (Ha) |
|---|---|---|---|---|---|
| CAS(4,4) | 1 | 0.00 | 0.00 | 0.00 | 1.8e-13 |
| | 3 | 0.00 | 0.02 | 0.01 | 1.7e-2 |
| | 5 | 0.00 | 0.03 | 0.01 | 1.7e-2 |
| CAS(6,6) | 1 | 0.02 | 0.12 | 0.10 | 3.9e-11 |
| | 3 | 0.02 | 0.98 | 0.17 | 1.4e-2 |
| | 5 | 0.02 | 1.71 | 0.24 | 1.4e-2 |
| CAS(6,7) | 1 | 0.02 | 1.47 | 1.06 | 2.9e-11 |
| | 3 | 0.02 | 16.9 | 2.45 | 5.0e-3 |
| | 5 | 0.03 | 34.3 | 3.79 | 1.0e-2 |
| CAS(8,8) | 1 | 0.02 | 114 | 61 | 6.9e-11 |
| | 3 | 0.05 | 1276 | 172 | 4.9e-3 |

(CAS(8,8) with five roots was not run.) These times predate the count of
solutions below each root that the crossing check needs; measured later on
a CAS(6,6) (commit 68aed94), it adds 13, 28 and 31% to
`'linearized'` for 1, 3 and 5 roots, and next to nothing to `'lowdin'`.

The ground state of water is weakly coupled (Z close to 1), and
`'linearized'` is exact there to 1e-10 Ha. Its excited states in these
spaces are strongly coupled: in the CAS(4,4), Delta lowers the two excited
states of the tests by 0.31 and 0.37 Ha (`'lowdin'`, commit b54a8ca), their
Z is 0.72 and 0.64, and `'linearized'` misses `'lowdin'` by 1.6e-3 and
1.7e-2 Ha. Linearized Loewdin
replaces H_eff(E) by its tangent at its starting point, so its error is
second order in the coupling for a state with Z close to 1 and grows quickly
as Z drops. It estimates its own error, Z |<c|H_eff(E)|c> - E| at the
result (`out['linearization_error']`), and flags a root whose estimate is
above `delta_linearization_warn` (1 mEh) with `'delta_linearization'`; on
those two water states the estimates were 1.56e-3 and 1.66e-2 Ha against
true errors of 1.56e-3 and 1.68e-2 (`tests/test_downfolding_no_delta_dense.py`,
commit f884386), and both are flagged. With both sides folded
(`rotate_occupied=True`) the two excited states miss by 3.85e-3 and
1.50e-3 Ha, Z 0.69 and 0.73, both flagged
(`tests/test_downfolding_no_hole.py`). A state whose error lies below the
threshold is not flagged: on PBE0 water (both sides) the excited states
miss `'lowdin'` by 6.3e-4 and 1.4e-4 Ha, Z 0.75 and 0.79, without a flag
(`tests/test_downfolding_no_hybrid_reference.py`); lower
`delta_linearization_warn` to be told of such errors. It also flags a root of Z below 0.5
(`'delta_satellite'`) and a linearization point next to a pole
(`'delta_near_pole'`). On the weakly coupled cases it agrees with `'lowdin'`:
benzene/cc-pVDZ pi CAS(6,6), six excitation energies, within 1.8e-7 eV (a
run at commit 7c4bf1b); N2/6-31G NO CAS(6,6), eight states, within 7.1e-9 Ha
(commit b54a8ca); the N2/6-31G ground state at 1.1 and 1.6 A in NO
CAS(6,5) to CAS(6,8), Z 0.992 to 0.998, estimate equal to the true error,
both at most 1e-8 Ha (a run at commit 7c4bf1b); the PBE0 water ground state
within 1.2e-11 Ha (`tests/test_downfolding_no_hybrid_reference.py`).

`'lowdin'` is exact on every state, at a cost that grows with the number of
roots (1276 s against 172 s for three roots of CAS(8,8) above). Next to a
pole its iteration can stop unconverged, which `'delta_not_converged'`
reports; the last energy is kept.

`'sa_remainder'` reaches active spaces beyond the memory limit of the dense
methods. On weakly coupled states it meets the bar of 0.01 eV
(3.7e-4 Ha) against `'lowdin'`: benzene/cc-pVDZ pi CAS(6,6), totals within
1.1e-4 Ha and excitation energies within 8.5e-5 Ha; N2/6-31G NO CAS(6,6),
totals within 1.7e-4 Ha (particle side) and 1.9e-4 Ha (both sides)
(`tests/test_downfolding_no_remainder.py`, measured 2026-11-27). On the strongly
coupled water states it does not: D_SA, evaluated at the energies without
Delta and averaged over the states, overshoots and changes the states of
h + D_SA themselves (on the HF particle side the third singlet of h + D_SA
is another state than Loewdin's, 0.86 Ha off), which a per-state scalar
cannot undo. A correction |dE_K| above `delta_remainder_warn` (1e-2 Ha)
raises `'delta_remainder_large'` as a sign of this regime; every water state
that missed the bar was flagged. The threshold is a heuristic, set between
the largest correction of the states that met the bar (7.0e-3 Ha, N2) and
the smallest of those that missed it (1.5e-2 Ha, water), a margin of 1.4 to
1.5: a flag does not prove an energy wrong, and its absence does not prove
it right. In example 24 (ethylene/cc-pVDZ, NO CAS(2,2)) the triplet is
flagged with |dE| = 1.3e-2 Ha and lies 1.3e-5 Ha from `'lowdin'`, while the
unflagged ground state lies 2.0e-4 Ha from it. Its Lanczos depth,
`delta_lanczos_depth`, is 5 blocks: on the strongly coupled water state the
correction was 7.2e-5 Ha from its full-depth value at depth 3 and 1.2e-7 Ha
at depth 5, for about 20% more time per solve (N2/6-31G and benzene/cc-pVDZ
NO CAS(6,6): 0.051 to 0.060 s and 0.044 to 0.053 s; 2026-11-27, commit
0d914e0).

The rule:

- **One root**, the ground state of a molecule near its equilibrium:
  `'linearized'`, the default, is exact enough, and its estimate shows when
  it is not.
- **Several roots**: `'linearized'`, and read the flags. A root flagged
  `'delta_linearization'`, `'delta_satellite'` or `'delta_near_pole'` is
  solved again with `'lowdin'`, which is exact (and, next to a pole, check
  `'delta_not_converged'`).
- **Beyond the memory limit** of the dense methods: `'sa_remainder'` after
  `prepare_delta`, declaring the states to be solved. A root flagged
  `'delta_remainder_large'` may be far from exact Loewdin; check it with
  `'lowdin'` or `'linearized'` on an active space they can hold.
- **A static Hamiltonian** for another solver (DMRG, coupled cluster, a
  solver outside `h.solve`): `'static_sa'`, within about 0.1 eV on benzene.
  `'static_mf'` is a quick estimate only (0.8 eV off on benzene; it warns).

### Memory of the dense Delta methods

`'lowdin'` and `'linearized'` hold every eigenvector of each sector with one
electron less (and, with the hole side, more), the matrices of the sector
of the states, and one sector's amplitudes <M|a_u|K> at a time, built one
active orbital at a time. Before anything is allocated the solve estimates
this (`dense_memory_gb`: 8 bytes times sum dim(M)^2 over the sectors, plus
the largest dim(M)^2 + 2 dim(M) dim(K), plus 4 dim(K)^2 for `'lowdin'` and
6 dim(K)^2 for `'linearized'`) and refuses the solve when the estimate is
above the limit. The estimates for the singlet sector (equal alpha and beta
counts), GB = 1e9 bytes:

| active space | dim(K) | dim(M), particle / hole | `'lowdin'`, particle side / both sides | `'linearized'`, particle side / both sides |
|---|---|---|---|---|
| CAS(6,6) | 400 | 300 / 300 | 0.009 / 0.011 GB | 0.012 / 0.013 GB |
| CAS(8,8) | 4,900 | 3,920 / 3,920 | 1.44 / 1.69 GB | 1.83 / 2.07 GB |
| CAS(6,10) | 14,400 | 5,400 / 25,200 | 8.6 / 28.1 GB | 11.9 / 31.5 GB |

(computed with `dense_memory_gb`; on N2/6-31G NO CAS(6,6) the estimate for
`'lowdin'`, 9.2 MB, was above the measured peak of 7.4 MB,
`tests/test_downfolding_no_delta_dense.py`).

The limit is `numerics.layout.delta_dense_gb`, in the layout section of
`Numerics`. `None`, the default, means: inside a Slurm job the share of the
allocation that the BSE and Davidson solvers also take
(`allocation_max_memory_mb`: 0.6 of the job's memory per task), outside one
8 GB (`DOWNFOLDING_DELTA_DENSE_GB`). Above it the solve is refused
(`ValueError`) with a message that names the sector, its dimension, the
estimate and the limit, and says: raise the limit, or use `'sa_remainder'`,
which needs no full sector spectrum. Nothing falls back silently. Another
limit is set on the build, whose numerics the record keeps:

```python
from src.Base.numerics.numerics import Numerics

h = build_effective_hamiltonian(low, active,
                                numerics=Numerics(layout={'delta_dense_gb': 32.0}))
```

(for a run with the default numerics; otherwise derive it from the run's,
`low.numerics.derive('more memory for Delta', layout={'delta_dense_gb': 32.0})`).

### The settings of Delta

All are fields of `numerics.downfolding` (default constants in
`src/Base/numerics/constants.py`), set on the build as above, e.g.
`Numerics(downfolding={'delta_linearization_warn': 5e-4})`; a build with
`method='hf'` takes no numerics and uses the defaults.

| setting | default | what it does |
|---|---|---|
| `no_cut_warn` | 2e-3 | the cut of a rotated block between occupations closer than this warns (`'no_cut_near_degenerate'`) |
| `delta_pole_warn` | 0.02 Ha | a root (or a linearization point) closer than this to a pole it sees is flagged `'delta_near_pole'` |
| `delta_conv` | 1e-8 Ha | `'lowdin'` stops when a step moves E by less; also the smallest denominator Delta resolves (a root on a pole is evaluated at this distance, finite); must be below `delta_pole_warn` |
| `delta_max_iter` | 50 | the most `'lowdin'` steps per root, then `'delta_not_converged'` |
| `delta_weight_floor` | 1e-8 | a pole whose weight (b_e . <M\|a\|c>)^2 is below this does not count for a root's smallest denominator |
| `delta_degenerate` | 1e-8 Ha | static states closer than this are refined together as one degenerate set, so the partners come out with one energy and as different states |
| `delta_linearization_warn` | 1e-3 Ha | `'linearized'` flags a root whose estimated error is above this (`'delta_linearization'`) |
| `delta_occupation_floor` | 1e-6 | D_SA drops a direction of the state-averaged occupation (of the active virtual orbitals, or of the holes of the active occupied ones) at or below this, which no state of the average empties or fills |
| `delta_lanczos_depth` | 5 | the blocks of the block Lanczos of `'sa_remainder'`; `out['lanczos_change']` is the change of each correction from one block less |
| `delta_rank_tol` | 1e-8 | block Lanczos takes a start vector of smaller norm as zero and drops a direction of smaller norm |
| `delta_remainder_warn` | 1e-2 Ha | `'sa_remainder'` flags a correction above this (`'delta_remainder_large'`), a heuristic threshold |
| `layout.delta_dense_gb` | None | the memory limit of the dense methods ([above](#memory-of-the-dense-delta-methods)) |

The Sigma^E settings of the same section (`srg_s`, `pole_margin`,
`continuation_margin`) are described with Sigma^E below.

### mRPA on a rotated basis

`screening='mrpa'` works on a rotated basis, on casida, as for canonical
orbitals. The full RPA does not depend on the active space, so its zeroth
moment on the rotated active pairs is a projection, eta0_A = E^T eta0 E
(E[(i,a),(t,u)] = O_ti U_ua), and the closed form carries over with the
semicanonical energy differences,
(ia|jb)_eff = 1/4 [eta0_A^-1 d_A eta0_A^-1 - d_A]. It is the interaction for which the RPA of the *decoupled* active
space reproduces that moment; the tests check that it does. With Delta on,
the solver's model is the coupled active space, so the moment is then
conserved only approximately, as cRPA's static defect is not zero on NOs.
For canonical orbitals it is the mRPA of before, and with every orbital
active the correction vanishes. No reference numbers exist for it on NOs.

### References that are not HF

On a KS reference (PBE, PBE0, ...) the operator an electron feels at the GW
level is F_KS + Sigma_x + Sigma_c - v_xc, and between an active orbital t
and an environment orbital e it has three pieces: the rotation (F_KS(t,e),
zero for canonical orbitals), the exchange part X = <t|Sigma_x - v_xc|e>
(zero on HF) and its correlation partner <t|Sigma_c|e>. Delta folds the
rotation only. X is treated as for canonical orbitals: kept inside the
active block and inside the environment (both are built from the bare Fock
operator, so their levels stay aligned) and dropped between them; the
correlation part is not included anywhere. So on HF nothing is left out,
and on a reference that is not HF the exchange part of the bare coupling
between active and environment orbitals, and its correlation partner, are.
Their measure is reported: `h.diagnostics['fock_coupling']['left_out']`, the
norms of X between active and environment orbitals (`'av_ev'`, `'ao_i'`)
and between occupied and virtual ones (`'ao_ev'`, `'i_av'`, `'i_ev'`); they
are below 1e-7 on HF with exact integrals and between 3.7e-3 and 0.16 on
PBE water/6-31G (`tests/test_downfolding_no_hybrid_reference.py`, commit
3071e1e). The
rotation-only form has no reference numbers on such a reference. The
prototype's PBE numbers (benzene, aug-cc-pVTZ: a mean absolute error of 0.06
eV on PBE against 0.12 eV on HF; butadiene 0.15 against 0.22) folded the
bare coupling in the virtual block instead, X included, with Sigma^E on.
Future work: the correlation part through quasiparticle energies (the same
rotation applied to QP instead of mean-field energies, which the low-level
run can already produce, `qp=True`; the E_0 term has to be rederived for
it), then the bare coupling X with its occupied-virtual blocks and the
singles energy that comes with it, and an mRPA for the coupled model.

### Unrestricted references and rotated orbitals

Canonical orbitals of a UHF/UKS reference work as before
([Unrestricted references](#unrestricted-references)). There are no NOs of
an unrestricted run (`natural_orbitals=True` and `natural_orbital_basis`
refuse it: there is no unrestricted GW density), but a rotated basis from
elsewhere, one per spin, is taken, with `delta=False` only: it gives the
static Hamiltonian (E_0, the one-body and the two-body part, built and
tested per spin channel), flagged `'delta_off'`, which is not a usable
method on its own (above). The default, `delta=True`, is refused
(`NotImplementedError`) because Delta for two spin channels is not built.
What is missing is an unrestricted GW density for the NOs, and Delta for two
spin channels: the sector operations on the spin-resolved FCI backend, and
the coupling's sides and D_SA per spin. The rest, the static part per spin
and the algebra of the Delta solvers (tested with two channels on a model),
is in place for them.

### What is refused

- Natural orbitals of an unrestricted run, and Delta on an unrestricted
  reference with a basis (`NotImplementedError`, above); a basis on one spin
  of an unrestricted pair only (`ValueError`).
- A basis built on other orbitals than the mean field's (another molecule or
  geometry, or the mean field converged again since), or one that does not
  fit its active space (`ValueError`).
- A cut between orbitals of a rotated block whose occupations differ by
  less than 1e-4 (`ValueError`).
- `df=False` with a basis (the rotation needs three-index factors) and
  `sigma_energies='qp'` with a basis (`NotImplementedError`);
  `permuted_mf` of an active space with a basis (`ValueError`: QDPT would
  take the rotated orbitals for canonical ones).
- In `h.solve`: a named `delta_method` on a record without a coupling
  (`ValueError`: nothing to fold); every method but `False` on a record
  built with `spin_free=False` (`NotImplementedError`); a dynamic method
  with a solver other than `'pyscf_fci'` (`ValueError`, naming the static
  forms); `'static_sa'` and `'sa_remainder'` without `prepare_delta`
  (`ValueError`); a dense method above the memory limit (`ValueError`,
  naming `'sa_remainder'`). `prepare_delta` with a backend without density
  matrices, and `'dmrg'` with a request of spin None (`ValueError`).

### Known risks

- The hole side had never been run before this code; its checks are
  internal (the explicit P + Q matrix, the attachment moment).
- `'sa_remainder'` is tested on benzene, N2 and water only, and its flag is
  a heuristic threshold with a margin of 1.4 to 1.5 between the cases that
  set it.
- The matrix-free linearized method of the prototype is not built (its cost,
  about 1,300 to 1,700 sigma vectors per root, may make it impractical), so
  above the reach of the dense methods no method follows the response of
  the wavefunction exactly.
- Roots next to a pole, as in diffuse bases: flagged, not fixed.
- A reference that is not HF: Delta folds the rotation only (above), with
  no reference numbers; a documented limitation, not a flag. It is the
  combination the first application uses.

## Unrestricted references

A `scf.UHF` or `dft.UKS` mean field runs through the same two calls. The
mean-field object alone decides: RHF/RKS is the restricted path, UHF/UKS the
unrestricted one, also when its alpha and beta orbitals coincide (nothing is
collapsed). The alpha and beta orbitals, orbital energies and occupied counts
differ, each spin has its own active space, and H_eff has different alpha and
beta blocks. (An unrestricted reference on a rotated orbital basis gives the
static Hamiltonian only: [Unrestricted references and rotated
orbitals](#unrestricted-references-and-rotated-orbitals).)

```python
from pyscf import gto, scf

from src.EffectiveHamiltonian.Downfolding import (build_effective_hamiltonian,
                                                  run_low_level)
from src.EffectiveHamiltonian.active_space import unrestricted_from_counts

mol = gto.M(atom='O 0 0 0; H 0 0 0.97', basis='cc-pvdz', spin=1)   # OH
mf = scf.UHF(mol).run()
nocc = tuple(int((occ > 0).sum()) for occ in mf.mo_occ)            # (5, 4)
active = unrestricted_from_counts(mf.mo_energy, nocc, 4, (3, 2))

low = run_low_level(mf, method='rpa', mode='casida')
h = build_effective_hamiltonian(low, active, screening='crpa', sigma='srg')
out = {}
roots = h.solve('pyscf_fci', nroots=2, out=out)   # passes h.space['spin_overlap']
print(roots, out['spin_square'])
```

**The active space is a pair** `(space_alpha, space_beta)` of `ActiveSpace`
objects, each built from that spin's orbital count and occupied count, with
independent orbital indices and the same number of active orbitals. The
number of active electrons is `(N_alpha, N_beta)`, the active occupied counts
of the two spins. `unrestricted_from_counts(eps, nocc, n_act, nelec_active)`
builds the contiguous pair (per spin the highest occupied and the lowest
virtual orbitals, pyscf UCASCI's convention). Any other choice is two
`ActiveSpace.from_indices(norb, nocc_spin, occ, virt)`, made in the calling
script; no selection by character or by overlap between the spins is
provided. One `ActiveSpace` with an unrestricted reference, a pair with a
restricted one, two spaces of different size, and a space built with the
other spin's counts are refused. Both selections may be empty.

**What comes back.** With `spin_free=True` (the default)

    g_eff = {0: E_0, 1: (h_alpha, h_beta), 2: (aa, ab, bb)}

each two-body block in the physicist order of the restricted `g_eff[2]`,
`block[p,q,r,s] = <pq|rs>`, the opposite-spin one
`<p_alpha q_beta | r_alpha s_beta>`. With `spin_free=False` it is the same
2 n_act interleaved spin orbitals as for a restricted reference (alpha at
2p, beta at 2p+1), built from those blocks. In `space`, `n_elec_active` is
`(N_alpha, N_beta)`, `act_idx`, `act_occ_idx`, `act_vir_idx` and
`active_space` are pairs (alpha, beta), and `spin_overlap` holds the
alpha-beta overlaps of the orbitals that `solve` needs for <S^2>. Every term
of `energies` is the total over both spins. Diagnostics that belong to one
spin are pairs (the pole-free window, the pole distances, Z, the
continuation spread); a summary of the whole stays one number (the static
defect, the largest over the three blocks; the SRG damping, the largest over
both spins). Messages that name an orbital name its spin
("beta orbital 6").

**The solver.** `'pyscf_fci'` solves the per-spin blocks with
`fci.direct_uhf` at the given `(N_alpha, N_beta)`. In spin-polarized
orbitals H_eff conserves S_z only, so the roots are the lowest states of
that S_z of any spin, and are not spin eigenfunctions in general.
`h.solve(..., out=out)`, which passes `h.space['spin_overlap']` to `solve`,
puts <S^2> of each root in `out['spin_square']`: that of the whole molecule's state, the active
CI vector with the inactive core determinant, from the true alpha-beta
overlaps. (pyscf's own `spin_square` assumes alpha orbital p and beta
orbital p are one function: for the triplet root of a broken-symmetry H2
it gives 0 instead of 2.) `'openfermion'` takes the spin-orbital form with
N_alpha = N_beta only; the DMRG and coupled-cluster backends refuse a
spin-resolved H_eff and name `'pyscf_fci'`.

**The formulas** are those of the restricted case written over the
particle-hole pairs of both spins, stacked (the alpha pairs, then the beta
ones); the code is one implementation with the spin weight g, the number of
electrons an occupied orbital of one channel holds: 2 for the one channel
of a restricted reference, 1 for each of two.

| quantity | with g | restricted (g = 2) | unrestricted (g = 1) |
|---|---|---|---|
| RPA problem | A + B = Delta + 2g K | Delta + 4K, singlet | Delta + 2K on the stacked pairs, K_(ia s),(jb s') = (i_s a_s\|j_s' b_s') |
| W(0) - v, cRPA v~ | -2g V (Delta + 2gK)_rr^-1 V^T | -4 | -2, between each spin's own active orbitals: three blocks |
| residues of W | sqrt(g) sum (tr\|ia)(X+Y) | sqrt(2) | 1, the sum over both spins' pairs |
| chi0(0) | -2g sum B_ia B_ia^T / Delta_ia | -4 | -2 per spin, summed |
| mRPA (ia\|jb)_eff | 1/(2g) [eta0_A^-1 Delta_A eta0_A^-1 - Delta_A] | 1/4, singlet eta0 | 1/2, eta0 of both spins |
| active RPA | Delta_A + 2g K^A | Delta_A + 4 K^A | Delta_A + 2 K^A on the active pairs of both spins |

W is one spin-independent operator, so the three blocks of v~ are the same
W^r - v evaluated between alpha and alpha, alpha and beta, beta and beta
orbitals. For a spin-polarized reference the same-spin and opposite-spin
blocks of the mRPA correction differ; for a closed shell they coincide.
v~ for `crpa_ph` and `mrpa`, E_c^A and the correction of the static defect
vanish only when neither spin has an active particle-hole pair; a spin
without one (a triplet with N^A = (2, 0), say) is fine.

**Sigma^E** is built per spin, with that spin's residues (or, on the
numerical routes, that spin's factors and its own mid-gap), its orbital
energies and the excitation energies of the two-spin RPA, which both spins
share; one `sigma` and one `srg_s` serve both. The pole-free window is per
spin. Its Omega_min is the lowest excitation of the two-spin problem, which
includes the spin-density modes: for a closed shell these carry no weight in
W, so the window of a UHF run on a closed shell is slightly narrower than
that of the RHF run (water/6-31G: 0.705 against 0.728 Ha). That is on the
safe side: the near-pole warning can come too often, never too rarely.

**The routes.** `'casida'` takes every unrestricted reference.
`'imagfrequency'` and `'space-time'` need occupied and virtual orbitals in
both spins and refuse a reference without them (triplet H2 has no occupied
beta orbital), pointing to casida. On `'space-time'` the ISDF factors hold
one collocation per spin on a shared Coulomb factor; the `'rows'`
realization of the fit builds one collocation and is refused for an
unrestricted reference (`numerics.isdf.fit='replicated'`, the default,
works). On the numerical routes Omega_min is the lowest root of the
two-spin problem on the stacked pairs, found as for a restricted reference
(below). `'mrpa'` is casida only, as for a restricted reference.

**Not supported.** An open shell run restricted (ROHF/ROKS) is refused: its
orbitals are not canonical for the alpha and beta Fock operators; run it as
UHF/UKS. A UHF solution whose occupied orbitals are not the lowest of their
spin (a symmetry-constrained or MOM solution) is refused, naming the spin,
and so is one with fractional occupations (`frac_occ`, a smearing): it is
not one determinant, and counting each fractionally occupied orbital as
occupied would describe a system with another number of electrons.

**Spin contamination and instability are reported, not refused.**
`run_low_level` runs the reference diagnostics of the GW driver on a UHF/UKS
determinant: <S^2> against S(S+1), and pyscf's internal stability analysis.
A determinant more than 0.1 above S(S+1) (broken-symmetry H2: 0.98 against
0), or one pyscf finds unstable, is taken with a `RuntimeWarning` and the
named flag `'uhf_spin_contaminated'` or `'uhf_reference_unstable'` in
`low.numerical_flags` and in `h.numerical_flags` of every Hamiltonian
built on it; the numbers are in `mf.reference_diagnostics`. They are
computed once per converged mean field, so the warning comes with the first
run on it and the flags with every run.

**A symmetry-broken reference stays visible in H_eff.** H_eff is built on
the UHF determinant, and its states inherit what that determinant breaks.
For the OH radical, whose 2-Pi ground state is doubly degenerate, the two
lowest states of the (5,4) H_eff are the two components, and they are split
by 0.97 eV with `method='hf'`, 0.90 eV with `crpa_ph` and `mrpa`, 3.6 eV
with `crpa` and `sigma=None`, and 0.02 eV with `crpa` and `sigma='srg'`
(cc-pVDZ; the
example below prints them).

## Where the code differs from the printed paper

The code follows the self-consistent convention, which is the one the paper's
own appendix and its numerical data use:

| item | printed | code |
|---|---|---|
| A, B (Eq. 2.8) | A = Delta - v, B = -v | **A = Delta + v, B = +v** (as Appendix B and Eq. 2.18 need) |
| E_RPA,corr (Eq. 2.22) | mixed forms of the sums | **1/2 Tr[Omega - A]** |
| E_HF,ee[v_eff] | not written | **1/2 Tr[t^DC rho^A,0]** (matches the SI's Hartree and exchange terms of H2 cRPA) |
| E_HF,core, E_HF,ee (Eq. 2.6) | environment | **environment only**, the active-environment terms are in f |
| t^DC for cRPA_ph, mRPA | exchange-like only | **no special case**; the Hartree part vanishes because v~ has no (kk\|ll) element |

**Deletion is the paper's zeroing.** The paper makes the residual response by
zeroing the active-active block of A and B (Eq. 2.14). The code instead
deletes the active pairs (occupied in the active space, virtual in the active
space) from the Casida matrix and solves the smaller problem once. The two are
the same: zeroing A + B on that block alone is wrong (it leaves a block that
screens), but zeroing both A + B and A - B makes the Casida matrix block
diagonal, with zero modes of vanishing amplitude on the active block and
exactly the deleted problem on the rest. The two agree to 4e-15 in the tests.
Cross pairs (active occupied to environment virtual, environment occupied to
active virtual) and environment-environment pairs stay in the residual
response.

## The environment self-energy Sigma^E (`sigma=`)

The paper keeps only the screened interaction and drops the self-energy the
environment gives the active orbitals. With the same assumption as the paper
(delta v_eff / delta G = 0) the correlation self-energy of the embedded
active space is, in general,

    Sigma_emb = iG^E (W - v) + iG^A v~ + iG^A (W - W^A[v_eff]),

W the full screened interaction and W^A[v_eff] the active-space RPA screened
interaction built from v_eff. `sigma='srg'` or `'averaged'` keeps the first two
terms and drops the third, for every screening:

| screening | W^A[v_eff] against W | the dropped term iG^A (W - W^A[v_eff]) |
|---|---|---|
| `crpa` | equal at omega = 0, differ at omega != 0 (v_eff is static) | the paper's static approximation |
| `crpa_ph` | the same active response, bare outer legs on non-ph pairs | nonzero, also at omega = 0 |
| `mrpa` | the same zeroth moment on the active ph pairs, not the same W(0) | nonzero, also at omega = 0 |
| `bare` | active-space screening only | the environment screening `bare` leaves out by design |

So the two-term split is exact only for `crpa`, and only in the static limit;
the `'static_defect'` diagnostic measures W(0) - W^A[v_eff](0) for the others.

The two kept terms change the one-body term and E_0 together, in one code
path (`dc='hartree'` in `DownfoldingTerms`):

- **One-body.** iG^A v~ is the exchange of v~ with the active reference
  density, so it cancels the exchange half of t^DC: only the Hartree double
  counting is left, h_eff = f - J~[gamma^A,0] + Sigma_bar^E (as in QDET).
  For `crpa_ph` and `mrpa` v~ has no (tu|kk) element, the Hartree part
  vanishes and h_eff = f + Sigma_bar^E.
- **E_0** gains the exchange energy term and -Tr[Sigma_bar^E gamma^A,0], at
  the same static Sigma_bar^E the solver sees (not in the paper). Together
  they undo what the one-body change does to the reference determinant,
  which keeps the energy E_HF + E_c[v] - E_c^A. For `crpa_ph` and `mrpa`
  the exchange term is zero.

| quantity with Sigma^E | restricted |
|---|---|
| t^DC_tu | sum over k in A_o of 2 v~_tu,kk (Hartree only) |
| exchange energy term | + 2 K~, K~ = sum over t, v in A_o of v~_tv,vt |
| trace term | - 2 sum over t in A_o of Sigma_bar_tt |

**The static reduction.** iG^E(W - v), with the internal orbital r over the
environment (inactive and environment virtual orbitals), is a pole sum
S_tu(omega) of the residues w of the full W (Casida route: from the RPA's
X + Y) with denominators D_tr = eps_t - eps_r - sgn(eps_r - eps_F) Omega_n.
Every energy (eps_t, eps_r, Omega_n) is the mean field's. It is made static
in one of two ways (Marie and Loos, JCTC 19, 3943 (2023)):

- `'srg'` (the preferred of the two forms, when Sigma^E is used): Eq. 44 at s = `numerics.downfolding.srg_s`
  (default 500 Ha^-2),
  Sigma_bar_tu = sum_rn (D_tr + D_ur)/(D_tr^2 + D_ur^2) w_tr w_ur
  [1 - exp(-(D_tr^2 + D_ur^2) s)].
  Finite everywhere: a term at an exact pole is 0, its limit. `srg_s=None`
  is s = infinity, Eq. 42.
- `'averaged'`: Eq. 12 at eta = 0, Sigma_bar_tu = 1/2 [S_tu(eps_t) +
  S_tu(eps_u)]: S evaluated at both orbital energies and averaged, mode A of
  Kotani, van Schilfgaarde and Faleev (named for what it does, since no qsGW
  loop runs here), and the form of the user's PySCF prototype. Undamped, so
  it diverges at the poles.

At s = infinity the diagonal of `'srg'` is that of `'averaged'`, and since
gamma^A,0 is diagonal every E_0 term is the same for the two; only the
off-diagonal h_eff differs.

**The pole check** (always on with `sigma`). Every pole of S lies outside the
window (max over inactive i of eps_i - Omega_min, min over environment
virtuals a of eps_a + Omega_min). `'averaged'` warns
(`SigmaNearPoleWarning`, named flag `'sigma_near_pole'`), with the orbitals
named, when an active orbital energy is within `numerics.downfolding.
pole_margin` / sqrt(2 s) of an edge (default 3 widths, 0.095 Ha at s = 500)
or outside the window. Sigma_bar is computed and H_eff built all the same;
the elements on those orbitals are then dominated by the nearest pole.
`'srg'` with a finite `srg_s` does not warn of the poles on the Casida
route; near a pole its damping matters, which the `'sigma_poles'` diagnostic
reports. At `srg_s=None` it is as undamped as `'averaged'` and warns the same
way.

Nothing is refused near a pole. Only a Sigma_bar that is not finite, an
active orbital energy exactly on a pole with `'averaged'`, is
(`FloatingPointError`).

`srg_s`, `pole_margin` and `continuation_margin` are read from `low.numerics`,
the `Numerics` given to `run_low_level`. They act after the RPA, so a build
can take other values on the same low-level result:
`build_effective_hamiltonian(low, active, sigma='srg', numerics=other)` uses
the `downfolding` section of `other` for that Hamiltonian and leaves `low` as
it is; a scan over `srg_s` pays for the RPA once. Every other section of
`other` has to equal the run's, which used it (`NumericsMismatch`, a
`ValueError`, otherwise); `low.numerics.derive('why', downfolding={...})`
makes such an object.

**Sigma^E on `mode='imagfrequency'`.** There are no residues on this route.
S^E(mu + i omega), mu the whole system's mid-gap, comes from the GW
density's Lorentzian fit of W - 1 on the minimax W grid
(`numerics.imagfreq.nfreq` points; W built on the first Sigma^E request and
kept), with the internal orbital restricted to the environment. It is
sampled on the `numerics.quadrature.npade` Pade nodes (16 Gauss-Legendre
points of scale `w0_gl`, the GW Pade settings), continued by one Pade fit
per element, and each element is evaluated at its own point:

- `'srg'` is Eq. 42, Sigma_bar_tu = Re S_tu(z_tu) with z_tu = (eps_t +
  eps_u)/2 + i (eps_u - eps_t)/2, because (a + b)/(a^2 + b^2) = Re[(1 +
  i)/(a + i b)]. That is **s = infinity**: `srg_s` enters only the pole
  check here, and `'sigma_poles'` records `srg_s` None.
- `'averaged'` is 1/2 Re[S_tu(eps_t) + S_tu(eps_u)] on the real axis.

Without damping, both forms warn near the poles on this route
(`SigmaNearPoleWarning`), and with a wider margin than on casida: within the
larger of pole_margin / sqrt(2 s) (s = `srg_s`, or 500 when it is None) and
`numerics.downfolding.continuation_margin` (default 0.26 Ha) of the window's
edge, or outside the window. The message names the orbitals and points to
`mode='casida'`, where `'srg'` is finite everywhere. Sigma_bar is continued
and H_eff built all the same, but **inside the margin the continued value is
not reliable**: on water its elements on the named orbitals were off by
1e-4 Eh at 0.16 Ha from the edge, 3.5e-4 at 0.11 Ha and 7.0e-4 at 0.09 Ha,
and no case closer than that or outside the window was measured. The
`'sigma_continuation'` spread does not show this error (3.6e-5 at 0.09 Ha),
so the pole warning is the only sign of it.

Omega_min of the window is the lowest root of the same Casida problem on
the route's own v, for one spin channel or two, computed once and kept. It
is found without building that problem. In the direct RPA A - B is the
diagonal delta of orbital-energy differences, so the Omega^2 are the
eigenvalues of M = delta^1/2 (delta + 2g B^T B) delta^1/2, B the
occupied-virtual fitting factors. A Lanczos iteration on M^-1, which the
Woodbury identity gives from the P(0) the route already has, finds the
lowest one in some tens of applications of B (through the collocation on
space-time, without the three-index object); Omega_min^2 is then the
Rayleigh quotient of M itself, so the quadrature error of space-time's
P(0) does not reach it. It agrees with the casida route's lowest root to
4e-14 Ha or better on water, NH3, N2, benzene, OH and CH2.

Omega_min changes no number of H_eff, only where the pole warning starts,
so an iteration that does not finish does not stop the build. Omega_min is
then replaced by its exact lower bound, the smallest orbital-energy gap,
with a `SigmaNearPoleWarning` and the named flag
`'sigma_pole_window_bound'`. The window is then narrower than the true one:
the near-pole warning can come for an orbital that is not near a pole,
never the other way.

The measured accuracy: the exact Casida S^E fed through the continuation
alone misses the exact Sigma_bar by at most 1.8e-7 Eh per element on
benzene (6,6) cc-pVDZ (2e-9 on water), against a target of 1e-4; the whole
route agrees with casida to 1.6e-7 Eh on benzene and 1e-7 on water, and E_0
to 2.5e-10. Towards the window's edge the continuation amplifies the small
error of the sampled S^E (2e-8): on a water space with a high virtual active
the error reaches 1e-4 at 0.16 Ha from the edge and 7e-4 at 0.09 Ha. Every
case measured beyond 0.17 Ha met 1e-4 (a few much closer did too), and
`continuation_margin` is 1.5 times that; the scan is in the comment of
`DOWNFOLDING_CONTINUATION_MARGIN` in `src/Base/numerics/constants.py`.

**Sigma^E on `mode='space-time'`** is sampled, continued, warned about and
diagnosed exactly as on imagfrequency, on the route's own objects: W on the
same minimax frequency grid (`numerics.imagfreq.nfreq`), from chi0 of the
ISDF factors (built on the first Sigma^E request and kept), its Lorentzian
fit convolved exactly with the environment's G, and the active-environment
fitting factors B_Q,tr = sum_P D_PQ X_Pt X_Pr. Only the active block is
needed, so the cost stays cubic, times the number of active orbitals.

Sampling S^E through the imaginary-time quadrature of the space-time GW
self-energy does not work here: it leaves about 1e-7 of noise on the
continuation's nodes, which the continuation amplifies to 1e-4 and more for
an active orbital far from the gap (2.2e-4 on an NH3 space with virtuals
1.2 Ha above the mid-gap). The fit's samples are an exact rational function
and continue cleanly. Against the Casida problem on the same ISDF factors,
which isolates the continuation, this was checked on 5 cases (three active
spaces of benzene and two of NH3, cc-pVDZ): 2.8e-8 on the nodes (the NH3
cases) and at most 9.3e-7 Eh per element of Sigma_bar^E. The scan over 160
active spaces of water, NH3 and benzene compares with casida on the mean
field's RI factors, where the difference also holds the ISDF fit of the
integrals: beyond the 0.26 Ha margin up to 4.2e-5 Eh on water, 8.2e-6 on
NH3 and 8.5e-5 on benzene (G1 grid). The FCI energy of H_eff with
`sigma='srg'` agrees to 1.2e-6 Eh on NH3.

### Sigma^E at the QP energies (`qp=True`, `sigma_energies='qp'`)

`run_low_level(mf, method='rpa', ..., qp=True)` also solves one-shot diagonal
G0W0@RPA for every orbital of every spin on the run's own W,

    eps_p = eps_p^mf + <p|Sigma_x - v_xc|p> + Re Sigma_c,pp(eps_p),

and keeps it as `low.qp` (a `QuasiParticles` record: `energies`, `z`,
`converged`, one array per spin channel; `None` without `qp`). The numbers
are those of `calc_qp_energy(mf, selfenergy='GW', polarizability='RPA')` on
the same route: on casida from the run's own Omega and X+Y (no second Casida
solve) at the GW code's default broadening `DEFAULT_BROADENING_ETA`; on
imagfrequency on the run's own W(i w) grid, the one Sigma^E uses (built once,
by the run, and the grid and W of calc_qp_energy's driver), with the
driver's own steps after it; on space-time through the GW driver on the
run's numerics, which builds its own W on its own grids (on the run's ISDF
factors for a restricted reference; an unrestricted one fits its own, and
has no Z).
An orbital whose equation has no root at all is NaN and not converged,
named in the flag `'qp_no_root'`; a root kept with Z outside (0, 1] is
named in `'qp_z_outside'`. A satellite is not NaN: on casida the pole sum has
a root between every pair of poles, and the root of largest weight is kept,
as calc_qp_energy keeps it, even when its Z is small (O 1s and high virtuals
of PBE water: Z = 0.29-0.42). `low.qp.quasiparticle` marks the roots that
are quasiparticles by the root finder's own test, Z in
(`QP_WINDOW_Z_MIN` = 0.5, 1] (`quasiparticle_weight`), and the flag
`'qp_satellite'` names the others (where Z is not known, unrestricted
space-time, every root counts).

`sigma_energies='qp'` then evaluates Sigma^E (`'srg'` or `'averaged'`) at the
QP energies of the active orbitals instead of the mean-field ones; the poles
of G^E stay at the mean-field energies, and the pole checks and the
`'sigma_poles'` record use the same energies. Refused without `low.qp`,
with `sigma=None`, and when an active orbital has no QP root or its root is
no quasiparticle (`low.qp.quasiparticle` False). The
`'qp_energies'` diagnostic records the active orbitals' QP energies and Z
whenever `low.qp` is set.

This is the energy choice of quantum defect embedding theory (Sheng, Vorwerk,
Govoni and Galli, JCTC 18, 3512 (2022)): its static one-body term evaluates
iG^R W0 (G^R = G0 - G0^A, the environment propagator, and W0 the full G0W0
screened interaction) in the averaged form at the full system's G0W0 QP
energies (their Eq. 29). Two differences remain: QDET makes iG^E W0 static
including exchange, with V_xc subtracted (their Eq. 26), while here
iG^E(W - v) is made static and the exchange enters through f and the
double counting; and QDET starts from KS orbitals, as `mf` does here for a
KS reference.

## Kohn-Sham references

A restricted or unrestricted KS mean field (GGA or hybrid) runs, with the KS
orbitals and energies. f and t^DC are still built from h^(0) and the bare v, never from the
KS Fock matrix, and E_HF,core and E_HF,ee are the HF energy expression on the
KS orbitals, without any XC energy. The RPA uses the KS orbital energies.
This is what the paper's SI data for H2/PBE show. The paper's Appendix A
guarantees no double counting only for an HF Green's function; for a KS
reference E_0 is a well-defined working choice, not a derived one. On a
rotated basis (natural orbitals) a KS reference leaves part of the coupling
unfolded; see [References that are not HF](#references-that-are-not-hf).

## Checks that are always on

- **Degenerate boundary.** An active orbital and an environment orbital of
  the same kind (both occupied or both virtual) closer than
  `QP_DEGENERACY_TOL` (1e-4 Eh) are refused, with the orbital numbers and
  energies on both sides: include or exclude the whole set (`from_irreps`,
  `from_population`). Arbitrary rotations inside a degenerate set would
  otherwise change single elements of H_eff. On a rotated block of an
  orbital basis (natural orbitals) the check compares occupations instead,
  refusing below 1e-4 and warning below `no_cut_warn`
  (`NearDegenerateCutWarning`, `'no_cut_near_degenerate'`; see
  [Getting the natural orbitals](#getting-the-natural-orbitals)).
- **An orbital basis.** A basis built on other orbitals than the mean
  field's (another geometry, or the mean field converged again since) is
  refused, as is one that does not fit its active space.
- **Delta.** Every root of a Delta solve is checked for a nearby pole, a
  small Z, its linearization error (`'linearized'`), convergence
  (`'lowdin'`), the size of its correction (`'sa_remainder'`) and the order
  of the roots; each is a named flag in `out['numerical_flags']`
  ([Delta in `h.solve`](#delta-in-hsolve)). A rotated basis without Delta is
  flagged `'delta_off'`, every use of `'static_mf'` `'delta_static_mf'`.
- **Mean field.** Gas phase, T = 0 only; restricted closed-shell or
  unrestricted with the lowest orbitals of each spin occupied by one
  electron each. Open-shell ROHF, solvent models and, for UHF/UKS,
  fractional occupations (a smearing, pyscf's `frac_occ`) are refused.
- **Serial only.** More than one MPI rank is refused (every rank would repeat
  the whole calculation).
- **Option validation.** An unknown or not-yet-available mode, screening,
  sigma or diagnostic is refused with the valid list. Their names are taken
  in any case and with `-` or `_` (`'SRG'`, `'Averaged'`, `'CRPA-PH'`); for
  `sigma` only `None` itself means no Sigma^E (the string `'None'` is
  refused).
- **Sigma^E poles.** `sigma='averaged'` warns near a pole of the
  environment self-energy (`SigmaNearPoleWarning`, named flag
  `'sigma_near_pole'`); on `mode='imagfrequency'` and `'space-time'` both
  `'srg'` and `'averaged'` do, within the wider continuation margin (see
  above). H_eff is built all the same and the flag is in
  `h.numerical_flags`; only a Sigma_bar that is not finite is
  refused.
- **Sigma^E warnings.** With `sigma` set, two checks of Sigma_bar always
  run, whatever `diagnostics` asks for, because Sigma_bar goes into H_eff
  and E_0 either way: a `SigmaDampingWarning` (casida, `'srg'`) when the
  SRG damping moves an element by more than 1e-4 Ha, and a
  `SigmaContinuationWarning` (imagfrequency, space-time) when two Pade fits
  disagree by more than 1e-4 Ha: one threshold for both, since they never
  fire for the same run and both mean Sigma_bar is uncertain at that level. Both are named flags
  (`docs/warnings.md`). The `diagnostics` keyword only decides whether the
  numbers behind them are recorded.
- **Named flags in the result.** A warning that is a named flag also leaves
  a `{'flag': name, 'message': text}` entry in the result, so a script can
  test for it without catching warnings: `low.numerical_flags` holds the
  flags of `run_low_level`, and `h.numerical_flags` those plus the
  ones raised by `build_effective_hamiltonian` for this active space. Both
  are empty lists for a clean calculation.

Diagnostics are informational and do not change a number. `diagnostics=` takes
a tuple of names (default: all; `None` is the same as `()`); a name that does
not apply to the call is skipped, so one tuple works across a loop. A
diagnostic that applies but is exactly zero (an empty active space, no
environment, no particle-hole pair) is recorded as 0.0; one that does not
apply is absent. Results are in
`h.diagnostics`, keyed by name. Which entries appear depends on the route
and on the call (`method='rpa'`):

- `'static_defect'` is recorded on every route. `'ec_active_casida'` only on
  imagfrequency and space-time. `'sigma_poles'` on every route, and
  `'sigma_continuation'` on imagfrequency and space-time, both only with
  `sigma` set. `'qp_energies'` only when `low.qp` is set (`qp=True`) and the
  active space is on canonical orbitals (on a rotated basis it is left out).
- `'fock_coupling'` is not one of the names `diagnostics=` takes: it is
  recorded whenever the Fock coupling is built (a rotated basis with Delta
  on), whatever `diagnostics` is.

The entries:

- `'static_defect'`: the largest element of W(0) - W^A[v_eff](0) on the
  active orbitals. On canonical orbitals it is zero for `crpa`, by
  construction, and measures how far `bare`, `crpa_ph` and `mrpa` are from
  it. On a rotated basis it is not zero for `crpa` either: the active
  space holds the bubble of the decoupled active pairs, and the difference
  is what Delta restores (5.3e-3 and 5.6e-3 on the NO spaces of
  `tests/test_downfolding_no_screening.py`, imagfrequency and space-time,
  commit c3beee9).
- `'fock_coupling'` (a rotated basis with Delta on): `{'folded': the norm
  of each coupling block, 'left_out': the norms of what Delta does not
  fold}` ([References that are not HF](#references-that-are-not-hf)).
- `'sigma_poles'` (with `sigma` only): a dict with `'window'`, the pole-free
  window (lo, hi); `'pole_distance'`, per active orbital the distance of
  its energy to the nearer edge (negative outside); `'z'`, per active
  orbital Z_t = 1/(1 - S'_tt(eps_t)) = 1/(1 + sum w_tr^2 / D_tr^2), 1 far
  from the poles and small near one; `'srg_s'`, the s applied (None for
  s = infinity and for `'averaged'`); and `'srg_damping'`, the largest
  |Sigma_bar(s) - Sigma_bar(infinity)| (0 when no finite s is applied).
  When the damping exceeds `DOWNFOLDING_SRG_DAMPING_WARN` (1e-4 Ha) a
  `SigmaDampingWarning` names the active orbitals concerned: their
  Sigma_bar depends on `srg_s`. On imagfrequency and space-time `'z'`
  comes from the continued fraction, `'srg_s'` is None and `'srg_damping'` 0.0 (s =
  infinity, no damping applied).
- `'ec_active_casida'` (imagfrequency, space-time): `{'casida': the
  dense-Casida E_c^A on the same v_eff, 'difference': the quadrature value
  minus it}`.
- `'qp_energies'` (whenever `low.qp` is set, `qp=True`): `{'energies',
  'z'}` of the active orbitals, pairs for UHF/UKS (see
  [Sigma^E at the QP energies](#sigmae-at-the-qp-energies-qptrue-sigma_energiesqp)).
- `'sigma_continuation'` (imagfrequency, space-time, with `sigma`): per
  element of Sigma_bar, the difference between two Pade fits on the even-
  and the odd-indexed nodes at the element's evaluation point. It is a
  lower-bound-type indicator, not an error bound: near the window's edge
  it falls short of the true error by ten to twenty times (3.6e-5 against
  7.0e-4 at 0.09 Ha from the edge, a space the pole warning names), so a
  small spread does not certify Sigma_bar there; far from
  the poles it can exceed the error (1.2e-5 against 1.6e-7 on benzene).
  Above `DOWNFOLDING_SIGMA_CONTINUATION_WARN` (1e-4 Ha) a
  `SigmaContinuationWarning` names the elements: use more nodes or
  `mode='casida'`.

## What is checked against what

- A brute-force oracle that builds H_eff from the definitions with dense
  matrices, on the same integrals (agreement 1e-8 Eh).
- The paper's SI numbers for H2 (2 electrons in 2 orbitals) and N2 (6 in 6),
  with spherical basis functions and exact integrals. The energy terms agree
  to at most 1.6e-5 Eh for H2 and 6.8e-4 Eh for N2 (the N2 numbers are the
  absolute HF terms; the screening-dependent differences agree far better),
  for every screening including `mrpa`. The SI comes from a different code
  and basis representation. Benzene, whose geometry the SI does not give,
  on the QUEST geometry instead: the Frobenius norms of v~ of the (6,6) pi
  space agree to 1.5e-3 (crpa) and 2e-4 (crpa_ph, mrpa) relative, and the
  lowest environment excitations Omega^R of the (10,10) space lie 1 to 3
  mEh above the SI's, the size a 0.5% change of the geometry makes.
- mRPA: the restricted form equals Eq. 5 written in spin orbitals (both
  spins, every singlet and triplet mode) to 2e-13, whose same-spin and
  opposite-spin blocks are equal; the RPA of the active pairs with the
  mRPA v_eff has the full RPA's projected zeroth moment; mRPA and crpa_ph
  norms differ by 8% on benzene (6,6), as in the SI.
- The user's PySCF prototype, run on its own orbitals: v_eff, f, t^DC, h_eff,
  the excitation energies and the CAS energies of the ethylene (2,2) and
  benzene (6,6) pi spaces for `bare`, `crpa` and `crpa_ph`, to machine
  precision, and its Sigma_bar (`'averaged'`). The orbitals are pinned because
  the degenerate pi pairs of benzene rotate arbitrarily from one SCF to the
  next.
- Sigma^E: `'srg'` with every orbital as the environment is the active block
  of the qsGW SRG kernel; both forms equal the oracle's pole sums; the
  reference determinant keeps its energy when the switch is turned on (with
  the real Sigma_bar and with Sigma_bar forced to 0); every orbital active
  still gives FCI; along a scan through a pole crossing `'srg'` stays smooth.

- Unrestricted references (no published numbers exist): the same
  determinant handed over as RHF and as a UHF object gives the same H_eff
  and energy terms on every route (1e-13 on casida); with every orbital of
  both spins active the screening vanishes, the E_c pair cancels and H_eff
  gives the FCI energy of the molecule; with none active E_0 is E_UHF + E_c;
  the reference determinant of H_eff has the energy
  E_UHF + E_c - E_c^A for every screening, with and without Sigma^E.
  Independently of the code: the oracle written in spin orbitals without
  any spin adaptation, for E_c, v~ (cRPA, mRPA), the residues and both
  Sigma^E forms, on spaces with different orbitals per spin,
  N_alpha != N_beta and unequal cores (agreement 2e-13); pyscf's UCASCI for
  `method='hf'` and its URPA for E_c on the same density fit; mRPA
  reproduces the full RPA's zeroth moment on the active pairs. The
  numerical routes against casida on the same factors: the mean field's own
  density fit on imagfrequency, the route's own ISDF factors on space-time
  (the FCI energy of H_eff to 3e-8 Eh, Sigma_bar to 1e-6 Eh). Molecules:
  the OH radical, triplet CH2, broken-symmetry H2, and OH with PBE0.

- Natural orbitals (no published numbers exist for this code's form): the
  density blocks against the GW density of `compute_gw_density_matrix`
  and between the routes; an identity or a permutation basis gives the
  canonical result; the residual screening of a rotated space against a
  dense solve over its residual pairs (Cholesky on casida, the explicit
  one-body matrix on imagfrequency and space-time), on water, NH3 and
  random models, with one and two spin channels; the coupling against the
  rotated orbital energies projected through the environment, and exactly
  zero for canonical orbitals on HF and PBE; E_c^{A,Delta} against the
  dense RPA of all canonical pairs; exact Loewdin against the eigenvalues
  of the explicit P + Q matrix (water, both sides, HF, PBE and PBE0, and a
  model with two spin channels); block Lanczos at full depth against the
  dense resolvent; `'linearized'` and `'sa_remainder'` against
  `'lowdin'` (water, N2, benzene); the degenerate partners of N2 with one
  energy and as two different states. The static part with two spin
  channels on the OH radical with a random rotation per spin: `method='hf'`
  against UCASCI, a closed shell run as UHF against the restricted result.

Use spherical basis functions (`cart=False`, the PySCF default) when comparing
with the paper: Cartesian ones are far further from its numbers.

## Choosing the active space and what to expect

The code checks that H_eff is built as defined; it cannot check that the
active space is a good one, or that a combination of options suits a
system. What follows comes from using it (the MOBH35 transition-metal
barriers, def2-TZVP, imagfrequency, PBE-type hybrids with alpha = 0 to 1 of
exact exchange), not from a derivation.

**Where a gain over the RPA comes from.** With every screening the energy is
E_HF + E_c[v] - E_c^A + E_act (plus the Sigma^E terms): the solver replaces
the RPA's description of the correlation inside the active space, and
nothing else. A gain over the RPA of the whole system can therefore only
come from correlation that the RPA gets wrong *and* that lies inside the
active space: near-degenerate (static) correlation, or the RPA's missing
exchange among the active electrons. For a single-reference system expect
the RPA's quality, not better; an error of the RPA that sits in orbitals
outside the active space stays as it is. The active space has to contain
the orbitals whose correlation the RPA gets wrong, which is a choice of the
user.

**Canonical orbitals or natural orbitals.** With canonical orbitals of
`mf`, a selection by character (`from_population`, or a score on the
reacting atoms) picks orbitals spread over the spectrum, with environment
orbitals of the same kind between them, and an orbital of the right
character can be spread over several canonical orbitals, none of which is
picked. Natural orbitals ([Natural-orbital active
spaces](#natural-orbital-active-spaces)) order the orbitals by their
occupation in the RPA density instead, which is not the order of
character either: an occupation can pick a dynamically correlated orbital
(N2 above). Either way, look at the orbitals (a Molden file) and at how
much of the chosen character the picked orbitals hold before trusting a
result; along a reaction path or a scan keep the same size and the same
character at every point, since a change of the space from one point to
the next goes straight into the energy difference.

**The starting point.** All screenings start from the orbitals and energies
of `mf`. The RPA of the whole system and the downfolded energy need not be
best on the same mean field: on GGA (PBE) orbitals the active-space part can
add error that a hybrid with more exact exchange does not, while the RPA of
the whole system may be best near the GGA. Compare a few starting points
before fixing one.

### Known difficulties

**cRPA with Sigma^E can be unreliable** (`screening='crpa'` with
`sigma='srg'` or `'averaged'`), which is why `sigma=None` is the default.
On transition-metal complexes, mostly 3d ones (Mn, Fe), the Sigma^E
correction came out at tens of kcal/mol, differed between the structures of
one reaction and jumped with the starting point; it did not go away with
another choice of the active space. With `crpa_ph` and `bare` it stayed at
a few kcal/mol. If you use Sigma^E with `crpa`, compare with `sigma=None`
on every structure.

**No gain over the RPA where the active space misses the correlation.** See
"Where a gain over the RPA comes from" above: on single-reference reactions
expect the RPA's accuracy, and a better one only where the active space
holds the orbitals whose correlation the RPA gets wrong. Selecting them is
the hard part, with canonical orbitals and with natural orbitals alike.

**Results depend on the starting point.** See above; the downfolded energy
and the RPA of the whole system can be best at different alpha.

**The Sigma^E warnings on large molecules.** `SigmaContinuationWarning`
(threshold 1e-4 Ha, imagfrequency and space-time) and `SigmaNearPoleWarning`
come on most transition-metal complexes of a few hundred to a thousand
basis functions, also where Sigma^E changes the energy little and smoothly
from one structure to the next. They mark Sigma_bar elements that are
uncertain at the 1e-4 Ha level, not results that are wrong: read them
together with the size of the Sigma^E contribution (the total energy with
and without `sigma`, and `h.energies.e_sigma_trace`) and how it varies over
the structures being compared.

## Not yet available

- For natural orbitals: the NOs of an unrestricted run and Delta for two
  spin channels; the matrix-free linearized Delta method of the prototype
  (`'linearized_iterative'`); the dynamic Delta methods on a backend other
  than `'pyscf_fci'` (DMRG has no sector operations); the part of the
  coupling that a reference that is not HF adds (the exchange part and its
  correlation partner); an mRPA for the coupled model. Selecting the
  active NOs is the user's.
- omega other than 0, and other ways of making W^r static.
- Self-consistent energies in Sigma^E; the term iG^A(W - W^A[v_eff]) for
  `bare`, `crpa_ph` and `mrpa`.
- For unrestricted references: ROHF/ROKS; spin-adapted states (H_eff
  conserves S_z only); the DMRG and coupled-cluster backends on a
  spin-resolved H_eff; selecting the two active spaces by character or by
  alpha-beta overlap.
- Further low-level methods (MP2, TDHF/BSE, CC polarizability), a MACIS
  backend and MPI-parallel downfolding.

Examples: [`examples/22_rpa_downfolding.py`](../../examples/22_rpa_downfolding.py)
(ethylene and benzene pi spaces) and
[`examples/23_unrestricted_downfolding.py`](../../examples/23_unrestricted_downfolding.py)
(the OH radical on a UHF reference, all three routes, <S^2>) and
[`examples/24_natural_orbital_downfolding.py`](../../examples/24_natural_orbital_downfolding.py)
(the pi NO CAS(2,2) of ethylene with Delta: `'linearized'`,
`'sa_remainder'` and `'lowdin'`, each total beside `out['e0']` and
`out['e_active']`).
