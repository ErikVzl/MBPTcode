"""The dynamic backends: solvers of an active-space Hamiltonian with an
energy-dependent one-body term, Delta, the folding of the coupling a rotated
orbital basis creates between the active and the environment orbitals
(spec 6, 6c; the coupling is `Coupling` of Downfolding/hybridization.py,
handed over as the plain data of `Coupling.sides()`).

A static backend (backends.py) gives the lowest states of a Hamiltonian. A
dynamic backend refines states that a static backend found: it does not
search, so how many states there are and of which spin was decided by the
static solve, and the result holds the same states, in the order they were
handed in. It works through an engine, a static backend with sector
operations (`has_sector_operations`: today `PyscfFCISolver`), from which it
takes everything that touches a sector with one electron less or more: the
sector matrices, and the creation and annihilation of one electron. With M
the eigenstates of the sector with one electron of spin sigma less
(particle side) or more (hole side),

    Delta(z)  = (b / (z - lam)) b^T   particle,  z = E - E_M
              = (b / (z + lam)) b^T   hole
    H_eff(E)  = H_N + sum_{sides, sigma, M} amp_M^T Delta(E - E_M) amp_M,
                amp_M[u, K] = <M|a_u|K> (particle) or <M|a+_u|K> (hole).

Delta is spin free, so H_eff(E) keeps the spin of a state; both sigma are
summed explicitly. Two dense methods, on the full spectrum of every sector
(`dense_memory_gb` is what that costs; above `dense_memory_limit_gb` a solve
is refused and names 'sa_remainder'):

- 'lowdin' (`LowdinSolver`), exact Loewdin, the reference: per state,
  E <- the eigenvalue of H_eff(E) whose vector overlaps most with the
  previous one, until E is self-consistent. Its converged energy is the
  exact eigenvalue of the active space coupled to the environment levels
  (the explicit P + Q matrix), however strong the coupling. It iterates, and
  next to a pole the iteration can oscillate or wander and stop without
  converging (DeltaNotConverged, 'delta_not_converged'; the last energy is
  kept).
- 'linearized' (`LinearizedLowdinSolver`): per state a Z-estimate E0, then
  one generalized eigenproblem (H_N + D0 - E0 D1) c = E (1 - D1) c with
  D0 = H_eff(E0) - H_N and D1 = dH_eff/dE(E0). It does not iterate, so it
  always returns. H_eff(E) is replaced by its tangent at E0, so the result
  is exact to first order in E - E0: for a weakly coupled state (Z close to
  1) E0 lies close to the self-consistent energy and the error is second
  order; as Z drops, E0 moves away and the error grows quickly. Each root
  carries the estimate Z |<c|H_eff(E)|c> - E| of its error in
  extra['linearization_error'], and one above
  numerics.downfolding.delta_linearization_warn raises
  DeltaLinearizationWarning, 'delta_linearization': for that root use
  'lowdin'.

Each root reports its static energy, Z = 1 / (1 - <c|dH_eff(E)|c>) and the
smallest denominator |E - E_M -+ lam| over the poles it sees (weight
above delta_weight_floor). Two verdicts flag a root for both methods: a
denominator below numerics.downfolding.delta_pole_warn (DeltaNearPoleWarning,
'delta_near_pole': the energy moves fast with E there), and Z below
QP_WINDOW_Z_MIN (DeltaNearPoleWarning, 'delta_satellite': the root is mostly
environment, not a state of the active space). Two more apply to 'linearized'
only: a linearization point E0 within delta_pole_warn of a pole of the states
handed in (also 'delta_near_pole'), and a linearization error above
numerics.downfolding.delta_linearization_warn (DeltaLinearizationWarning,
'delta_linearization'). A denominator smaller than delta_conv is evaluated at
that size, so a root on a pole is finite, never inf or NaN. The algebra is
the kernel `delta_roots`, on dense arrays only, which knows neither the
engine nor the spin structure of the integrals: it is tested with two spin channels and both kinds of side on a
model.

Each root also reports how many solutions of the Delta problem lie below it
(extra['solutions_below']), so that a caller can tell whether a state that
was not handed in went below a refined one. Below the lowest pole every
eigenvalue of H_eff(E) decreases with E and meets the line lambda = E once,
so the eigenvalues of H_eff(E_k) below E_k are the solutions below E_k
(above poles the count is that number minus the poles below E_k:
Haynsworth's inertia additivity on the explicit P + Q matrix). The
dynamic backends do not sort: `solvers.solve.solve` sorts the refined roots
and raises DeltaRootsCrossedWarning from these counts.

The removal and attachment Green's functions of one CI state K need no
sector spectrum: `removal_spectrum` and `attachment_spectrum` run block
Lanczos (`block_lanczos`, which sees a product and its start vectors alone)
on the engine's product of the sector with one electron of spin sigma less
or more, started from a_{u sigma}|K> or a+_{u sigma}|K>, and return poles
and weights with G_tu(z) = sum_m weights[t, m] weights[u, m] / (z - poles[m]).
Its depth is numerics.downfolding.delta_lanczos_depth blocks; a start
vector with a norm below numerics.downfolding.delta_rank_tol is taken as
zero, and a start or Krylov direction whose part outside the directions
kept before it has a norm below delta_rank_tol is dropped.

The third method, 'sa_remainder' (`RemainderSolver`), needs no sector
spectrum and is not dense: a static Hamiltonian h + D_SA, the static form
of Delta built beforehand from a declared set of states, solved once, plus
a first-order dynamic correction per state, the expectation value of what
D_SA leaves out, r_K(E) = <K|Delta(E)|K> - (tr(D dm1_K) + c), from those
Green's functions by Lanczos, as one Z-renormalized step
E_K + r_K / (1 - r'_K). It does not relax the states. Its limit is strong
coupling: when Delta shifts a state by an amount comparable to the
distance between its removal and attachment energies and the environment
levels, D_SA overshoots and changes the states of h + D themselves, and
the scalar cannot undo that; a correction above
numerics.downfolding.delta_remainder_warn is flagged
'delta_remainder_large' (DeltaRemainderWarning). It flags
'delta_near_pole' and 'delta_satellite' as the dense methods do, reports in
extra['lanczos_change'] the change of each correction from one Lanczos
block less, and counts no solutions below a root (it forms no H_eff(E)),
so `solvers.solve.solve` leaves all its roots unchecked for missing
states.
"""
import math
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg

from src.Base.numerical_flags import flag
from src.Base.numerics.constants import (BYTES_PER_GB,
                                         DOWNFOLDING_DELTA_DENSE_GB,
                                         FCI_SPIN_SQUARE_TOL, MB_PER_GB,
                                         QP_WINDOW_Z_MIN)
from src.Base.numerics.numerics import numerics_or_default
from src.Base.utils.parallel.memory import allocation_max_memory_mb
from src.EffectiveHamiltonian.solvers.backends import sector_shape

#: the sentence that ends the per-root Delta messages: k is the static order
_STATIC_INDEX = (". The state number is that of the static solve; "
                 "out['static_index'] of `solve` maps the static states to "
                 "the returned roots, which are sorted ascending.")

#: the dense methods and the number of dim(K)^2 matrices each holds
DENSE_MATRICES = {'lowdin': 4, 'linearized': 6}


class DeltaNearPoleWarning(RuntimeWarning):
    """A root of a Delta solve next to a pole, two flags.
    'delta_near_pole': its smallest denominator |E - E_M -+ lam| over the
    poles it sees, or for 'linearized' that of its linearization point E0
    (for 'sa_remainder' the poles are those of Lanczos, also at the static
    energy E_K where the remainder was taken), is below
    numerics.downfolding.delta_pole_warn; the energy is finite, but
    it moves fast with E there, and the root may belong to the environment
    state rather than to the active one. 'delta_satellite': its Z is below
    QP_WINDOW_Z_MIN, so it is mostly environment, a satellite, not a state
    of the active space."""


class DeltaLinearizationWarning(RuntimeWarning):
    """A root of linearized Loewdin whose estimated linearization error,
    Z |<c|H_eff(E)|c> - E|, is above
    numerics.downfolding.delta_linearization_warn: the state is coupled
    strongly enough (Z well below 1) that one linearization of H_eff(E) does
    not reach its self-consistent energy. Exact Loewdin
    (delta_method='lowdin') gives the exact value."""


class DeltaNotConverged(RuntimeWarning):
    """Exact Loewdin ran out of steps (numerics.downfolding.delta_max_iter)
    before one step moved E by less than delta_conv; the last E is kept."""


class DeltaRemainderWarning(RuntimeWarning):
    """A state of the remainder method 'sa_remainder' whose correction
    |dE_K| exceeds numerics.downfolding.delta_remainder_warn (flag
    'delta_remainder_large'). The correction is first order in
    Delta - D_SA; a large one means strong coupling: D_SA, evaluated at
    the energies of the states without Delta, overshoots and has changed
    the states of h + D themselves, which a per-state scalar cannot undo,
    so the energy can be far from exact Loewdin. 'lowdin' or 'linearized'
    gives the energy-dependent result where the active space fits their
    memory limit."""


class DeltaRootsCrossedWarning(RuntimeWarning):
    """The refined roots of a Delta solve are not the static states in
    their order, two flags, raised by `solvers.solve.solve`.
    'delta_roots_reordered': Delta moved two returned roots past each other,
    so sorting the result ascending changed their order; out['static_index']
    maps each returned root to its static state. 'delta_roots_missing': more
    solutions of the Delta problem lie below a returned root than there are
    returned roots below it, so a state that the static solve did not
    compute went below it; more roots are needed to compute it, and how
    many is not known in advance (the missing state can come from a
    static state far above the last one computed). The count is not made
    for every root, and never for 'sa_remainder', which forms no H_eff(E):
    out['crossing_unchecked'] marks the roots left out (see
    `solvers.solve.solve`), and no flag does not mean that no state is
    missing."""


@dataclass(frozen=True)
class DeltaResult:
    """The states of one static solve refined with Delta, in the order they
    were handed in.

    energies (nroots,): with Delta, without any constant; static (nroots,):
    the energies handed in; z (nroots,): 1 / (1 - <c|dH_eff(E)|c>);
    min_denominator (nroots,): the smallest |E - E_M -+ lam| over the poles
    whose weight (b_e . <M|a|c>)^2 exceeds delta_weight_floor (inf where
    none does); vectors: one array per root, of the shape the vector handed
    in had, unit Euclidean norm; converged (nroots,) bool; extra: what a
    method reports beyond these. Both dense methods report
    extra['solutions_below'] (nroots,) int, the eigenvalues of H_eff(E_k) of
    the requested spin (refine's `spin`; every one for spin None) other than
    the root's own and those of its degenerate set, lying more than
    delta_degenerate below the root's own eigenvalue: the solutions of the
    Delta problem below the root (module docstring), and
    extra['near_pole'] (nroots,) bool, True where the root was flagged
    'delta_near_pole', and extra['above_pole'] (nroots,) bool, True where
    the root lies above the lowest pole E_M + lam (hole side E_M - lam) of
    any sector, a pole that couples, where the count is solutions minus
    poles below. 'linearized' also reports
    extra['linearization_error'] (nroots,), its error estimate
    Z |<c|H_eff(E)|c> - E| in Ha, and extra['linearization_point']
    (nroots,), the E0 each root was linearized at. 'sa_remainder'
    (`RemainderSolver`) reports extra['near_pole'] and
    extra['lanczos_change'] (nroots,), |dE(depth) - dE(depth - 1)| in Ha,
    and no 'solutions_below' or 'above_pole'; its z is taken at the static
    energy, its vectors are those handed in."""
    energies: np.ndarray
    static: np.ndarray
    z: np.ndarray
    min_denominator: np.ndarray
    vectors: list
    converged: np.ndarray
    extra: dict = field(default_factory=dict)


def dense_memory_gb(dim_k, dims_m, method):
    """The memory, GB, a dense Delta solve takes: 8 bytes times the
    eigenvectors of every sector M, sum dim(M)^2, plus the largest
    dim(M)^2 + 2 dim(M) dim(K) (one sector matrix in its diagonalization
    and two amplitude slices), plus n dim(K)^2 with n = 4 for 'lowdin'
    (H_N, H_eff, its eigenvectors, one work matrix) and 6 for 'linearized'
    (also D1 and the right-hand matrix). dim_k: the dimension of the sector
    of the states; dims_m: those of the sectors with one electron less or
    more."""
    if method not in DENSE_MATRICES:
        raise ValueError(f'method={method!r}: the dense Delta methods are '
                         f'{sorted(DENSE_MATRICES)}')
    dims_m = [int(d) for d in dims_m]
    numbers = (sum(d * d for d in dims_m)
               + max((d * d + 2 * d * dim_k for d in dims_m), default=0)
               + DENSE_MATRICES[method] * dim_k * dim_k)
    return 8.0 * numbers / BYTES_PER_GB


def dense_memory_limit_gb(numerics):
    """The memory, GB, the dense Delta methods may take:
    numerics.layout.delta_dense_gb when set; else the share of the Slurm
    allocation the BSE and Davidson solvers also take
    (`allocation_max_memory_mb`), and DOWNFOLDING_DELTA_DENSE_GB outside
    Slurm. numerics None: the defaults."""
    numerics = numerics_or_default(numerics)
    if numerics.layout.delta_dense_gb is not None:
        return float(numerics.layout.delta_dense_gb)
    return allocation_max_memory_mb(
        default=MB_PER_GB * DOWNFOLDING_DELTA_DENSE_GB) / MB_PER_GB


# ----------------------------------------------------------------- the kernel
class _Sector:
    """One entry of `sectors` with the rows of b that couple; `resolution`,
    the smallest denominator Delta resolves (`delta`)."""

    def __init__(self, kind, e_m, amp, lam, b, resolution):
        if kind not in ('particle', 'hole'):
            raise ValueError(f"kind={kind!r}: 'particle' or 'hole'")
        self.sign = 1.0 if kind == 'particle' else -1.0
        self.e_m, self.amp = np.asarray(e_m, dtype=float), amp
        self.lam = np.asarray(lam, dtype=float)
        b = np.asarray(b, dtype=float)
        self.rows = np.flatnonzero(np.any(b != 0.0, axis=1))
        self.b = b[self.rows]
        self.resolution = resolution

    def lowest_pole(self):
        """The lowest pole E_M + lam (particle) or E_M - lam (hole) of this
        sector over the levels that couple (a column of b that is not zero);
        inf where nothing couples."""
        used = np.any(self.b != 0.0, axis=0)
        if len(self.rows) == 0 or not np.any(used):
            return np.inf
        return float((self.e_m.min() + (self.sign * self.lam[used]).min()))

    def denominators(self, energy):
        """E - E_M - lam (particle) or E - E_M + lam (hole), (dim M, n_lev)."""
        return (energy - self.e_m)[:, None] - self.sign * self.lam[None, :]

    def delta(self, energy, derivative):
        """Delta_tu(E - E_M) over the coupled rows, (dim M, r, r); its
        derivative in E with `derivative`. A denominator smaller in size than
        `resolution` (delta_conv: E itself is known no better) is taken at
        that size, with its sign (+ for an exact zero), so that a root on a
        pole gives a finite Delta, which the smallest denominator reports,
        and never inf or NaN."""
        den = self.denominators(energy)
        small = np.abs(den) < self.resolution
        if np.any(small):
            den = np.where(small, np.where(den < 0, -self.resolution,
                                           self.resolution), den)
        r = -1.0 / den ** 2 if derivative else 1.0 / den
        return np.einsum('me,te,ue->mtu', r, self.b, self.b, optimize=True)

    def amplitudes(self, c):
        """<M|a_u|c> (a+_u on a hole sector) over the coupled rows u,
        (r, dim M)."""
        return np.array([self.amp(u) @ c for u in self.rows]).reshape(
            len(self.rows), len(self.e_m))


def _fold(sectors, energy, dim, derivative=False):
    """sum_{sectors, M} amp_M^T Delta(E - E_M) amp_M (or its derivative in E),
    dense (dim, dim), with at most two amplitude slices in memory: for each
    coupled row t, Y_t = sum_u Delta_tu * A_u, then A_t^T Y_t."""
    out = np.zeros((dim, dim))
    for s in sectors:
        if len(s.rows) == 0:
            continue
        d = s.delta(energy, derivative)
        for i, t in enumerate(s.rows):
            y = None
            for j, u in enumerate(s.rows):
                a_u = s.amp(u)
                a_u *= d[:, i, j][:, None]
                if y is None:
                    y = a_u
                else:
                    y += a_u
                del a_u
            a_t = s.amp(t)
            out += a_t.T @ y
            del a_t, y
    return out


def _expectation(sectors, energy, c, derivative):
    """<c|H_eff(E) - H_N|c> (or <c|dH_eff(E)|c>) from the amplitudes of c."""
    total = 0.0
    for s in sectors:
        if len(s.rows) == 0:
            continue
        w = s.amplitudes(c)
        total += float(np.einsum('tm,mtu,um->', w, s.delta(energy, derivative),
                                 w, optimize=True))
    return total


def _min_denominator(sectors, energy, c, floor):
    """The smallest |E - E_M -+ lam| over the poles (M, e) whose weight
    (b_e . <M|a|c>)^2 exceeds `floor`; inf where none does."""
    best = np.inf
    for s in sectors:
        if len(s.rows) == 0:
            continue
        weight = (s.amplitudes(c).T @ s.b) ** 2
        seen = weight > floor
        if np.any(seen):
            best = min(best, float(np.abs(s.denominators(energy))[seen].min()))
    return best


def delta_roots(h_n, sectors, energies, vectors, method, numerics,
                keep=None):
    """The states (energies, vectors) of the P-space matrix h_n refined with
    Delta (module docstring), a `DeltaResult` in the order handed in.

    h_n: (dim K, dim K) dense; sectors: one entry per kind and spin,
    (kind, e_m, amp, lam, b), kind 'particle' or 'hole', e_m the spectrum of
    that sector, amp(u) a function returning a new array
    <M|a_u|K> (particle) or <M|a+_u|K> (hole) of shape (dim M, dim K), over
    the eigenstates M in the order of e_m (the kernel overwrites it), lam
    (n_levels,) and b (n_orbitals, n_levels). energies, vectors: states of
    h_n (vectors of any shape with dim K entries). method: 'lowdin' or
    'linearized'. numerics: a `Numerics` (None: the defaults), its
    downfolding section the settings. keep: None, or a function of one
    eigenvector of H_eff (flat, dim K) that says whether it is a state of
    the request, applied to the eigenvectors that extra['solutions_below']
    counts (refine passes the test of the requested spin); None counts every
    one. No sector at all is no Delta: the static states as handed in, Z = 1,
    and extra['solutions_below'] counts the states handed in below each
    one, which are the lowest of the static solve."""
    if method not in DENSE_MATRICES:
        raise ValueError(f'method={method!r}: the dense Delta methods are '
                         f'{sorted(DENSE_MATRICES)}')
    settings = numerics_or_default(numerics).downfolding
    static = np.array(energies, dtype=float).reshape(-1)
    shapes = [np.shape(v) for v in vectors]
    flat = [np.asarray(v, dtype=float).ravel() for v in vectors]
    flat = [v / np.linalg.norm(v) for v in flat]
    if len(flat) != len(static):
        raise ValueError(f'{len(static)} energies and {len(flat)} vectors')
    n = len(static)
    margin = settings.delta_degenerate
    if not sectors:
        # nothing is folded: the linearization error is zero and the point
        # is the static energy; the states handed in are the lowest of the
        # static solve, so the solutions below a state are those of them
        # below it, outside its degenerate set
        below = np.zeros(n, dtype=int)
        for members in degenerate_sets(static, numerics):
            lower = static < static[members].mean() - margin
            lower[members] = False
            below[members] = np.count_nonzero(lower)
        extra = {'solutions_below': below, 'near_pole': np.zeros(n, bool),
                 'above_pole': np.zeros(n, bool)}
        if method == 'linearized':
            extra.update(linearization_error=np.zeros(n),
                         linearization_point=static.copy())
        return DeltaResult(static.copy(), static, np.ones(n), np.full(n, np.inf),
                           [v.reshape(s) for v, s in zip(flat, shapes)],
                           np.ones(n, dtype=bool), extra)
    sectors = [_Sector(*s, settings.delta_conv) for s in sectors]
    h_n = np.asarray(h_n, dtype=float)
    solve = _lowdin if method == 'lowdin' else _linearized
    out_e, out_c, converged = np.empty(n), [None] * n, np.ones(n, dtype=bool)
    point, point_distance = np.full(n, np.nan), np.full(n, np.inf)
    below = np.zeros(n, dtype=int)
    for members in degenerate_sets(static, numerics):
        starts = np.column_stack([flat[k] for k in members])
        e_set, c_set, ok, e0, below[members] = solve(
            h_n, sectors, static[members], starts, settings, keep)
        if e0 is not None:
            # the linearization point, and how close it lies to a pole the
            # states handed in see
            point[members] = e0
            point_distance[members] = [
                _min_denominator(sectors, e0, k, settings.delta_weight_floor)
                for k in starts.T]
        for i, k in enumerate(members):
            c = c_set[:, i] / np.linalg.norm(c_set[:, i])
            out_e[k], out_c[k], converged[k] = e_set[i], (
                -c if np.dot(c, flat[k]) < 0 else c), ok
        if not ok:
            flag('delta_not_converged',
                 f'Delta ({method}): the states {list(members)} (static '
                 f'energies {np.round(static[members], 8).tolist()}) did not '
                 f'converge in {settings.delta_max_iter} steps to '
                 f'{settings.delta_conv:.1e} Ha; their last energies '
                 f'{np.round(out_e[members], 8).tolist()} are kept',
                 DeltaNotConverged)
    z = np.array([1.0 / (1.0 - _expectation(sectors, e, c, True))
                  for e, c in zip(out_e, out_c)])
    smallest = np.array([_min_denominator(sectors, e, c,
                                          settings.delta_weight_floor)
                         for e, c in zip(out_e, out_c)])
    near = (smallest < settings.delta_pole_warn) | (
        point_distance < settings.delta_pole_warn)
    lowest = min(s.lowest_pole() for s in sectors)
    extra = {'solutions_below': below, 'near_pole': near,
             'above_pole': out_e > lowest}
    if method == 'linearized':
        # Z |<c|H_eff(E)|c> - E|: the residual of the root in the exact
        # H_eff(E), turned into an energy by the Newton step of Loewdin
        residual = np.array([float(c @ h_n @ c)
                             + _expectation(sectors, e, c, False) - e
                             for e, c in zip(out_e, out_c)])
        extra.update(linearization_error=z * np.abs(residual),
                     linearization_point=point)
    for k in np.flatnonzero(near):
        where = (f'lies {smallest[k]:.2e} Ha from a pole E_M -+ lam'
                 if smallest[k] < settings.delta_pole_warn else
                 f'was linearized at {point[k]:.8f}, '
                 f'{point_distance[k]:.2e} Ha from a pole E_M -+ lam')
        flag('delta_near_pole',
             f'Delta ({method}): static state {k} at {out_e[k]:.8f} {where} (below '
             f'delta_pole_warn = {settings.delta_pole_warn} Ha); the energy '
             f'is finite, but it moves fast with E there and may belong to '
             f'the environment state' + _STATIC_INDEX, DeltaNearPoleWarning)
    for k in np.flatnonzero(z < QP_WINDOW_Z_MIN):
        flag('delta_satellite',
             f'Delta ({method}): static state {k} at {out_e[k]:.8f} has Z = '
             f'{z[k]:.3f}, below {QP_WINDOW_Z_MIN}: it is mostly environment, '
             f'a satellite, not a state of the active space' + _STATIC_INDEX,
             DeltaNearPoleWarning)
    for k in np.flatnonzero(extra.get('linearization_error', np.zeros(n))
                            > settings.delta_linearization_warn):
        flag('delta_linearization',
             f'Delta (linearized): static state {k} at {out_e[k]:.8f} has an '
             f'estimated linearization error of '
             f'{extra["linearization_error"][k]:.1e} Ha (Z |<c|H_eff(E)|c> - '
             f'E|), above delta_linearization_warn = '
             f'{settings.delta_linearization_warn} Ha; '
             f"delta_method='lowdin' gives the exact value" + _STATIC_INDEX,
             DeltaLinearizationWarning)
    return DeltaResult(out_e, static, z, smallest,
                       [c.reshape(s) for c, s in zip(out_c, shapes)],
                       converged, extra)


def degenerate_sets(energies, numerics):
    """The states as sets of indices refined together: sorted by energy, a
    set ends where the next energy is numerics.downfolding.delta_degenerate
    or more above the last (numerics None: the defaults). A state alone is a
    set of one."""
    tol = numerics_or_default(numerics).downfolding.delta_degenerate
    order = np.argsort(energies, kind='stable')
    sets, current = [], [int(order[0])]
    for k in order[1:]:
        if energies[k] - energies[current[-1]] < tol:
            current.append(int(k))
        else:
            sets.append(sorted(current))
            current = [int(k)]
    sets.append(sorted(current))
    return [np.array(s) for s in sets]


def _select(values, vecs, overlaps):
    """(energies, vectors, chosen) of the n roots whose vectors weigh most on the
    previous set: `overlaps` (n_roots, n), each root's overlaps with the n
    previous vectors in the metric of the method. The n roots are rotated
    among themselves to lie closest to the previous vectors (the orthogonal
    Procrustes rotation U V^T of the SVD of their overlap matrix), so that a
    degenerate set comes back as distinct states aligned to those handed in;
    each energy is the expectation value of the rotated root, the eigenvalue
    itself for a set of one or an exactly degenerate one; chosen, the
    indices of the n roots among `values`."""
    n = overlaps.shape[1]
    chosen = np.sort(np.argsort(-(overlaps ** 2).sum(1), kind='stable')[:n])
    u, _, vt = np.linalg.svd(overlaps[chosen])
    rotation = u @ vt
    return ((rotation ** 2).T @ values[chosen], vecs[:, chosen] @ rotation,
            chosen)


def _solutions_below(values, vecs, chosen, margin, keep):
    """How many eigenvalues of H_eff(E) (values ascending, vecs their
    eigenvectors), outside `chosen`, the set's own, lie more than `margin`
    below the mean of the set's own eigenvalues and are accepted by `keep`
    (None: every one). An eigenvalue within `margin` of the set's own is a
    degenerate partner, not a solution below it."""
    own = float(np.mean(values[chosen]))
    others = np.setdiff1d(np.flatnonzero(values < own - margin), chosen)
    if keep is None:
        return len(others)
    return sum(1 for i in others if keep(vecs[:, i]))


def _lowdin(h_n, sectors, energies, starts, settings, keep):
    """(energies, vectors, converged, None, below) of one degenerate set (one
    state, or the partners of a degenerate one), starts (dim K, n): at the
    set's common E, the n eigenvalues of H_eff(E) whose vectors overlap most
    with the previous ones, E their mean, until one step moves E by less
    than delta_conv. For one state this is the eigenvalue whose vector
    overlaps most with the previous one. below: `_solutions_below` from the
    spectrum of the last step, whose E is within delta_conv of the result."""
    energy, previous, converged = float(np.mean(energies)), starts, False
    vecs = None
    for _ in range(settings.delta_max_iter):
        vecs = None                    # one dim(K)^2 matrix less in the fold
        h = _fold(sectors, energy, len(h_n))
        h += h_n
        values, vecs = np.linalg.eigh(h)
        del h
        found, previous, chosen = _select(values, vecs, vecs.T @ previous)
        step = float(np.mean(found)) - energy
        energy = float(np.mean(found))
        if abs(step) < settings.delta_conv:
            converged = True
            break
    below = _solutions_below(values, vecs, chosen, settings.delta_degenerate,
                             keep)
    return found, previous, converged, None, below


def _linearized(h_n, sectors, energies, starts, settings, keep):
    """(energies, vectors, True, E0, below) of one degenerate set, starts
    (dim K, n): the Z-estimate E0 = E_K + g / (1 - g') (the mean over the
    set), then the n roots of (H_N + D0 - E0 D1) c = E (1 - D1) c with the
    largest weight <c|(1 - D1)|K> on the states handed in. below:
    `_solutions_below` from the spectrum of H_eff at the set's result E (one
    more fold), the set's own eigenvalues those whose vectors weigh most on
    the roots found."""
    e0 = float(np.mean([
        e + _expectation(sectors, e, k, False)
        / (1.0 - _expectation(sectors, e, k, True))
        for e, k in zip(energies, starts.T)]))
    metric = _fold(sectors, e0, len(h_n), derivative=True)
    metric *= -1.0                                       # -D1
    a = _fold(sectors, e0, len(h_n))
    a += h_n
    a += e0 * metric                                     # H_N + D0 - E0 D1
    metric[np.diag_indices_from(metric)] += 1.0          # 1 - D1
    target = metric @ starts
    values, vecs = scipy.linalg.eigh(a, metric, overwrite_a=True,
                                     overwrite_b=True)
    del a, metric
    found, vectors, _ = _select(values, vecs, vecs.T @ target)
    del vecs
    h = _fold(sectors, float(np.mean(found)), len(h_n))
    h += h_n
    values, vecs = np.linalg.eigh(h)
    del h
    unit = vectors / np.linalg.norm(vectors, axis=0)
    _, _, chosen = _select(values, vecs, vecs.T @ unit)
    below = _solutions_below(values, vecs, chosen, settings.delta_degenerate,
                             keep)
    return found, vectors, True, e0, below


# ------------------------------------------------------------ the backends
def _require_sector_operations(engine):
    """Refuse (ValueError) an engine without `has_sector_operations`, naming
    the engine to use and the static forms of Delta, which need none."""
    if not getattr(engine, 'has_sector_operations', False):
        raise ValueError(
            f'{type(engine).__name__} has no sector operations (the '
            f'sectors with one electron less or more), which a dynamic '
            f'Delta method takes from its engine: use PyscfFCISolver as '
            f"the engine, or a static form of Delta, 'static_sa' or "
            f"'static_mf', which changes h and runs with any backend")


class EnergyDependentSolver:
    """A dynamic backend: refines the states of a static solve with Delta.

    engine: a static backend with `has_sector_operations` (an engine without
    them is refused, naming the static forms 'static_sa' and 'static_mf',
    which change h and run with any backend); sides: the plain data of
    `Coupling.sides()`, [(kind, sigma, lam, b), ...], sigma None for both
    spins; numerics: a `Numerics` (None: the defaults); d_sa: the static
    D_SA (D, c), read by the remainder method alone (`RemainderSolver`),
    ignored by the dense ones."""

    #: the name of the method in `DYNAMIC_METHODS`; None in the base class
    method = None

    def __init__(self, engine, sides, numerics=None, d_sa=None):
        _require_sector_operations(engine)
        self.engine = engine
        self.sides = [(kind, sigma, np.asarray(lam, dtype=float),
                       np.asarray(b, dtype=float))
                      for kind, sigma, lam, b in sides]
        self.numerics = numerics_or_default(numerics)
        self.d_sa = d_sa

    def planned_sectors(self, norb, nelec):
        """[(kind, sigma, target nelec, lam, b)]: one per side and spin (both
        spins for sigma None), a sector that does not exist (no electron of
        that spin to remove, or the spin full) left out."""
        out = []
        for kind, sigma, lam, b in self.sides:
            for s in ((0, 1) if sigma is None else (sigma,)):
                target = list(nelec)
                target[s] += -1 if kind == 'particle' else 1
                if 0 <= target[s] <= norb:
                    out.append((kind, s, tuple(target), lam, b))
        return out

    def refine(self, h1, eri, norb, nelec, energies, vectors, spin=None):
        """The states (energies, vectors) of the static Hamiltonian (h1, eri)
        in the sector nelec, as the engine's `roots` returned them, refined
        with Delta: a `DeltaResult`, energies without a constant, in the
        order handed in, vectors the engine's CI arrays of unit norm.
        spin: None, or the 2S of the request the states belong to; then
        extra['solutions_below'] counts only the eigenvectors of H_eff of
        total spin S (`spin_square`, within FCI_SPIN_SQUARE_TOL of S(S+1)):
        the sector nelec also holds states of higher spin, which are not
        states of the request. Refused (ValueError) before anything is
        allocated when the dense estimate exceeds the limit
        (`dense_memory_gb`, `dense_memory_limit_gb`)."""
        if self.method is None:
            raise NotImplementedError('EnergyDependentSolver is the base of '
                                      'the dynamic backends')
        nelec = (int(nelec[0]), int(nelec[1]))
        plan = self.planned_sectors(norb, nelec)
        if not plan:
            return delta_roots(None, [], energies, vectors, self.method,
                               self.numerics)
        keep = None
        if spin is not None:
            want = 0.25 * spin * (spin + 2)
            shape = sector_shape(norb, nelec)

            def keep(v):
                return abs(self.spin_square(v.reshape(shape), norb, nelec)
                           - want) < FCI_SPIN_SQUARE_TOL
        dim_k = math.prod(sector_shape(norb, nelec))
        dims_m = [math.prod(sector_shape(norb, p[2])) for p in plan]
        estimate = dense_memory_gb(dim_k, dims_m, self.method)
        limit = dense_memory_limit_gb(self.numerics)
        if estimate > limit:
            raise ValueError(
                f"Delta '{self.method}' on the sector {nelec} of {norb} "
                f'orbitals (dimension {dim_k}, sectors {dims_m}) needs about '
                f'{estimate:.3g} GB, above the limit of {limit:.3g} GB '
                f'(numerics.layout.delta_dense_gb). Raise the limit, or use '
                f"'sa_remainder', which needs no full sector spectrum")
        h_n = self.engine.sector_matrix(h1, eri, norb, nelec)
        sectors = [self.sector(h1, eri, norb, nelec, *p) for p in plan]
        return delta_roots(h_n, sectors, energies, vectors, self.method,
                           self.numerics, keep)

    def spin_square(self, vector, norb, nelec):
        """<S^2> of a CI array of the sector nelec, from the engine's
        operators alone: S_z (S_z + 1) + |S_+ c|^2 / |c|^2 with
        S_+ = sum_p a+_{p alpha} a_{p beta}."""
        n_alpha, n_beta = nelec
        s_z = 0.5 * (n_alpha - n_beta)
        flip = 0.0
        if n_beta > 0 and n_alpha < norb:
            raised = sum(self.engine.create(
                self.engine.annihilate(vector, norb, nelec, p, 1), norb,
                (n_alpha, n_beta - 1), p, 0) for p in range(norb))
            flip = float(np.vdot(raised, raised))
        return s_z * (s_z + 1.0) + flip / float(np.vdot(vector, vector))

    def sector(self, h1, eri, norb, nelec, kind, sigma, target, lam, b):
        """(kind, e_m, amp, lam, b) of one sector for `delta_roots`: the
        spectrum of the sector `target` and amp(u) = <M|a_u|K> (particle) or
        <M|a+_u|K> (hole) from the engine's operator, whose matrix, a signed
        selection of determinants, is read once per orbital."""
        e_m, v_m = scipy.linalg.eigh(self.engine.sector_matrix(
            h1, eri, norb, target), overwrite_a=True)
        op = self.engine.annihilate if kind == 'particle' else self.engine.create
        rows = np.flatnonzero(np.any(np.asarray(b) != 0.0, axis=1))
        maps = {int(u): _selection(op, norb, nelec, int(u), sigma,
                                   len(e_m)) for u in rows}
        dim_k = math.prod(sector_shape(norb, nelec))

        def amp(u):
            source, sign, image = maps[int(u)]
            out = np.zeros((len(e_m), dim_k))
            out[:, source] = (v_m[image] * sign[:, None]).T
            return out
        return kind, e_m, amp, lam, b


def _selection(op, norb, nelec, orbital, sigma, dim_m):
    """(source, sign, image) of one creation or annihilation operator from
    the sector nelec to its target: op|source[i]> = sign[i] |image[i]>, in
    determinant indices, read with one call on the CI array that holds
    k + 1 at determinant k. Such an operator maps each determinant to at
    most one, with a sign, and no two to the same; anything else is
    refused."""
    dim_k = math.prod(sector_shape(norb, nelec))
    labels = np.arange(1, dim_k + 1, dtype=float).reshape(
        sector_shape(norb, nelec))
    moved = np.asarray(op(labels, norb, nelec, orbital, sigma)).ravel()
    if moved.size != dim_m:
        raise ValueError(f'the operator on orbital {orbital}, spin {sigma} '
                         f'returned {moved.size} entries for a sector of '
                         f'{dim_m}')
    image = np.flatnonzero(moved)
    value = moved[image]
    source = np.rint(np.abs(value)).astype(int) - 1
    if (np.abs(np.abs(value) - (source + 1)).max(initial=0.0) > 0
            or len(np.unique(source)) != len(source)):
        raise ValueError(f'the operator on orbital {orbital}, spin {sigma} '
                         f'is not a signed selection of determinants')
    return source, np.sign(value), image


class LowdinSolver(EnergyDependentSolver):
    """Exact Loewdin, the reference method: per state, E <- the eigenvalue of
    H_eff(E) whose vector overlaps most with the previous one, until one
    step moves E by less than delta_conv. A degenerate set is refined
    together (`delta_roots`).

    Reliable whenever it converges: the energy is then an exact eigenvalue
    of the active space coupled to the environment levels, at any strength
    of the coupling. It can fail to converge next to a pole, where H_eff(E)
    changes fast with E: after delta_max_iter steps the last energy is kept
    with DeltaNotConverged and the flag 'delta_not_converged'. Like every
    root, a root near a pole is flagged 'delta_near_pole', and one with
    Z below QP_WINDOW_Z_MIN 'delta_satellite'. Dense: the full spectrum of
    every sector with one electron less (more), one diagonalization of the
    sector of the states per step."""
    method = 'lowdin'


class LinearizedLowdinSolver(EnergyDependentSolver):
    """Linearized Loewdin: E0 = E_K + g / (1 - g') from g = <K|H_eff(E_K) -
    H_N|K> and g' = <K|dH_eff(E_K)|K>, then one generalized eigenproblem
    (H_N + D0 - E0 D1) c = E (1 - D1) c, D0 = H_eff(E0) - H_N and
    D1 = dH_eff(E0), and the root with the largest <c|(1 - D1)|K>. Dense,
    as `LowdinSolver`; not iterative, so it always returns.

    When it is reliable: H_eff(E) is replaced by its tangent at E0, so the
    energy is exact to first order in E - E0. For a weakly coupled state,
    Z close to 1, E0 is close to the self-consistent energy and the error is
    second order, far below any chemical scale; as Z drops (a state that
    leaks strongly into the environment) E0 moves away and the error grows
    quickly. Each root's estimate Z |<c|H_eff(E)|c> - E|, the residual of
    the root in the exact H_eff(E) turned into an energy by one Newton step,
    is in extra['linearization_error']; above
    numerics.downfolding.delta_linearization_warn the root is flagged
    'delta_linearization' (DeltaLinearizationWarning), and 'lowdin' gives
    its exact value. A linearization point within delta_pole_warn of a pole
    is flagged 'delta_near_pole': the tangent is then taken where H_eff(E)
    bends most, and the root may be a satellite ('delta_satellite', Z below
    QP_WINDOW_Z_MIN)."""
    method = 'linearized'


class RemainderSolver(EnergyDependentSolver):
    """The remainder method 'sa_remainder': a static Hamiltonian,
    h + D_SA, solved once for all the states, plus a first-order dynamic
    correction per state for what D_SA leaves out (spec 6; the benzene
    handoff, method 8). It needs no sector spectrum, only a short block
    Lanczos per state, and is the method for active spaces beyond the
    dense methods' memory limit.

    What it computes. d_sa = (D, c), refused (ValueError) when None, is the
    static form of Delta built beforehand from a declared set of states
    (`DownfoldedHamiltonian.prepare_delta`, `state_averaged_delta`): D a
    one-body matrix on the active orbitals and c the constant of its hole
    part. Each state K handed to `refine` is an eigenstate of h + D, the
    static Hamiltonian with D_SA in it, and its energy E_K the eigenvalue
    PLUS c: E_K is the static estimate of the state's energy, and the
    energy at which Delta is evaluated. `refine` itself takes h1, the
    Hamiltonian WITHOUT D. The remainder of K is the expectation value of
    the energy-dependent operator, both sides and both spins, minus that of
    its whole static stand-in, constant included,

        r_K(E)  = <K|Delta(E)|K> - (tr(D dm1_K) + c),
        dE_K    = r_K(E_K) / (1 - r'_K(E_K)),   Z_K = 1 / (1 - r'_K(E_K)),

    and the refined energy is E_K + dE_K (c is counted once: it is in E_K,
    and the remainder takes it out together with D). <K|Delta(E)|K> is
    sum over the poles of the removal (particle side) and attachment (hole
    side) Green's functions of K in the Hamiltonian h1,
    sum_{e, m} (b^T w)_em^2 / (E - p_m -+ lam_e), poles p_m and weights w
    of `removal_spectrum` and `attachment_spectrum` started from a_u|K>
    and a+_u|K> on the rows u of b that couple, at
    numerics.downfolding.delta_lanczos_depth blocks. dm1_K is the
    spin-summed one-body density matrix of K: from the engine's `rdms` when
    it has them (`has_rdms`), otherwise <K|E_tu|K> from the engine's
    annihilation operator.

    Its approximation. The vector of a state is that of h + D and is not
    changed: Delta - D_SA enters as one first-order (Z-renormalized)
    energy shift per state, a Newton step from E_K, and the relaxation of
    the state is the part D_SA already holds. It is not exact Loewdin: how
    close it comes depends on how well D_SA represents Delta at each
    state's energy, so on the states D_SA was built from (prepare_delta
    should declare the states that are solved, and a state far from that
    set is described worse).

    Its limit: strong coupling. The correction is first order in
    Delta - D_SA, which is accurate while the shift Delta gives a state is
    small against the distance between the state's removal and attachment
    energies and the environment levels lam, the denominators of Delta.
    Then D_SA is close to Delta at each state's energy, the states of
    h + D are close to the exact ones, and the correction is small. When
    the shift is comparable to that distance (strong coupling), D_SA,
    evaluated at the removal and attachment energies of the states without
    Delta and averaged over the declared states, overshoots: it is large
    for every state of the set and changes the states of h + D themselves,
    their character and even their order, so that a static state can be
    another state than the one exact Loewdin finds. A per-state scalar
    cannot undo a change of the state, and the energy can then be far from
    'lowdin'. This is not the case of a root next to a pole
    ('delta_near_pole'): no denominator needs to be small. The sign of it
    in the dense methods is a Z well below 1 on the declared states; the
    sign of it here is a large correction, and a state whose |dE_K|
    exceeds numerics.downfolding.delta_remainder_warn is flagged
    'delta_remainder_large' (DeltaRemainderWarning): for it use 'lowdin'
    or 'linearized' where the active space fits their memory limit.

    What it reports. Every root is reported with converged True
    (nothing iterates), Z, and the smallest denominator |E - p_m -+ lam|
    over the Lanczos poles whose weight (b^T w)^2 exceeds
    delta_weight_floor at the refined energy. Below
    numerics.downfolding.delta_pole_warn, at the refined energy or at E_K,
    where r and r' were taken, the root is flagged 'delta_near_pole'
    (DeltaNearPoleWarning), and with Z below QP_WINDOW_Z_MIN
    'delta_satellite'. extra['near_pole'] marks the first; extra
    ['lanczos_change'] holds |dE_K(depth) - dE_K(depth - 1)|, the change
    of the correction when one block less is used, from the same Lanczos
    run (zero once the Krylov space is exhausted before the depth): a large
    value means the depth is too small for that state. The vectors
    returned are those handed in, of unit norm.

    What it does not report. It never forms H_eff(E) as a matrix, so the
    number of solutions of the Delta problem below a root, which the dense
    methods count (extra['solutions_below']), is not computed:
    `solvers.solve.solve` marks every root of this method as unchecked for
    missing states (out['crossing_unchecked'] True). The states are those
    of h + D and their number and spin are decided by the static solve, as
    for every dynamic backend."""
    method = 'sa_remainder'

    def __init__(self, engine, sides, numerics=None, d_sa=None):
        if d_sa is None:
            raise ValueError(
                "'sa_remainder' needs d_sa = (D, c), the static D_SA built "
                "once from a declared set of states: h = "
                "h.prepare_delta(((nroots, spin), ...)) first "
                "(DownfoldedHamiltonian.prepare_delta).")
        super().__init__(engine, sides, numerics, d_sa)
        d, c = d_sa
        self.d, self.c = np.asarray(d, dtype=float), float(c)
        if self.d.ndim != 2 or self.d.shape[0] != self.d.shape[1]:
            raise ValueError(f'd_sa: D of shape {self.d.shape} is not a '
                             f'square one-body matrix')

    def refine(self, h1, eri, norb, nelec, energies, vectors, spin=None):
        """The states (vectors, eigenstates of h1 + D in the sector nelec,
        as the engine's `roots` returned them) and their energies E_K
        (eigenvalues plus c) refined by the remainder (class docstring): a
        `DeltaResult` in the order handed in, energies E_K + dE_K, static
        E_K. h1 and eri are the static Hamiltonian WITHOUT D. spin is taken
        for the common interface and not used: no count of solutions is
        made. A D_SA on another number of orbitals is refused
        (ValueError)."""
        if self.d.shape != (norb, norb):
            raise ValueError(f'd_sa: D of shape {self.d.shape} on {norb} '
                             f'active orbitals')
        settings = self.numerics.downfolding
        nelec = (int(nelec[0]), int(nelec[1]))
        static = np.array(energies, dtype=float).reshape(-1)
        shapes = [np.shape(v) for v in vectors]
        flat = [np.asarray(v, dtype=float).ravel() for v in vectors]
        flat = [v / np.linalg.norm(v) for v in flat]
        if len(flat) != len(static):
            raise ValueError(f'{len(static)} energies and {len(flat)} vectors')
        n, shape = len(static), sector_shape(norb, nelec)
        plan = self.planned_sectors(norb, nelec)
        short = settings.delta_lanczos_depth - 1
        shift, z, change = np.empty(n), np.empty(n), np.empty(n)
        smallest, at_point = np.empty(n), np.empty(n)
        for k in range(n):
            vector = flat[k].reshape(shape)
            stand_in = float(np.einsum(
                'tu,ut->', self.d, self._one_body_density(vector, norb,
                                                          nelec))) + self.c
            runs = []
            for kind, sigma, _, lam, b in plan:
                rows = np.flatnonzero(np.any(b != 0.0, axis=1))
                runs.append((1.0 if kind == 'particle' else -1.0, lam,
                             b[rows], _one_electron_krylov(
                                 self.engine, -1 if kind == 'particle' else 1,
                                 vector, h1, eri, norb, nelec, rows, sigma,
                                 self.numerics)))
            poles = [_Poles(sign, lam, b, *krylov.spectrum())
                     for sign, lam, b, krylov in runs]
            shift[k], z[k] = _remainder_step(poles, static[k], stand_in,
                                             settings)
            fewer = [_Poles(sign, lam, b, *krylov.spectrum(short))
                     for sign, lam, b, krylov in runs]
            change[k] = abs(shift[k] - _remainder_step(
                fewer, static[k], stand_in, settings)[0])
            smallest[k] = min((p.min_denominator(static[k] + shift[k],
                                                 settings.delta_weight_floor)
                               for p in poles), default=np.inf)
            at_point[k] = min((p.min_denominator(static[k],
                                                 settings.delta_weight_floor)
                               for p in poles), default=np.inf)
        out_e = static + shift
        near = (smallest < settings.delta_pole_warn) | (
            at_point < settings.delta_pole_warn)
        for k in np.flatnonzero(near):
            where = (f'lies {smallest[k]:.2e} Ha from a pole E_M -+ lam'
                     if smallest[k] < settings.delta_pole_warn else
                     f'was evaluated at its static energy {static[k]:.8f}, '
                     f'{at_point[k]:.2e} Ha from a pole E_M -+ lam')
            flag('delta_near_pole',
                 f'Delta (sa_remainder): static state {k} at {out_e[k]:.8f} '
                 f'{where} (Lanczos poles; below delta_pole_warn = '
                 f'{settings.delta_pole_warn} Ha); the energy is finite, but '
                 f'it moves fast with E there and may belong to the '
                 f'environment state' + _STATIC_INDEX, DeltaNearPoleWarning)
        for k in np.flatnonzero(z < QP_WINDOW_Z_MIN):
            flag('delta_satellite',
                 f'Delta (sa_remainder): static state {k} at {out_e[k]:.8f} '
                 f'has Z = {z[k]:.3f}, below {QP_WINDOW_Z_MIN}: it is mostly '
                 f'environment, a satellite, not a state of the active space'
                 + _STATIC_INDEX, DeltaNearPoleWarning)
        for k in np.flatnonzero(np.abs(shift) > settings.delta_remainder_warn):
            flag('delta_remainder_large',
                 f'Delta (sa_remainder): static state {k} at {out_e[k]:.8f} '
                 f'has a correction |dE| = {abs(shift[k]):.2e} Ha, above '
                 f'delta_remainder_warn = {settings.delta_remainder_warn} '
                 f'Ha: the coupling is strong there, D_SA has changed the '
                 f'state itself, and the first-order correction may be far '
                 f"from exact Loewdin; 'lowdin' or 'linearized' gives the "
                 f'energy-dependent result where the active space fits '
                 f'their memory limit' + _STATIC_INDEX,
                 DeltaRemainderWarning)
        return DeltaResult(out_e, static, z, smallest,
                           [v.reshape(s) for v, s in zip(flat, shapes)],
                           np.ones(n, dtype=bool),
                           {'near_pole': near, 'lanczos_change': change})

    def _one_body_density(self, vector, norb, nelec):
        """The spin-summed dm1 of a CI array of unit norm, dm1[p, q] =
        <q+ p>: the engine's `rdms` when it has them, otherwise
        sum_sigma <a_{q sigma} K|a_{p sigma} K> from its `annihilate`."""
        if getattr(self.engine, 'has_rdms', False):
            return np.asarray(self.engine.rdms(vector, norb, nelec,
                                               spin=None)[0])
        dm1 = np.zeros((norb, norb))
        for sigma in (0, 1):
            if nelec[sigma] == 0:
                continue
            a = np.array([np.asarray(self.engine.annihilate(
                vector, norb, nelec, p, sigma)).ravel() for p in range(norb)])
            dm1 += a @ a.T
        return dm1


@dataclass(frozen=True)
class _Poles:
    """The Lanczos poles of one side and spin as `RemainderSolver` uses
    them: sign +1 (particle) or -1 (hole), the levels lam, b on the rows
    the Lanczos started from, and its poles and weights."""
    sign: float
    lam: np.ndarray
    b: np.ndarray
    poles: np.ndarray
    weights: np.ndarray

    def denominators(self, energy):
        """E - p_m - lam_e (particle) or E - p_m + lam_e (hole),
        (n_levels, m)."""
        return (energy - self.poles)[None, :] - self.sign * self.lam[:, None]

    def strengths(self):
        """(b^T w)_em^2, the weight of each pole m and level e."""
        return (self.b.T @ self.weights) ** 2

    def min_denominator(self, energy, floor):
        """The smallest |E - p_m -+ lam_e| over the poles whose weight
        exceeds `floor`; inf where none does."""
        seen = self.strengths() > floor
        if not np.any(seen):
            return np.inf
        return float(np.abs(self.denominators(energy))[seen].min())


def _remainder_step(poles, energy, stand_in, settings):
    """(dE, Z) of one state: g = <K|Delta(E)|K> and g' its derivative from
    the `_Poles` of every side and spin, r = g - stand_in, Z = 1 / (1 - g'),
    dE = r Z. A denominator smaller in size than delta_conv is taken at that
    size with its sign (+ for an exact zero), as in the dense kernel, so a
    state on a pole gives a finite energy."""
    g, dg = 0.0, 0.0
    for p in poles:
        if p.poles.size == 0:
            continue
        den = p.denominators(energy)
        small = np.abs(den) < settings.delta_conv
        if np.any(small):
            den = np.where(small, np.where(den < 0, -settings.delta_conv,
                                           settings.delta_conv), den)
        strength = p.strengths()
        g += float(np.sum(strength / den))
        dg -= float(np.sum(strength / den ** 2))
    z = 1.0 / (1.0 - dg)
    return (g - stand_in) * z, z


#: the dynamic backends by the name of their method
DYNAMIC_METHODS = {'lowdin': LowdinSolver,
                   'linearized': LinearizedLowdinSolver,
                   'sa_remainder': RemainderSolver}


# ------------------------------- the Green's functions of a state by Lanczos
def block_lanczos(matvec, start, numerics):
    """(poles (m,), weights (n_start, m)): the Green's function of a real
    symmetric H between the start vectors s_t by block Lanczos,

        G_tu(z) = sum_m weights[t, m] weights[u, m] / (z - poles[m]),

    which approximates s_t^T (z - H)^-1 s_u and equals it, up to the
    dropped columns below, once the Krylov space of the start vectors is
    exhausted. It sees the product and the start vectors alone.

    matvec: v -> H v on one vector of shape (dim,), returning shape (dim,);
    start: (dim, n_start), the start vectors as columns; numerics: a
    `Numerics` (None: the defaults), whose downfolding.delta_lanczos_depth
    is the largest number of blocks and downfolding.delta_rank_tol the
    threshold below.

    A start vector whose norm is below delta_rank_tol is set to zero. The
    start block S is factorized by QR with column pivoting, and the columns
    from the first j with |R_jj| below delta_rank_tol on are dropped (with
    pivoting |R_jj| does not increase): S = Q_0 B, Q_0 the r_0
    kept orthonormal columns and B (r_0, n_start) the kept rows of R, taken
    back to the order of the start vectors. With pivoting, a dropped start
    vector is one whose part outside the span of the kept ones has a norm
    below delta_rank_tol; B replaces it by its projection on that span. The
    block j + 1 is the same QR of H Q_j after every earlier block has been
    projected out of it twice (full reorthogonalization), with the same
    columns dropped: a dropped column is a direction of the Krylov space
    whose part outside the blocks built so far has a norm below
    delta_rank_tol, and G misses what H reaches only through it. Its kept
    rows of R are the coupling block C_{j+1} = Q_{j+1}^T H Q_j. Blocks are
    added until there are delta_lanczos_depth of them, or until a new block
    has no column left (the Krylov space is exhausted). poles are the
    eigenvalues of the block-tridiagonal T, diagonal blocks Q_j^T H Q_j
    (symmetrized) and off-diagonal blocks C_j, so m is the number of kept
    columns of all blocks. weights = B^T Y_0, Y_0 the rows of the
    eigenvectors of T on the start block: weights[t, m] is the projection
    of s_t on the m-th Ritz vector, rotated back through B.

    With no column dropped, d blocks reproduce the moments
    sum_m weights[t, m] weights[u, m] poles[m]^n = s_t^T H^n s_u for
    n = 0 .. 2d - 1. weights has one row per start vector, in the order
    given, those set to zero or dropped included; a start vector set to
    zero has zero weights in every pole. A start block with no column left
    (every start vector zero, or no start vector) gives poles (0,) and
    weights (n_start, 0)."""
    return _krylov(matvec, start, numerics).spectrum()


class _Krylov:
    """The blocks of one run of `block_lanczos`: the diagonal blocks
    Q_j^T H Q_j, the couplings C_{j+1} and B, the kept rows of R of the
    start block (n_start columns). `spectrum(n)` is the result of the same
    run stopped after n blocks, which is what a run at depth n gives: the
    first n blocks do not depend on the depth."""

    def __init__(self, diagonal, coupling, coef):
        self.diagonal, self.coupling, self.coef = diagonal, coupling, coef

    @property
    def blocks(self):
        """The number of blocks built."""
        return len(self.diagonal)

    def spectrum(self, n_blocks=None):
        """(poles, weights) of `block_lanczos` from the first n_blocks
        blocks (None: every block built); no block gives poles (0,) and
        weights (n_start, 0)."""
        n = self.blocks if n_blocks is None else min(n_blocks, self.blocks)
        if n == 0:
            return np.zeros(0), np.zeros((self.coef.shape[1], 0))
        diagonal, coupling = self.diagonal[:n], self.coupling[:n - 1]
        sizes = [d.shape[0] for d in diagonal]
        edges = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
        t = np.zeros((edges[-1], edges[-1]))
        for j, d in enumerate(diagonal):
            t[edges[j]:edges[j + 1], edges[j]:edges[j + 1]] = d
        for j, c in enumerate(coupling):
            t[edges[j + 1]:edges[j + 2], edges[j]:edges[j + 1]] = c
            t[edges[j]:edges[j + 1], edges[j + 1]:edges[j + 2]] = c.T
        poles, vecs = np.linalg.eigh(t)
        return poles, self.coef.T @ vecs[:edges[1]]


def _krylov(matvec, start, numerics):
    """The `_Krylov` of `block_lanczos` (its arguments and conventions)."""
    settings = numerics_or_default(numerics).downfolding
    start = np.array(start, dtype=float)
    if start.ndim != 2:
        raise ValueError(f'start of shape {start.shape}: (dim, n_start), '
                         f'the start vectors as columns')
    start[:, np.linalg.norm(start, axis=0) < settings.delta_rank_tol] = 0.0
    q, coef = _kept_columns(start, settings)
    if q.shape[1] == 0:
        return _Krylov([], [], np.zeros((0, start.shape[1])))
    blocks, diagonal, coupling = [q], [], []
    while True:
        hq = np.column_stack([np.asarray(matvec(c), dtype=float).ravel()
                              for c in q.T])
        a = q.T @ hq
        diagonal.append(0.5 * (a + a.T))
        if len(blocks) == settings.delta_lanczos_depth:
            break
        basis = np.hstack(blocks)
        for _ in range(2):
            hq -= basis @ (basis.T @ hq)
        q, c = _kept_columns(hq, settings)
        if q.shape[1] == 0:
            break
        blocks.append(q)
        coupling.append(c)
    return _Krylov(diagonal, coupling, coef)


def _kept_columns(block, settings):
    """(Q, C) of one block of Lanczos: the QR factorization with column
    pivoting of `block` (dim, k), the columns from the first with
    |R_jj| < settings.delta_rank_tol on dropped (settings: the downfolding
    section of a `Numerics`), so that block = Q C up to the dropped part;
    Q (dim, r) orthonormal, C (r, k) the kept rows of R in the order of the
    columns of `block`. r = 0 for a block with no rows, no columns, or no
    column above the threshold."""
    dim, k = block.shape
    if dim == 0 or k == 0:
        return np.zeros((dim, 0)), np.zeros((0, k))
    q, r, perm = scipy.linalg.qr(block, mode='economic', pivoting=True)
    small = np.abs(np.diag(r)) < settings.delta_rank_tol
    rank = int(np.argmax(small)) if np.any(small) else len(small)
    c = np.empty((rank, k))
    c[:, perm] = r[:rank]
    return q[:, :rank], c


def removal_spectrum(engine, vector, h1, eri, norb, nelec, orbitals, sigma,
                     numerics):
    """(poles (m,), weights (len(orbitals), m)): the removal Green's
    function of the CI state `vector`,

        G_tu(z) = <K|a+_{t sigma} (z - H^(N-1))^-1 a_{u sigma}|K>
                ~ sum_m weights[t, m] weights[u, m] / (z - poles[m]),

    by `block_lanczos` with H^(N-1) the engine's `sector_product` of the
    sector with one electron of spin sigma less, started from the vectors
    a_{u sigma}|K> of the engine's `annihilate`, one per entry u of
    `orbitals`, in that order. Exact, up to the dropped columns of
    `block_lanczos`, once the Krylov space is exhausted; the depth is
    numerics.downfolding.delta_lanczos_depth (blocks), and with no column
    dropped d blocks reproduce the moments <K|a+_t H^n a_u|K>,
    n = 0 .. 2d - 1. An orbital whose a_{u sigma}|K> has a norm below
    numerics.downfolding.delta_rank_tol (for K of unit norm, an occupation
    of u sigma in K below delta_rank_tol^2) has zero weights in every
    pole.

    engine: a static backend with `has_sector_operations` (one without is
    refused, ValueError, as by `EnergyDependentSolver`); vector: a CI array
    of the sector nelec = (n_alpha, n_beta) of the integrals (h1, eri) in
    norb orbitals, used as given (G scales with its norm squared; the
    engine's roots have norm 1); sigma: the spin of the removed electron,
    0 (alpha) or 1 (beta); numerics: a `Numerics` (None: the defaults).
    poles carry no constant, as the engine's energies. No electron of spin
    sigma to remove: no sector, poles (0,) and weights
    (len(orbitals), 0). The kernel works on flattened CI vectors; the
    engine's product sees CI arrays of the sector's shape."""
    return _one_electron_krylov(engine, -1, vector, h1, eri, norb, nelec,
                                orbitals, sigma, numerics).spectrum()


def attachment_spectrum(engine, vector, h1, eri, norb, nelec, orbitals,
                        sigma, numerics):
    """(poles (m,), weights (len(orbitals), m)): the attachment Green's
    function of the CI state `vector`,

        G_tu(z) = <K|a_{t sigma} (z - H^(N+1))^-1 a+_{u sigma}|K>
                ~ sum_m weights[t, m] weights[u, m] / (z - poles[m]),

    the mirror of `removal_spectrum` (the same arguments, depth, threshold
    and conventions): H^(N+1) the engine's `sector_product` of the sector
    with one electron of spin sigma more, started from the vectors
    a+_{u sigma}|K> of the engine's `create`. An orbital whose
    a+_{u sigma}|K> has a norm below numerics.downfolding.delta_rank_tol
    (for K of unit norm, a hole of u sigma in K below delta_rank_tol^2) has
    zero weights in every pole. The spin
    sigma full (n_sigma = norb): no sector, poles (0,) and weights
    (len(orbitals), 0)."""
    return _one_electron_krylov(engine, +1, vector, h1, eri, norb, nelec,
                                orbitals, sigma, numerics).spectrum()


def _one_electron_krylov(engine, step, vector, h1, eri, norb, nelec,
                         orbitals, sigma, numerics):
    """The `_Krylov` of `removal_spectrum` (step -1) or
    `attachment_spectrum` (step +1); a sector that does not exist has no
    block."""
    _require_sector_operations(engine)
    if sigma not in (0, 1):
        raise ValueError(f'sigma={sigma!r}: 0 (alpha) or 1 (beta)')
    nelec = (int(nelec[0]), int(nelec[1]))
    orbitals = [int(u) for u in orbitals]
    target = list(nelec)
    target[sigma] += step
    target = tuple(target)
    shape = sector_shape(norb, target)
    if 0 in shape:
        return _Krylov([], [], np.zeros((0, len(orbitals))))
    op = engine.annihilate if step < 0 else engine.create
    start = np.zeros((math.prod(shape), len(orbitals)))
    for i, u in enumerate(orbitals):
        start[:, i] = np.asarray(op(vector, norb, nelec, u, sigma)).ravel()
    product = engine.sector_product(h1, eri, norb, target)

    def matvec(v):
        return np.asarray(product(v.reshape(shape))).ravel()
    return _krylov(matvec, start, numerics)
