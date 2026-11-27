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

- 'lowdin' (`LowdinSolver`): per state, E <- the eigenvalue of H_eff(E)
  whose vector overlaps most with the previous one, to self-consistency;
- 'linearized' (`LinearizedLowdinSolver`): per state a Z-estimate E0, then
  the generalized eigenproblem (H_N + D0 - E0 D1) c = E (1 - D1) c with
  D0 = H_eff(E0) - H_N and D1 = dH_eff/dE(E0).

Each root reports its static energy, Z = 1 / (1 - <c|dH_eff(E)|c>) and the
smallest denominator |E - E_M -+ lam| over the poles it sees; one below
numerics.downfolding.delta_pole_warn raises DeltaNearPoleWarning and the
flag 'delta_near_pole'. The algebra is the kernel `delta_roots`, on dense
arrays only, which knows neither the engine nor the spin structure of the
integrals: it is tested with two spin channels and both kinds of side on a
model.
"""
import math
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg

from src.Base.numerical_flags import flag
from src.Base.numerics.constants import (BYTES_PER_GB,
                                         DOWNFOLDING_DELTA_DENSE_GB, MB_PER_GB)
from src.Base.numerics.numerics import numerics_or_default
from src.Base.utils.parallel.memory import allocation_max_memory_mb
from src.EffectiveHamiltonian.solvers.backends import sector_shape

#: the dense methods and the number of dim(K)^2 matrices each holds
DENSE_MATRICES = {'lowdin': 4, 'linearized': 6}


class DeltaNearPoleWarning(RuntimeWarning):
    """A root of a Delta solve whose smallest denominator |E - E_M -+ lam|
    over the poles it sees is below numerics.downfolding.delta_pole_warn:
    the energy is finite, but it moves fast with E there, and the root may
    belong to the environment state rather than to the active one."""


class DeltaNotConverged(RuntimeWarning):
    """Exact Loewdin ran out of steps (numerics.downfolding.delta_max_iter)
    before one step moved E by less than delta_conv; the last E is kept."""


@dataclass(frozen=True)
class DeltaResult:
    """The states of one static solve refined with Delta, in the order they
    were handed in.

    energies (nroots,): with Delta, without any constant; static (nroots,):
    the energies handed in; z (nroots,): 1 / (1 - <c|dH_eff(E)|c>);
    min_denominator (nroots,): the smallest |E - E_M -+ lam| over the poles
    whose weight (b_e . <M|a|c>)^2 exceeds delta_weight_floor (inf where
    none does); vectors: one array per root, of the shape the vector handed
    in had, unit Euclidean norm; converged (nroots,) bool; extra: what one
    method reports beyond these."""
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
    """One entry of `sectors` with the rows of b that couple."""

    def __init__(self, kind, e_m, amp, lam, b):
        if kind not in ('particle', 'hole'):
            raise ValueError(f"kind={kind!r}: 'particle' or 'hole'")
        self.sign = 1.0 if kind == 'particle' else -1.0
        self.e_m, self.amp = np.asarray(e_m, dtype=float), amp
        self.lam = np.asarray(lam, dtype=float)
        b = np.asarray(b, dtype=float)
        self.rows = np.flatnonzero(np.any(b != 0.0, axis=1))
        self.b = b[self.rows]

    def denominators(self, energy):
        """E - E_M - lam (particle) or E - E_M + lam (hole), (dim M, n_lev)."""
        return (energy - self.e_m)[:, None] - self.sign * self.lam[None, :]

    def delta(self, energy, derivative):
        """Delta_tu(E - E_M) over the coupled rows, (dim M, r, r); its
        derivative in E with `derivative`."""
        den = self.denominators(energy)
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


def delta_roots(h_n, sectors, energies, vectors, method, numerics):
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
    downfolding section the settings. No sector at all is no Delta: the
    static states as handed in, Z = 1."""
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
    if not sectors:
        return DeltaResult(static.copy(), static, np.ones(n), np.full(n, np.inf),
                           [v.reshape(s) for v, s in zip(flat, shapes)],
                           np.ones(n, dtype=bool))
    sectors = [_Sector(*s) for s in sectors]
    h_n = np.asarray(h_n, dtype=float)
    solve = _lowdin if method == 'lowdin' else _linearized
    out_e, out_c, converged = np.empty(n), [None] * n, np.ones(n, dtype=bool)
    for members in degenerate_sets(static, numerics):
        starts = np.column_stack([flat[k] for k in members])
        e_set, c_set, ok = solve(h_n, sectors, static[members], starts,
                                 settings)
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
    for k in np.flatnonzero(smallest < settings.delta_pole_warn):
        flag('delta_near_pole',
             f'Delta ({method}): state {k} at {out_e[k]:.8f} lies '
             f'{smallest[k]:.2e} Ha from a pole E_M -+ lam (below '
             f'delta_pole_warn = {settings.delta_pole_warn} Ha); the energy '
             f'is finite, and may belong to the environment state',
             DeltaNearPoleWarning)
    return DeltaResult(out_e, static, z, smallest,
                       [c.reshape(s) for c, s in zip(out_c, shapes)],
                       converged)


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
    """(energies, vectors) of the n roots whose vectors weigh most on the
    previous set: `overlaps` (n_roots, n), each root's overlaps with the n
    previous vectors in the metric of the method. The n roots are rotated
    among themselves to lie closest to the previous vectors (the orthogonal
    Procrustes rotation U V^T of the SVD of their overlap matrix), so that a
    degenerate set comes back as distinct states aligned to those handed in;
    each energy is the expectation value of the rotated root, the eigenvalue
    itself for a set of one or an exactly degenerate one."""
    n = overlaps.shape[1]
    chosen = np.sort(np.argsort(-(overlaps ** 2).sum(1), kind='stable')[:n])
    u, _, vt = np.linalg.svd(overlaps[chosen])
    rotation = u @ vt
    return (rotation ** 2).T @ values[chosen], vecs[:, chosen] @ rotation


def _lowdin(h_n, sectors, energies, starts, settings):
    """(energies, vectors, converged) of one degenerate set (one state, or
    the partners of a degenerate one), starts (dim K, n): at the set's
    common E, the n eigenvalues of H_eff(E) whose vectors overlap most with
    the previous ones, E their mean, until one step moves E by less than
    delta_conv. For one state this is the eigenvalue whose vector overlaps
    most with the previous one."""
    energy, previous, converged = float(np.mean(energies)), starts, False
    for _ in range(settings.delta_max_iter):
        h = _fold(sectors, energy, len(h_n))
        h += h_n
        values, vecs = np.linalg.eigh(h)
        del h
        found, previous = _select(values, vecs, vecs.T @ previous)
        del vecs
        step = float(np.mean(found)) - energy
        energy = float(np.mean(found))
        if abs(step) < settings.delta_conv:
            converged = True
            break
    return found, previous, converged


def _linearized(h_n, sectors, energies, starts, settings):
    """(energies, vectors, True) of one degenerate set, starts (dim K, n):
    the Z-estimate E0 = E_K + g / (1 - g') (the mean over the set), then the
    n roots of (H_N + D0 - E0 D1) c = E (1 - D1) c with the largest weight
    <c|(1 - D1)|K> on the states handed in."""
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
    found, vectors = _select(values, vecs, vecs.T @ target)
    return found, vectors, True


# ------------------------------------------------------------ the backends
class EnergyDependentSolver:
    """A dynamic backend: refines the states of a static solve with Delta.

    engine: a static backend with `has_sector_operations` (an engine without
    them is refused, naming the static forms 'static_sa' and 'static_mf',
    which change h and run with any backend); sides: the plain data of
    `Coupling.sides()`, [(kind, sigma, lam, b), ...], sigma None for both
    spins; numerics: a `Numerics` (None: the defaults); d_sa: the static
    D_SA, read by the remainder method alone."""

    #: the name of the method in `DYNAMIC_METHODS`; None in the base class
    method = None

    def __init__(self, engine, sides, numerics=None, d_sa=None):
        if not getattr(engine, 'has_sector_operations', False):
            raise ValueError(
                f'{type(engine).__name__} has no sector operations (the '
                f'sectors with one electron less or more), which a dynamic '
                f'Delta method takes from its engine: use PyscfFCISolver as '
                f"the engine, or a static form of Delta, 'static_sa' or "
                f"'static_mf', which changes h and runs with any backend")
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

    def refine(self, h1, eri, norb, nelec, energies, vectors):
        """The states (energies, vectors) of the static Hamiltonian (h1, eri)
        in the sector nelec, as the engine's `roots` returned them, refined
        with Delta: a `DeltaResult`, energies without a constant, in the
        order handed in, vectors the engine's CI arrays of unit norm.
        Refused (ValueError) before anything is allocated when the dense
        estimate exceeds the limit (`dense_memory_gb`,
        `dense_memory_limit_gb`)."""
        if self.method is None:
            raise NotImplementedError('EnergyDependentSolver is the base of '
                                      'the dynamic backends')
        nelec = (int(nelec[0]), int(nelec[1]))
        plan = self.planned_sectors(norb, nelec)
        if not plan:
            return delta_roots(None, [], energies, vectors, self.method,
                               self.numerics)
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
                           self.numerics)

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
    """Exact Loewdin: per state, E <- the eigenvalue of H_eff(E) whose vector
    overlaps most with the previous one, until one step moves E by less
    than delta_conv (at most delta_max_iter steps, then DeltaNotConverged).
    Dense: the full spectrum of every sector with one electron less (more)."""
    method = 'lowdin'


class LinearizedLowdinSolver(EnergyDependentSolver):
    """Linearized Loewdin: E0 = E_K + g / (1 - g') from g = <K|H_eff(E_K) -
    H_N|K> and g' = <K|dH_eff(E_K)|K>, then one generalized eigenproblem
    (H_N + D0 - E0 D1) c = E (1 - D1) c, D0 = H_eff(E0) - H_N and
    D1 = dH_eff(E0), and the root with the largest <c|(1 - D1)|K>. Dense,
    as `LowdinSolver`; not iterative."""
    method = 'linearized'


#: the dynamic backends by the name of their method
DYNAMIC_METHODS = {'lowdin': LowdinSolver, 'linearized': LinearizedLowdinSolver}
