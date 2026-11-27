"""The many-body solvers of an active-space Hamiltonian g_eff, as the two
builders of src/EffectiveHamiltonian produce it: solve.py dispatches by name
(`solve`, `SOLVERS`) over the engines of backends.py, the static backends
(FCI, the determinant basis, DMRG, CCSDT/CCSDTQ behind the ActiveSpaceSolver
interface); dynamic_backends.py holds the dynamic backends, which refine the
states of a static backend with the energy-dependent term Delta of a
downfolding on a rotated basis (exact and linearized Loewdin,
`DYNAMIC_METHODS`, on the FCI engine's sector operations) and import
backends.py, never the reverse; determinants.py builds the explicit determinant-basis matrix and the
spin-orbital form of a spatial g_eff. Each entry point is imported from the
module that defines it; this package re-exports nothing.
"""
