import numpy as np
import numba as nb
import scipy as sp
from scipy.optimize import brentq
from tbforge.hamiltonian import ham_numba


class Solver:
    def __init__(self, ham):
        self.ham  = ham


    def get_bands(self, kpts):
        n_kpts = len(kpts)
        n_state = self.ham.nstate()
        bands = np.zeros((n_kpts, n_state), dtype=np.float64)
        for ik in range(n_kpts):
            H = self.ham.build(kpts[ik])
            energies = np.linalg.eigvalsh(H)
            bands[ik,:] = energies
        return bands

    def plot_bands(self, lat, ax=None, **kwargs):
        """Compute and plot the band structure along the default k-path.

        Args:
            lat: Lattice object used to generate the k-path via find_kpts().
            ax: Matplotlib Axes to draw on. Defaults to current axes.
            **kwargs: Forwarded to lat.find_kpts() (e.g. n_kpts, kpath_labels).

        Returns:
            bands: (n_kpts, n_state) array of eigenvalues.
        """
        from .plotter import Plotter
        kpath, kpath_1d, ticks = lat.find_kpts(**kwargs)
        bands = self.get_bands(kpath)
        Plotter(ax=ax).plot_bands(kpath_1d, bands, ticks)
        return bands
    

    def get_berry_curvature(self, kgrid):
        omega = berry_curvature_numba(self.ham.nstate(),
                                     self.ham._terms,
                                     self.ham._terms_phi,
                                     self.ham.params,
                                     ham_numba,
                                     kgrid)
        area_bz = self.ham.bz_area
        prefac = (1/(2*np.pi))* (area_bz/kgrid.shape[0])
        return omega, prefac* np.sum(omega[:,2])

    

    def sweep(self, param_list):
        return self._sweep_numba(self.ham.nstate, self.ham.get_terms(),  param_list)


    @staticmethod
    @nb.njit
    def _sweep_numba(nstate, terms, lam, ham_numba, param_list):
        n_lam = len(param_list)
        arr = np.zeros((n_lam * nstate, 2))
        c = 0
        for i in range(n_lam):
            lam = param_list[i]
            H = ham_numba(nstate, terms, lam)   # Numba-friendly call
            e_vals = np.linalg.eigvalsh(H)
            for j in range(nstate):
                arr[c, 0] = lam
                arr[c, 1] = e_vals[j]
                c += 1
        return arr




    
    @staticmethod
    def evals(M):
        return np.linalg.eigvalsh(M)
    
    @staticmethod
    def evecs(M):
        return np.linalg.eigh(M) 
    

    def get_spin_proj_bands(self, kpts):
        if self.ham.nspin != 2:
            raise ValueError("This term requires nspin=2")
        
        nkpts = len(kpts)
        nstate = self.ham.nstate()
        bands = np.zeros((nkpts, nstate), dtype=np.float64)
        w = np.zeros((nkpts, nstate), dtype=np.float64)
        sz_diag = np.zeros(nstate, dtype=np.float64)
        for ii in range(self.ham.nsite):
            for orb in range(self.ham.norb):
                i_el_up = self.ham._index(ii, orb, 0, 0)
                i_el_dn = self.ham._index(ii, orb, 1, 0)
                sz_diag[i_el_up] = 0.5
                sz_diag[i_el_dn] = -0.5

        for ik, k in enumerate(kpts):
            H = self.ham.build(k)
            evals, evecs = np.linalg.eigh(H)
            bands[ik, :] = evals
            # <psi|Sz|psi> = sum_j |psi_j|^2 * sz_diag[j]
            w[ik, :] = np.sum(np.abs(evecs)**2 * sz_diag[None, :], axis=1)
        return bands, w
    

    def get_orb_proj_bands(self, kpts):
        nkpts = len(kpts)
        nstate = self.ham.nstate()
        norb = self.ham.norb
        nsite = self.ham.nsite

        bands = np.zeros((nkpts, nstate), dtype=np.float64)
        w_orb = np.zeros((nkpts, nstate, norb), dtype=np.float64)

        # Precompute orbital indices for all sites
        orb_diag = [[] for _ in range(norb)]
        for ii in range(nsite):
            for orb in range(norb):
                idxs = []
                for spin in range(self.ham.nspin):
                    for ph in range(self.ham.nph):
                        idxs.append(self.ham._index(ii, orb, spin, ph))
                orb_diag[orb].extend(idxs)

        for ik, k in enumerate(kpts):
            H = self.ham.build(k)
            evals, evecs = np.linalg.eigh(H)
            bands[ik, :] = evals

            # Project eigenvectors onto orbitals
            for orb in range(norb):
                idxs = orb_diag[orb]
                # sum of |c_j|^2 over all states corresponding to this orbital
                w_orb[ik, :, orb] = np.sum(np.abs(evecs[idxs, :])**2, axis=0)
        return bands, w_orb
    

    def get_particle_proj_bands(self, kpts):
        nkpts = len(kpts)
        nstate = self.ham.nstate()
        bands = np.zeros((nkpts, nstate), dtype=np.float64)
        w_el = np.zeros((nkpts, nstate), dtype=np.float64)
        w_hl = np.zeros((nkpts, nstate), dtype=np.float64)

        # identify electron and hole indices
        el_indices = []
        hl_indices = []
        for ii in range(self.ham.nsite):
            for spin in range(self.ham.nspin):
                for orb in range(self.ham.norb):
                    el_indices.append(self.ham._index(ii, orb, spin, 0)) 
                    hl_indices.append(self.ham._index(ii, orb, spin, 1))  

        el_indices = np.array(el_indices, dtype=int)
        hl_indices = np.array(hl_indices, dtype=int)

        for ik, k in enumerate(kpts):
            H = self.ham.build(k)
            evals, evecs = np.linalg.eigh(H)
            bands[ik, :] = evals
            # # electron weight
            w_el[ik, :] = np.sum(np.abs(evecs[el_indices, :])**2, axis=0)
            # # hole weight
            # w_hl[ik, :] = np.sum(np.abs(evecs[hl_indices, :])**2, axis=0)
        return bands, w_el



    def unfold_bands(self, kpts_prim, prim_lat_vecs, prim_basis):
        """Compute unfolded spectral weights of supercell bands onto the primitive BZ.

        Implements the Fourier-transform unfolding method. For each k-point in
        *kpts_prim* and each supercell eigenstate |ψ_n⟩ the spectral weight is

            P_n(k) = (1/N_fold) Σ_{α,μ} |Σ_{j∈α} exp(-i k·R_j) c^n_{j,μ}|²

        where α labels primitive-cell sublattices, μ = (orb, spin, ph),
        R_j = r_j − τ_α is the primitive lattice vector for supercell site j,
        and c^n_{j,μ} is the eigenvector coefficient. The column sum of weights
        equals ``n_pc * norb * nspin * nph`` exactly.

        Args:
            kpts_prim (array_like): (N_k, 3) k-points in Cartesian coordinates
                lying in the primitive Brillouin zone.
            prim_lat_vecs (array_like): (3, 3) primitive lattice vectors, one
                vector per row.
            prim_basis (array_like): (n_pc, 3) Cartesian positions of the
                primitive-cell basis atoms.

        Returns:
            kpts (ndarray): (N_k, 3) copy of *kpts_prim*.
            bands (ndarray): (N_k, N_states) supercell eigenvalues.
            weights (ndarray): (N_k, N_states) spectral weights in [0, 1].

        Raises:
            ValueError: If N_sc is not divisible by n_pc, or any supercell site
                cannot be assigned to a primitive-cell sublattice.
        """
        kpts_prim  = np.asarray(kpts_prim,    dtype=np.float64)
        prim_lv    = np.asarray(prim_lat_vecs, dtype=np.float64)
        prim_basis = np.asarray(prim_basis,    dtype=np.float64)

        sc_pos   = self.ham.rlist
        N_sc     = len(sc_pos)
        n_pc     = len(prim_basis)
        norb     = self.ham.norb
        nspin    = self.ham.nspin
        nph      = self.ham.nph
        N_states = self.ham.nstate()
        n_kpts   = len(kpts_prim)
        n_dof    = norb * nspin * nph

        if N_sc % n_pc != 0:
            raise ValueError(
                f"N_sc ({N_sc}) is not divisible by n_pc ({n_pc}); "
                "supercell must contain an integer number of primitive cells."
            )
        N_fold = N_sc // n_pc

        # Map each supercell site to a primitive sublattice; record R_j = r_j - τ_α
        prim_lv_inv = np.linalg.inv(prim_lv)
        sublattice  = np.empty(N_sc, dtype=int)
        R_offsets   = np.empty((N_sc, 3), dtype=np.float64)

        for i, r_i in enumerate(sc_pos):
            assigned = False
            for alpha, tau in enumerate(prim_basis):
                delta = r_i - tau
                frac  = delta @ prim_lv_inv
                if np.allclose(frac, np.round(frac), atol=1e-4):
                    sublattice[i] = alpha
                    R_offsets[i]  = delta
                    assigned = True
                    break
            if not assigned:
                raise ValueError(
                    f"Supercell site {i} at {r_i} could not be mapped to any "
                    "primitive-cell sublattice."
                )

        # Pre-build idx_mat and R_mat (computed once, outside the k-loop).
        # Group index g = alpha * n_dof + dof_flat, dof_flat iterates (orb, spin, ph).
        # idx_mat[g, l]  = Hamiltonian state index for group g, fold l
        # R_mat[g, l, :] = lattice vector R_j for that site
        n_groups = n_pc * n_dof
        idx_mat  = np.empty((n_groups, N_fold), dtype=int)
        R_mat    = np.empty((n_groups, N_fold, 3), dtype=np.float64)

        dof_list = [(o, s, p)
                    for o in range(norb)
                    for s in range(nspin)
                    for p in range(nph)]

        for alpha in range(n_pc):
            sites = np.where(sublattice == alpha)[0]   # (N_fold,)
            R_grp = R_offsets[sites]                    # (N_fold, 3)
            for d, (orb, spin, ph) in enumerate(dof_list):
                g = alpha * n_dof + d
                idx_mat[g] = [self.ham._index(s_i, orb, spin, ph) for s_i in sites]
                R_mat[g]   = R_grp

        bands   = np.zeros((n_kpts, N_states), dtype=np.float64)
        weights = np.zeros((n_kpts, N_states), dtype=np.float64)

        for ik, k in enumerate(kpts_prim):
            H = self.ham.build(k)
            evals, evecs = np.linalg.eigh(H)
            bands[ik] = evals

            # phases[g, l] = exp(-i k · R_{g,l})           shape: (n_groups, N_fold)
            phases = np.exp(-1j * (R_mat @ k))
            # C[g, l, n]   = evecs[idx_mat[g,l], n]         shape: (n_groups, N_fold, N_states)
            C = evecs[idx_mat]
            # F[g, n]      = Σ_l phases[g,l] · C[g,l,n]    shape: (n_groups, N_states)
            F = np.einsum('gl,gln->gn', phases, C)
            # P_n(k) = (1/N_fold) Σ_g |F[g,n]|²
            weights[ik] = np.sum(np.abs(F) ** 2, axis=0) / N_fold

        return kpts_prim, bands, weights


    def plot_dos(self, erange, kpts=None, eps=1e-2, ax=None, **kwargs):
        """Compute and plot the density of states.

        Args:
            erange: 1-D array of energy values at which the DOS is evaluated.
            kpts: (n_kpts, 3) k-point array. If None, uses the Gamma point
                (suitable for finite systems).
            eps: Lorentzian broadening parameter.
            ax: Matplotlib Axes to draw on. Defaults to current axes.
            **kwargs: Forwarded to Plotter().plot_dos() (e.g. ylim).

        Returns:
            dos: (nE, 2) array — column 0 is energy, column 1 is DOS.
        """
        from .plotter import Plotter
        dos = self.get_dos(erange, kpts=kpts, eps=eps)
        Plotter(ax=ax).plot_dos(dos, **kwargs)
        return dos

    def get_dos(self, erange, kpts=None, eps=1e-2):
        if kpts is None:
            return get_dos_finite_numba(self.ham.nstate(),
                                        ham_numba,
                                        self.ham._terms,
                                        self.ham._terms_phi,
                                        self.ham.params,
                                        erange, eps)
        else:
            return get_dos_numba(self.ham.nstate(),
                                 ham_numba,
                                 self.ham._terms,
                                 self.ham._terms_phi,
                                 self.ham.params,
                                 kpts, erange, eps)
        


#############################################
        
# =============================================================================
# Mean-field self-consistent solver
# =============================================================================

class SCFResult:
    """Container returned by MFSolver.solve()."""

    def __init__(self, fields, mu, history, converged, mfham, kgrid, T):
        self.fields    = fields       # (nsite, 2) converged density
        self.mu        = mu           # chemical potential
        self.history   = history      # [(iter, diff), ...]
        self.converged = converged    # bool
        self._mfham    = mfham
        self._kgrid    = kgrid
        self._T        = T

    @property
    def magnetization(self):
        """Local magnetic moment per site: m[i] = <n_{i,↑}> − <n_{i,↓}>."""
        return self.fields[:, 0] - self.fields[:, 1]

    @property
    def staggered_m(self):
        """Staggered magnetization (scalar): (1/N) Σ_i (−1)^i m_i."""
        m    = self.magnetization
        sign = np.array([(-1)**i for i in range(len(m))], dtype=float)
        return float(np.mean(m * sign))

    @property
    def gap(self):
        """Minimum single-particle gap: min(empty) − max(filled) over the k-grid."""
        nstate = self._mfham.ham.nstate()
        Nk     = len(self._kgrid)
        evals  = np.empty((Nk, nstate))
        for ki, k in enumerate(self._kgrid):
            evals[ki] = np.linalg.eigvalsh(self._mfham.build(k, self.fields))
        occ   = _fermi(evals, self.mu, self._T)
        filled = evals[occ > 0.5]
        empty  = evals[occ < 0.5]
        if len(filled) == 0 or len(empty) == 0:
            return 0.0
        return float(empty.min() - filled.max())

    def get_bands(self, kpts):
        """Compute band structure with the converged fields."""
        nstate = self._mfham.ham.nstate()
        bands  = np.zeros((len(kpts), nstate))
        for ik, k in enumerate(kpts):
            bands[ik] = np.linalg.eigvalsh(self._mfham.build(k, self.fields))
        return bands

    def total_energy(self):
        """Ground-state energy per unit cell (kinetic + double-counting correction).

        E = (1/Nk) Σ_{k,n} E_n(k) f(E_n(k)−μ)  −  U Σ_i <n_{i,↑}><n_{i,↓}>
        """
        Nk     = len(self._kgrid)
        nstate = self._mfham.ham.nstate()
        E_kin  = 0.0
        for k in self._kgrid:
            e   = np.linalg.eigvalsh(self._mfham.build(k, self.fields))
            occ = _fermi(e, self.mu, self._T)
            E_kin += float(np.sum(occ * e))
        E_kin /= Nk
        E_dc = self._mfham.U * float(np.sum(self.fields[:, 0] * self.fields[:, 1]))
        return E_kin - E_dc

    def __repr__(self):
        status = "converged" if self.converged else "NOT converged"
        return (f"SCFResult({status}, niter={len(self.history)}, "
                f"mu={self.mu:.4f}, m_stag={self.staggered_m:.4f}, gap={self.gap:.4f})")


class MFSolver:
    """Self-consistent Hubbard mean-field solver.

    Solves    <n_{i,σ}> = (1/Nk) Σ_{k,n} f(E_n(k)−μ) |<i,σ|ψ_n(k)>|²
    self-consistently with the Hubbard Hartree Hamiltonian.

    Parameters
    ----------
    mfham : MeanFieldHamiltonian

    Example (graphene at U=3t, half-filling)
    -----------------------------------------
    lat   = Lattice.honeycomb()
    hop   = Hopping(lat)
    ham   = Hamiltonian(lat, hop, nspin=2)
    ham.add_nnhops(P.t)
    ham.finalize()
    ham.set_params(t=1.0)

    mfham  = MeanFieldHamiltonian(ham, U=3.0)
    solver = MFSolver(mfham)
    kgrid  = lat.find_kgrid([30, 30, 1])
    result = solver.solve(kgrid, n_elec=2, T=0.02)
    print(result)              # staggered_m > 0 → AFM insulator
    """

    def __init__(self, mfham):
        self.mfham = mfham

    def solve(self, kgrid, n_elec, T=1e-2, mix=0.5,
              max_iter=300, tol=1e-8, init_fields=None):
        """Run the SCF loop.

        Parameters
        ----------
        kgrid       : (Nk, 3) array of k-points covering the full BZ
        n_elec      : float   electrons per unit cell (e.g. 2 for half-filled graphene)
        T           : float   electronic temperature for Fermi smearing (units of t)
        mix         : float   linear mixing fraction 0 < mix ≤ 1
        max_iter    : int     maximum iterations
        tol         : float   convergence threshold on max |Δ<n>|
        init_fields : (nsite, 2) optional initial density; default: staggered AFM guess

        Returns
        -------
        SCFResult
        """
        mfham    = self.mfham
        nsite    = mfham.nsite
        Nk       = len(kgrid)
        nstate   = mfham.ham.nstate()
        n_target = float(n_elec) * Nk

        fields = (_afm_init(nsite) if init_fields is None
                  else np.array(init_fields, dtype=np.float64))

        mu      = 0.0
        history = []

        for it in range(max_iter):
            # --- Diagonalise at every k-point ---
            evals_all = np.empty((Nk, nstate), dtype=np.float64)
            evecs_all = np.empty((Nk, nstate, nstate), dtype=np.complex128)
            for ki in range(Nk):
                H = mfham.build(kgrid[ki], fields)
                evals_all[ki], evecs_all[ki] = np.linalg.eigh(H)

            # --- Chemical potential via bisection ---
            flat = evals_all.ravel()
            mu   = _find_mu(flat, n_target, T)

            # --- New density fields ---
            occ        = _fermi(evals_all, mu, T)           # (Nk, nstate)
            new_fields = _compute_density(evecs_all, occ,
                                          mfham.idx_up, mfham.idx_dn,
                                          nsite, Nk)

            diff = float(np.max(np.abs(new_fields - fields)))
            history.append((it, diff))

            # --- Linear mixing ---
            fields = mix * new_fields + (1.0 - mix) * fields

            if diff < tol:
                return SCFResult(fields, mu, history,
                                 converged=True, mfham=mfham, kgrid=kgrid, T=T)

        return SCFResult(fields, mu, history,
                         converged=False, mfham=mfham, kgrid=kgrid, T=T)


# ---------------------------------------------------------------------------
# MFSolver helpers (pure numpy, no numba — called from Python loop)
# ---------------------------------------------------------------------------

def _fermi(E, mu, T):
    """Numerically stable Fermi-Dirac function."""
    x = np.clip((E - mu) / T, -300.0, 300.0)
    return 1.0 / (np.exp(x) + 1.0)


def _find_mu(evals_flat, n_target, T):
    """Find μ such that Σ_n f(E_n − μ) = n_target (bisection)."""
    emin = evals_flat.min()
    emax = evals_flat.max()
    margin = 20.0 * T

    def nelec(mu_):
        return float(np.sum(_fermi(evals_flat, mu_, T))) - n_target

    lo, hi = emin - margin, emax + margin
    # Guard: ensure the root is bracketed
    if nelec(lo) * nelec(hi) > 0:
        # All states filled or all empty — clamp
        return lo if nelec(lo) > 0 else hi
    return float(brentq(nelec, lo, hi, xtol=1e-12))


def _compute_density(evecs_all, occ, idx_up, idx_dn, nsite, Nk):
    """Compute <n_{i,σ}> from eigenvectors and Fermi weights.

    evecs_all : (Nk, nstate, nstate) — columns are eigenvectors
    occ       : (Nk, nstate)         — Fermi-Dirac weights
    Returns   : (nsite, 2) fields
    """
    fields = np.zeros((nsite, 2), dtype=np.float64)
    for s, idx in enumerate((idx_up, idx_dn)):
        # psi2[k, i, n] = |<i,σ|ψ_n(k)>|²
        psi2 = np.abs(evecs_all[:, idx, :])**2    # (Nk, nsite, nstate)
        fields[:, s] = np.einsum('kin,kn->i', psi2, occ) / Nk
    return fields


def _afm_init(nsite, m0=0.1):
    """Staggered initial density to seed antiferromagnetic order."""
    fields = np.full((nsite, 2), 0.5)
    for i in range(nsite):
        fields[i, 0] += m0 * (-1)**i    # spin-up: higher on even sites
        fields[i, 1] -= m0 * (-1)**i    # spin-dn: lower on even sites
    return np.clip(fields, 0.0, 1.0)


#############################################

@nb.njit
def berry_curvature_numba(nstate, terms, terms_phi, lam, ham, kgrid, delta=1e-5, gap_tol=1e-8):
    nk = kgrid.shape[0]
    Omega = np.zeros((nk, 3), dtype=np.float64)
    dkx = np.array([delta, 0.0, 0.0])
    dky = np.array([0.0, delta, 0.0])

    for ik in range(nk):
        k = kgrid[ik]
        H = ham(nstate, terms, terms_phi, lam, k)
        e, v = np.linalg.eigh(H)
        dHdkx = (ham(nstate, terms, terms_phi, lam, k + dkx) - ham(nstate, terms, terms_phi, lam, k - dkx)) / (2 * delta)
        dHdky = (ham(nstate, terms, terms_phi, lam, k + dky) - ham(nstate, terms, terms_phi, lam, k - dky)) / (2 * delta)
        
        omega_z = 0.0
        # Use sgn for occupation: occupied if e[n] < 0
        for n in range(nstate):
            if e[n] >= 0.0:  # skip unoccupied
                continue
            for m in range(nstate):
                if m == n:
                    continue
                gap = e[m] - e[n]
                if np.abs(gap) < gap_tol:
                    continue
                # Use vdot instead of full V† dH V
                num = np.vdot(v[:, n], dHdkx @ v[:, m]) * np.vdot(v[:, m], dHdky @ v[:, n])
                omega_z += -2.0 * np.imag(num / (gap * gap))

        Omega[ik, 0] = k[0]
        Omega[ik, 1] = k[1]
        Omega[ik, 2] = omega_z
    return Omega




@nb.njit
def delta_func(x, eps):
    return (1.0 / np.pi) * (eps / (x**2 + eps**2))

@nb.njit
def get_dos_numba(nstate, ham, terms, terms_phi, lam, kpts, erange, eps=1e-2):
    nkpts = len(kpts)
    nE = len(erange)
    energies = np.zeros((nkpts, nstate), dtype=np.float64)
    for ik in range(nkpts):
        H = ham(nstate, terms, terms_phi, lam, kpts[ik])
        energies[ik, :] = np.linalg.eigvalsh(H)

    dos = np.zeros((nE, 2), dtype=np.float64)
    for ie in range(nE):
        e = erange[ie]
        s = 0.0
        for ik in range(nkpts):
            for ib in range(nstate):
                x = e - energies[ik, ib]
                s += (1.0/np.pi) * (eps / (x**2 + eps**2))
        dos[ie, 0] = e
        dos[ie, 1] = s/nkpts
    return dos


@nb.njit
def get_dos_finite_numba(nstate, ham, terms, terms_phi, lam, erange, eps=1e-2):
    nE = len(erange)
    H = ham(nstate, terms, terms_phi, lam, np.zeros(3))
    energies = np.linalg.eigvalsh(H)
    dos = np.zeros((nE, 2), dtype=np.float64)
    
    for ie in range(nE):
        e = erange[ie]
        s = 0.0
        for ib in range(nstate):
            x = e - energies[ib]
            s += (eps / (x*x + eps*eps)) / np.pi
        dos[ie, 0] = e
        dos[ie, 1] = s
    
    # normalize DOS to nstate
    dE = erange[1] - erange[0]
    norm = np.sum(dos[:,1]) * dE
    for ie in range(nE):
        dos[ie,1] = dos[ie,1] / norm * nstate
    
    return dos

