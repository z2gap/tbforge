import numpy as np
import numba as nb
import scipy as sp
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
    

    def get_berry_curvature(self, kgrid):
        omega = berry_curvature_numba(self.ham.nstate(), 
                                     self.ham.get_terms(), 
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
            H = self.ham.ham(self.ham.get_terms(), k)
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
            H = self.ham.ham(self.ham.get_terms(), k)
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
            H = self.ham.ham(self.ham.get_terms(), k)
            evals, evecs = np.linalg.eigh(H)
            bands[ik, :] = evals
            # # electron weight
            w_el[ik, :] = np.sum(np.abs(evecs[el_indices, :])**2, axis=0)
            # # hole weight
            # w_hl[ik, :] = np.sum(np.abs(evecs[hl_indices, :])**2, axis=0)
        return bands, w_el



    def get_dos(self, erange, kpts=None, eps=1e-2):
        if kpts is None:
            return get_dos_finite_numba(self.ham.nstate(), 
                                        ham_numba, 
                                        self.ham.get_terms(),
                                        erange, eps)
        else:
            return get_dos_numba(self.ham.nstate(), 
                                ham_numba, 
                                self.ham.get_terms(), 
                                kpts, erange, eps)
        


#############################################
        
@nb.njit
def berry_curvature_numba(nstate, terms, lam, ham, kgrid, delta=1e-5, gap_tol=1e-8):
    nk = kgrid.shape[0]
    Omega = np.zeros((nk, 3), dtype=np.float64)
    dkx = np.array([delta, 0.0, 0.0])
    dky = np.array([0.0, delta, 0.0])

    for ik in range(nk):
        k = kgrid[ik]
        H = ham(nstate, terms, lam, k)
        e, v = np.linalg.eigh(H)
        dHdkx = (ham(nstate, terms, lam, k + dkx) - ham(nstate, terms, lam, k - dkx)) / (2 * delta)
        dHdky = (ham(nstate, terms, lam, k + dky) - ham(nstate, terms, lam, k - dky)) / (2 * delta)
        
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
def get_dos_numba(nstate, ham, terms, kpts, erange, eps=1e-2):
    nkpts = len(kpts)
    nE = len(erange)
    energies = np.zeros((nkpts, nstate), dtype=np.float64)
    for ik in range(nkpts):
        H = ham(nstate, terms, kpts[ik])
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
def get_dos_finite_numba(nstate, ham, terms, erange, eps=1e-2):
    nE = len(erange)
    H = ham(nstate, terms, k=[0,0,0])
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

