import numpy as np
import scipy as sp
import numba as nb
from typing import List, Tuple


class Hamiltonian:
    def __init__(self, 
                 sys, 
                 hopping,
                 params=None,
                 norb=1, nspin=1, nph=1):
        self.sys = sys
        self.hopping = hopping
        self.params =params
        self.norb = norb
        self.nspin = nspin
        self.nph = nph 
        self.hops_nn = self.hopping.get_hops_nn()
        self.terms = []
        if self.sys.is_bulk() is True:
            self.rlist = self.sys.basisVecs
        else:
            self.rlist = self.sys.positions
        self.nsite = len(self.rlist)
        try:
            self.bz_area = self.sys.bz_area()
        except:
            None

    def reset_terms(self):
        self.terms = []
        return self.terms
        
    def nstate(self):
        return self.nsite* self.norb* self.nspin* self.nph
    

    def _index(self, site, orb, spin, ph):
        return site + \
                orb*(self.nsite) + \
                spin*(self.nsite*self.norb) + \
                ph*(self.nsite*self.norb*self.nspin)


    def add_nnhops(self, pid):
        for c in range(len(self.hops_nn)):
            ii = int(self.hops_nn[c,3])
            jj = int(self.hops_nn[c,4])
            rij = self.hops_nn[c,6:9]
            for orb in range(self.norb):
                for spin in range(self.nspin):
                    for ph in range(self.nph):
                        i = self._index(ii,orb,spin,ph)
                        j = self._index(jj,orb,spin,ph)
                        coeff = (-1)**ph
                        self.terms.append([i, j, *rij, pid, coeff])


    def add_2nnhops(self, pid):
        self.hops_2nn = self.hopping.get_hops_2nn()

        for c in range(len(self.hops_2nn)):
            ii = int(self.hops_2nn[c,3])
            jj = int(self.hops_2nn[c,4])
            rij = self.hops_nn[c,6:9]
            for orb in range(self.norb):
                for spin in range(self.nspin):
                    for ph in range(self.nph):
                        i = self._index(ii,orb,spin,ph)
                        j = self._index(jj,orb,spin,ph)
                        coeff = (-1)**ph
                        self.terms.append([i, j, *rij, pid, coeff])
    
    def add_hopsz(self, pid, rz):
        self.hopsz = self.hopping.get_hopsz(rz)
        for c in range(len(self.hopsz)):
            for orb in range(self.norb):
                for spin in range(self.nspin):
                    for ph in range(self.nph):
                        ii = int(self.hopsz[c,3])
                        jj = int(self.hopsz[c,4])
                        i = self._index(ii,orb,spin,ph)
                        j = self._index(jj,orb,spin,ph)
                        rij = self.hopsz[c,6:9]
                        sign = (-1)**ph
                        self.terms.append([i, j, rij[0], rij[1], rij[2], pid, sign])



    def add_peierls(self, pid):
        if self.nspin == 2 or self.nph == 2:
            raise ValueError("This term requires nspin=1 and nph=1")
        for c in range(len(self.hops_nn)):
            ii = int(self.hops_nn[c, 3])
            jj = int(self.hops_nn[c, 4])
            rij = self.hops_nn[c,6:9]
            ri = self.rlist[ii]
            rj = self.rlist[jj]
            x_mean = 0.5* (ri[0]+rj[0])
            dy     =  rj[1]-ri[1]
            for orb in range(self.norb):    
                for spin in range(self.nspin):
                    for ph in range(self.nph):
                        i = self._index(ii, orb, spin, ph)
                        j = self._index(jj, orb, spin, ph)
                        coeff = 2*np.pi* x_mean* dy
                        self.terms.append([i, j, *rij, pid, coeff])


    def add_mu(self, pid):
        for ii in range(self.nsite):
            for orb in range(self.norb):
                for spin in range(self.nspin):
                    for ph in range(self.nph):
                        i = self._index(ii,orb,spin,ph)
                        sign = (-1)**ph
                        self.terms.append([i, i, 0, 0, 0, pid, sign])

    def add_bilayer_polarized_pot(self, pid):
        for ii in range(self.nsite):
            if self.rlist[ii,2]==0.0:
                coeff = -1
            else:
                coeff = 1
            for orb in range(self.norb):
                for spin in range(self.nspin):
                    for ph in range(self.nph):
                        i = self._index(ii,orb,spin,ph)
                        self.terms.append([i, i, 0, 0, 0, pid, coeff*(-1)**ph])


    def add_quasiperiodic_pot(self, pid_x, pid_y):
        beta = (np.sqrt(5)+1)* 0.5
        for ii in range(self.nsite):
            m,n,_ = self.sys.site_to_cell(ii)
            for orb in range(self.norb):
                for spin in range(self.nspin):
                    for ph in range(self.nph):
                        i = self._index(ii,orb,spin,ph)
                        coeff = np.cos(2*np.pi* beta* m)* (-1)**ph
                        self.terms.append([i, i, 0, 0, 0, pid_x, coeff])
                        coeff = np.cos(2*np.pi* beta* n)* (-1)**ph
                        self.terms.append([i, i, 0, 0, 0, pid_y, coeff])


    def add_impurity(self, pid):
        o=self.sys.get_mid_site()
        for orb in range(self.norb):
            for spin in range(self.nspin):
                for ph in range(self.nph):
                    i = self._index(o,orb,spin,ph)
                    self.terms.append([i, i, 0, 0, 0, pid, (-1)**ph*(-1)**spin])


    def add_zeeman(self, pid):
        if self.nspin != 2:
            raise ValueError("This term requires nspin=2")
        
        for ii in range(self.nsite):
            for orb in range(self.norb):
                for spin in range(self.nspin):
                    for ph in range(self.nph):
                        i = self._index(ii,orb,spin,ph)
                        sign = (-1)**ph* (-1)**spin
                        self.terms.append([i, i, 0, 0, 0, pid, sign])


    def add_zeeman_in_plane(self, pid_x=None, pid_y=None):
        if self.nspin != 2:
            raise ValueError("In-plane Zeeman requires nspin=2")

        for ii in range(self.nsite):
            for orb in range(self.norb):
                for ph in range(self.nph):
                    i_up = self._index(ii, orb, 0, ph)
                    i_dn = self._index(ii, orb, 1, ph)
                    sign = (-1)**ph

                    # sigma_x term
                    if pid_x is not None:
                        self.terms.append([i_up, i_dn, 0, 0, 0, pid_x, sign])
                        self.terms.append([i_dn, i_up, 0, 0, 0, pid_x, sign])

                    # sigma_y term
                    if pid_y is not None:
                        self.terms.append([i_up, i_dn, 0, 0, 0, pid_y, -1j*sign])
                        self.terms.append([i_dn, i_up, 0, 0, 0, pid_y, 1j*sign])
    

    def add_rashba(self, pid):
        if self.nspin != 2:
            raise ValueError("This term requires nspin=2")
        
        for c in range(len(self.hops_nn)):
            ii = int(self.hops_nn[c,3])
            jj = int(self.hops_nn[c,4])
            rij = self.hops_nn[c,6:9]
            dx, dy, dz = rij
            for orb in range(self.norb):
                for ph in range(self.nph):
                    i_up = self._index(ii, orb, 0, ph)
                    i_dn = self._index(ii, orb, 1, ph)
                    j_up = self._index(jj, orb, 0, ph)
                    j_dn = self._index(jj, orb, 1, ph)
                    if ph==0:
                        coeff_up_down = 1j*(dy+1j*dx)
                        coeff_down_up = 1j*(dy-1j*dx)
                    else:
                        coeff_up_down = -np.conj(1j*(dy+1j*dx))
                        coeff_down_up = -np.conj(1j*(dy-1j*dx))
                    #up_down terms
                    self.terms.append([i_up, j_dn, *rij, pid, coeff_up_down])
                    #down_up terms
                    self.terms.append([i_dn, j_up, *rij, pid, coeff_down_up])



    def add_stag_pot(self, pid):
        for ii in range(self.nsite):
            for orb in range(self.norb):
                for spin in range(self.nspin):
                    for ph in range(self.nph):
                        i = self._index(ii,orb,spin,ph)
                        coeff = (-1)**(i%2)* (-1)**ph
                        self.terms.append([i, i, 0, 0, 0, pid, coeff])



    def add_haldane(self, pid):
        if self.nspin == 2:
            raise ValueError("This term requires nspin=1")
        
        #nu=1 or -1 for each NNN bond
        self.hops_2nn = self.hopping.get_hops_2nn()
        nu_list = self.hopping.get_kmsign()
        for c in range(len(self.hops_2nn)):
            ii = int(self.hops_2nn[c,3])
            jj = int(self.hops_2nn[c,4])
            rij = self.hops_2nn[c,6:9]
            nu = nu_list[c]
            for orb in range(self.norb):
                for ph in range(self.nph):
                    i = self._index(ii, orb, 0, ph)
                    j = self._index(jj, orb, 0, ph)
                    #CHECK: dont need (-1)**ph due to 1j 
                    coeff = 1j* nu
                    if pid is not None:
                        self.terms.append([i, j, *rij, pid, coeff])


    def add_kmsoc(self, pid):
        if self.nspin != 2:
            raise ValueError("This term requires nspin=2")
        
        #nu=1 or -1 for each NNN bond
        self.hops_2nn = self.hopping.get_hops_2nn()
        nu_list = self.hopping.get_kmsign()
        for c in range(len(self.hops_2nn)):
            ii = int(self.hops_2nn[c,3])
            jj = int(self.hops_2nn[c,4])
            rij = self.hops_2nn[c,6:9]
            nu = nu_list[c]
            for orb in range(self.norb):
                for spin in range(self.nspin):
                    for ph in range(self.nph):
                        i = self._index(ii, orb, spin, ph)
                        j = self._index(jj, orb, spin, ph)
                        #CHECK: dont need (-1)**ph due to 1j 
                        coeff = 1j*(-1)**spin* nu
                        if pid is not None:
                            self.terms.append([i, j, *rij, pid, coeff])


    def add_s_wave(self, pid):
        if self.nspin != 2 or self.nph != 2:
            raise ValueError("This term requires nspin=2 and nph=2")

        for ii in range(self.nsite):
            for orb in range(self.norb):
                i_el_up = self._index(ii, orb, 0, 0)   # el_up
                i_hl_dn = self._index(ii, orb, 1, 1)   # hl_dn
                i_el_dn = self._index(ii, orb, 1, 0)   # el_dn
                i_hl_up = self._index(ii, orb, 0, 1)   # hl_up

                # el_up <-> hl_dn
                self.terms.append([i_el_up, i_hl_dn, 0, 0, 0,  pid, 1.0])
                self.terms.append([i_hl_dn, i_el_up, 0, 0, 0,  np.conj(pid), 1.0])

                # el_dn <-> hl_up
                self.terms.append([i_el_dn, i_hl_up, 0, 0, 0,  pid, 1.0])
                self.terms.append([i_hl_up, i_el_dn, 0, 0, 0,  np.conj(pid), 1.0])

    def add_pip_pairing(self, pid):
        if self.nph != 2:
            raise ValueError("Requires nph=2")

        for c in range(len(self.hops_nn)):
            ii = int(self.hops_nn[c,3])
            jj = int(self.hops_nn[c,4])
            rij = self.hops_nn[c,6:9]
            theta = np.arctan2(rij[1], rij[0])
            phase = np.exp(1j * theta)

            for orb in range(self.norb):
                # up-up pairing
                i_el_up = self._index(ii, orb, 0, 0)
                j_hl_up = self._index(jj, orb, 0, 1)
                self.terms.append([i_el_up, j_hl_up, *rij, pid, phase])
                j_el_up = self._index(jj, orb, 0, 0)
                i_hl_up = self._index(ii, orb, 0, 1)
                self.terms.append([j_el_up, i_hl_up, *rij, pid, -phase])

                # down-down pairing
                i_el_dn = self._index(ii, orb, 1, 0)
                j_hl_dn = self._index(jj, orb, 1, 1)
                self.terms.append([i_el_dn, j_hl_dn, *rij, pid, phase])
                j_el_dn = self._index(jj, orb, 1, 0)
                i_hl_dn = self._index(ii, orb, 1, 1)
                self.terms.append([j_el_dn, i_hl_dn, *rij, pid, -phase])
                

    def add_multiorb_sk(self, amp_matrices, pid_matrix=None, SK_params=None):
        """
        Add multi-orbital Slater-Koster hoppings and onsite energies using PIDs.

        amp_matrices: list of (norb x norb) hopping amplitudes per NN pair
        pid_matrix: list of (norb x norb) parameter IDs corresponding to amp_matrices
        SK_params: dict with 'onsite': {orbital_name: pid} or None
        """
        for c, amp_matrix in enumerate(amp_matrices):
            ii = int(self.hops_nn[c, 3])
            jj = int(self.hops_nn[c, 4])
            rij = self.hops_nn[c, 6:9]

            for orb_i in range(self.norb):
                for orb_j in range(self.norb):
                    if pid_matrix is None:
                        # fallback: treat amp_matrix as numerical, store as dummy PID
                        pid = None
                    else:
                        pid = pid_matrix[c][orb_i, orb_j]

                    # Skip zero PIDs
                    if pid is None:
                        continue

                    for spin in range(self.nspin):
                        for ph in range(self.nph):
                            i = self._index(ii, orb_i, spin, ph)
                            j = self._index(jj, orb_j, spin, ph)
                            sign = (-1)**ph
                            # Rij still stored for Bloch phase
                            self.terms.append([i, j, *rij, pid, sign])

        # Onsite energies
        if SK_params is None or 'onsite' not in SK_params:
            return

        onsite_dict = SK_params['onsite']
        # consistent orbital order
        onsite_orbs = ('s', 'px', 'py', 'pz')
        onsite_pids = np.array([onsite_dict[o] for o in onsite_orbs], dtype=int)

        for site in range(self.nsite):
            for orb in range(self.norb):
                pid = onsite_pids[orb]
                if pid is None:
                    continue
                for spin in range(self.nspin):
                    for ph in range(self.nph):
                        i = self._index(site, orb, spin, ph)
                        sign = (-1)**ph
                        self.terms.append([i, i, 0.0, 0.0, 0.0, pid, sign])
    

    def sk_table(self, orb_type, SK_params=None):
        hopping_matrices = []
        for c in range(len(self.hops_nn)):
            rij = np.array(self.hops_nn[c, 6:9], dtype=np.float64)
            rnorm = np.linalg.norm(rij)
            if rnorm == 0:
                raise ValueError("Bond vector cannot be zero")

            l, m, n = rij / rnorm #direction cosines

            if orb_type == 's':
                amp_matrix = np.array([[SK_params['Vss']]], dtype=np.float64)

            elif orb_type == 'p':
                Vpp_sigma = SK_params['Vpp_sigma']
                Vpp_pi = SK_params['Vpp_pi']

                amp_matrix = np.zeros((3, 3), dtype=np.float64)

                amp_matrix[0,0] = l*l*Vpp_sigma + (1-l*l)*Vpp_pi
                amp_matrix[1,1] = m*m*Vpp_sigma + (1-m*m)*Vpp_pi
                amp_matrix[2,2] = n*n*Vpp_sigma + (1-n*n)*Vpp_pi

                amp_matrix[0,1] = amp_matrix[1,0] = l*m*(Vpp_sigma - Vpp_pi)
                amp_matrix[0,2] = amp_matrix[2,0] = l*n*(Vpp_sigma - Vpp_pi)
                amp_matrix[1,2] = amp_matrix[2,1] = m*n*(Vpp_sigma - Vpp_pi)

            elif orb_type == 's+p':
                Vss = SK_params['Vss']
                Vsp = SK_params['Vsp']
                Vpp_sigma = SK_params['Vpp_sigma']
                Vpp_pi = SK_params['Vpp_pi']

                amp_matrix = np.zeros((4, 4), dtype=np.float64)

                # s–s and s–p
                amp_matrix[0,0] = Vss
                amp_matrix[0,1:4] = [ l*Vsp,  m*Vsp,  n*Vsp ]
                amp_matrix[1:4,0] = [-l*Vsp, -m*Vsp, -n*Vsp]

                # p–p
                amp_matrix[1,1] = l*l*Vpp_sigma + (1-l*l)*Vpp_pi
                amp_matrix[2,2] = m*m*Vpp_sigma + (1-m*m)*Vpp_pi
                amp_matrix[3,3] = n*n*Vpp_sigma + (1-n*n)*Vpp_pi

                amp_matrix[1,2] = amp_matrix[2,1] = l*m*(Vpp_sigma - Vpp_pi)
                amp_matrix[1,3] = amp_matrix[3,1] = l*n*(Vpp_sigma - Vpp_pi)
                amp_matrix[2,3] = amp_matrix[3,2] = m*n*(Vpp_sigma - Vpp_pi)
            else:
                raise ValueError(f"Unsupported orbital type: {orb_type}")
            hopping_matrices.append(amp_matrix)

        return np.asarray(hopping_matrices)
    
    def add_fm(self, pid):
        for ii in range(self.nsite):
            for orb in range(self.norb):
                for ph in range(self.nph):
                    i = self._index(ii,orb,0,ph)
                    coeff =  (-1)**ph
                    self.terms.append([i, i, 0, 0, 0, pid, coeff])
        
    
    def finalize(self):
        self.terms = np.asarray(self.terms, dtype=np.complex128)

    def get_terms(self):
        return self.terms
    
    def build(self, k=np.array([0.,0.,0.])):
        return ham_numba(self.nstate(), self.get_terms(), self.params, k)
    
    def hamf(self, lam):
        return ham_numba(self.nstate(), self.terms, lam=lam, k=np.array([0.,0.,0.]))
    

# @nb.njit
# def ham_numba(nstate, terms, k):
#     H = np.zeros((nstate, nstate), dtype=np.complex128)
#     for c in range(len(terms)):
#         i = int(terms[c, 0].real)
#         j = int(terms[c, 1].real)
#         rx = terms[c, 2].real
#         ry = terms[c, 3].real
#         rz = terms[c, 4].real
#         amp = terms[c, 5]
#         sign = int(terms[c, 6].real)
#         bloch_phase = np.exp(1j * (rx*k[0] + ry*k[1] + rz*k[2]))
#         H[i, j] += -amp * sign * bloch_phase
#     return H

@nb.njit
def ham_numba(nstate, terms0, lam, k=np.array([0.,0.,0.])):
    H = np.zeros((nstate, nstate), dtype=np.complex128)
    mask_peierls = terms0[:,5].real==16
    terms_peierls =  terms0[mask_peierls]
    terms = terms0[~mask_peierls]

    for c in range(terms_peierls.shape[0]):
        i = int(terms_peierls[c, 0].real)
        j = int(terms_peierls[c, 1].real)
        dx = terms_peierls[c, 2].real
        dy = terms_peierls[c, 3].real
        dz = terms_peierls[c, 4].real
        pid = int(terms_peierls[c, 5].real)
        coeff = terms_peierls[c, 6]
        bloch_phase = np.exp(1j * (dx*k[0] + dy*k[1] + dz*k[2]))
        H[i, j] -= np.exp(1j* lam[pid] * coeff) * bloch_phase

    for c in range(terms.shape[0]):
        i = int(terms[c, 0].real)
        j = int(terms[c, 1].real)
        dx = terms[c, 2].real
        dy = terms[c, 3].real
        dz = terms[c, 4].real
        pid = int(terms[c, 5].real)
        coeff = terms[c, 6]
        bloch_phase = np.exp(1j * (dx*k[0] + dy*k[1] + dz*k[2]))
        H[i, j] -= lam[pid] * coeff * bloch_phase

    return H

@nb.njit
def ham_custom(nstate, terms, lam, k=np.array([0.,0.,0.])):
    H = np.zeros((nstate, nstate), dtype=np.complex128)
    lamb=1.0

    for c in range(terms.shape[0]):
        i = int(terms[c, 0].real)
        j = int(terms[c, 1].real)
        dx = terms[c, 2].real
        dy = terms[c, 3].real
        dz = terms[c, 4].real
        pid = int(terms[c, 5].real)
        coeff = terms[c, 6]
        bloch_phase = np.exp(1j * (dx*k[0] + dy*k[1] + dz*k[2]))
        if pid==16:
            H[i, j] -= np.exp(1j* lam[pid] * coeff) * bloch_phase
        elif pid==3:
            dij = np.sqrt(dx**2+dy**2+dz**2)
            exp_factor = np.exp(-(dij-dz)/lamb)
            H[i, j] -= lam[pid] * exp_factor* coeff * bloch_phase
        else:
            H[i, j] -= lam[pid] * coeff * bloch_phase
    return H

