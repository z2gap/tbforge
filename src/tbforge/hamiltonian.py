import numpy as np
import scipy as sp
import numba as nb
from typing import List, Tuple


class Hamiltonian:
    def __init__(self, 
                 norb, nspin, nph,
                 lat, 
                 hops_nn, hops_2nn=None, hopsz=None):
        self.rlist = lat.basisVecs
        self.nsite = len(self.rlist)
        self.norb = norb
        self.nspin = nspin
        self.nph = nph 
        self.hops_nn = hops_nn 
        self.hops_2nn = hops_2nn
        self.hopsz = hopsz 
        self.terms = []

    def nstate(self):
        return self.nsite* self.norb* self.nspin* self.nph

    def get_terms(self):
        return np.array(self.terms, dtype=np.complex128)
    
    def ham(self, terms, k):
        return ham_numba(self.nstate(), terms, k)
    

    def _index(self, site, orb, spin, ph):
        return site + \
                orb*(self.nsite) + \
                spin*(self.nsite*self.norb) + \
                ph*(self.nsite*self.norb*self.nspin)


    def add_nnhops(self, amp):
        for c in range(len(self.hops_nn)):
            for orb in range(self.norb):
                for spin in range(self.nspin):
                    for ph in range(self.nph):
                        ii = int(self.hops_nn[c,3])
                        jj = int(self.hops_nn[c,4])
                        i = self._index(ii,orb,spin,ph)
                        j = self._index(jj,orb,spin,ph)
                        rij = self.hops_nn[c,6:9]
                        self.terms.append([i, j, rij[0], rij[1], rij[2], amp, (-1)**ph])

    def add_2nnhops(self, amp):
        if self.hops_2nn is not None:
            for c in range(len(self.hops_2nn)):
                for orb in range(self.norb):
                    for spin in range(self.nspin):
                        for ph in range(self.nph):
                            ii = int(self.hops_2nn[c,3])
                            jj = int(self.hops_2nn[c,4])
                            i = self._index(ii,orb,spin,ph)
                            j = self._index(jj,orb,spin,ph)
                            rij = self.hops_2nn[c,6:9]
                            self.terms.append([i, j, rij[0], rij[1], rij[2], amp, (-1)**ph])
    
    def add_hopsz(self, amp):
        if self.hopsz is not None and len(self.hopsz) > 0:
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
                            self.terms.append([i, j, rij[0], rij[1], rij[2], amp, sign])


    def add_mu(self, amp):
        for ii in range(self.nsite):
            for orb in range(self.norb):
                for spin in range(self.nspin):
                    for ph in range(self.nph):
                        i = self._index(ii,orb,spin,ph)
                        sign = (-1)**ph
                        self.terms.append([i, i, 0, 0, 0, amp, sign])

    def add_zeeman(self, amp):
        if self.nspin != 2:
            raise ValueError("This term requires nspin=2")
        
        for ii in range(self.nsite):
            for orb in range(self.norb):
                for spin in range(self.nspin):
                    for ph in range(self.nph):
                        i = self._index(ii,orb,spin,ph)
                        sign = (-1)**ph* (-1)**spin
                        self.terms.append([i, i, 0, 0, 0, amp, sign])


    def add_zeeman_in_plane(self, hx=0.0, hy=0.0):
        for ii in range(self.nsite):
            for orb in range(self.norb):
                for ph in range(self.nph):
                    # spin-up / spin-down indices
                    i_up = self._index(ii, orb, 0, ph)
                    i_dn = self._index(ii, orb, 1, ph)
                    ri = self.rlist[ii]
                    # sigma_x term
                    if hx != 0.0:
                        self.terms.append([i_up, i_dn, 0, 0, 0, hx, 1.0])
                        self.terms.append([i_dn, i_up, 0, 0, 0, hx, 1.0])
                    # hy term
                    if hy != 0.0:
                        self.terms.append([i_up, i_dn, 0, 0, 0, hy, -1j])
                        self.terms.append([i_dn, i_up, 0, 0, 0, hy, 1j])
        return self.hops
    
    def add_rashba(self, amp):
        if self.nspin != 2:
            raise ValueError("This term requires nspin=2")
        
        for c in range(len(self.hops_nn)):
            for orb in range(self.norb):
                for ph in range(self.nph):
                    ii = int(self.hops_nn[c,3])
                    jj = int(self.hops_nn[c,4])
                    i_up = self._index(ii, orb, 0, ph)
                    i_dn = self._index(ii, orb, 1, ph)
                    j_up = self._index(jj, orb, 0, ph)
                    j_dn = self._index(jj, orb, 1, ph)
                    ri = self.rlist[ii]
                    rj = self.rlist[jj]
                    rij = ri-rj
                    dx = rij[0]
                    dy = rij[1]
                    sign = (-1)**ph
                    self.terms.append([i_up, j_dn, rij[0], rij[1], rij[2], 1j*amp*dy, sign])
                    self.terms.append([i_dn, j_up, rij[0], rij[1], rij[2], 1j*amp*dy, sign])
                    self.terms.append([i_up, j_dn, rij[0], rij[1], rij[2], amp*dx, sign])
                    self.terms.append([i_dn, j_up, rij[0], rij[1], rij[2], -amp*dx, sign])


    def add_s_wave(self, amp):
        if self.nspin != 2 or self.nph != 2:
            raise ValueError("This term requires nspin=2 and nph=2")

        for ii in range(self.nsite):
            for orb in range(self.norb):
                i_el_up = self._index(ii, orb, 0, 0)   # el_up
                i_hl_dn = self._index(ii, orb, 1, 1)   # hl_dn
                i_el_dn = self._index(ii, orb, 1, 0)   # el_dn
                i_hl_up = self._index(ii, orb, 0, 1)   # hl_up

                # el_up <-> hl_dn
                self.terms.append([i_el_up, i_hl_dn, 0, 0, 0,  amp, 1.0])
                self.terms.append([i_hl_dn, i_el_up, 0, 0, 0,  np.conj(amp), 1.0])

                # el_dn <-> hl_up
                self.terms.append([i_el_dn, i_hl_up, 0, 0, 0,  amp, 1.0])
                self.terms.append([i_hl_up, i_el_dn, 0, 0, 0,  np.conj(amp), 1.0])
                

    def add_multiorb_sk(self, amp_matrices, SK_params=None):
        for c, amp_matrix in enumerate(amp_matrices):
            ii = int(self.hops_nn[c, 3])
            jj = int(self.hops_nn[c, 4])
            rij = self.hops_nn[c, 6:9]

            for orb_i in range(self.norb):
                for orb_j in range(self.norb):
                    amp = amp_matrix[orb_i, orb_j]
                    if abs(amp) < 1e-12:
                        continue
                    for spin in range(self.nspin):
                        for ph in range(self.nph):
                            i = self._index(ii, orb_i, spin, ph)
                            j = self._index(jj, orb_j, spin, ph)
                            self.terms.append([i, j, *rij, amp, (-1)**ph])

        if SK_params is None or 'onsite' not in SK_params:
            return  

        onsite_dict = SK_params['onsite']
        # enforce consistent orbital order: s, px, py, pz
        onsite_vec = np.array([onsite_dict[o] for o in ('s', 'px', 'py', 'pz')], dtype=float)

        for site in range(self.nsite):
            for orb in range(self.norb):
                eps = onsite_vec[orb]
                if abs(eps) < 1e-12:
                    continue
                for spin in range(self.nspin):
                    for ph in range(self.nph):
                        i = self._index(site, orb, spin, ph)
                        self.terms.append([i, i, 0.0, 0.0, 0.0, eps, (-1)**ph])
    

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
    
    

@nb.njit
def ham_numba(nstate, terms, k):
    H = np.zeros((nstate, nstate), dtype=np.complex128)
    for c in range(len(terms)):
        i = int(terms[c, 0].real)
        j = int(terms[c, 1].real)
        rx = terms[c, 2].real
        ry = terms[c, 3].real
        rz = terms[c, 4].real
        amp = terms[c, 5]
        sign = int(terms[c, 6].real)
        bloch_phase = np.exp(1j * (rx*k[0] + ry*k[1] + rz*k[2]))
        H[i, j] += -amp * sign * bloch_phase
    return H

