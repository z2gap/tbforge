import numpy as np
import scipy as sp
import numba as nb
from .params import P


class Hamiltonian:
    def __init__(self,
                 sys=None,
                 hopping=None,
                 params=None,
                 norb=1, nspin=1, nph=1,
                 lat=None, hops_nn=None):
        if sys is None and lat is not None:
            sys = lat
        self.sys = sys
        if hops_nn is not None:
            self.hops_nn = hops_nn
        else:
            if hopping is None:
                from .hopping import Hopping
                hopping = Hopping(sys)
            self.hopping = hopping
            self.hops_nn = self.hopping.get_hops_nn()
        self.norb = norb
        self.nspin = nspin
        self.nph = nph
        self.terms = []
        if self.sys.is_bulk() is True:
            self.rlist = self.sys.basis_vecs
        else:
            self.rlist = self.sys.positions
        self.nsite = len(self.rlist)
        try:
            self.bz_area = self.sys.bz_area()
        except Exception:
            pass

        if params is None:
            self.params = np.zeros(P.n_param, dtype=np.float64)
        else:
            self.params = np.asarray(params, dtype=np.float64)
        self.params[P.const] = 1.0

        # Separated term arrays (populated by finalize())
        self._terms = None
        self._terms_phi = None

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def set_params(self, **kwargs):
        """Update individual parameters by name, e.g. ham.set_params(t=1.0, mu=0.5)."""
        for name, value in kwargs.items():
            if not hasattr(P, name) or name == 'n_param':
                raise ValueError(f"Unknown parameter '{name}'")
            self.params[getattr(P, name)] = value
        return self

    # ------------------------------------------------------------------
    # Index / size helpers
    # ------------------------------------------------------------------

    def reset_terms(self):
        self.terms = []
        self._terms = None
        self._terms_phi = None
        return self.terms

    def nstate(self):
        return self.nsite * self.norb * self.nspin * self.nph

    def _index(self, site, orb, spin, ph):
        return site + \
                orb*(self.nsite) + \
                spin*(self.nsite*self.norb) + \
                ph*(self.nsite*self.norb*self.nspin)

    # ------------------------------------------------------------------
    # Term builders
    # ------------------------------------------------------------------

    def add_nnhops(self, pid=None):
        if pid is None:
            pid = P.t
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
        return self


    def add_2nnhops(self, pid=None):
        if pid is None:
            pid = P.tnnn
        self.hops_2nn = self.hopping.get_hops_2nn()

        for c in range(len(self.hops_2nn)):
            ii = int(self.hops_2nn[c,3])
            jj = int(self.hops_2nn[c,4])
            rij = self.hops_2nn[c,6:9]          # fixed: was hops_nn
            for orb in range(self.norb):
                for spin in range(self.nspin):
                    for ph in range(self.nph):
                        i = self._index(ii,orb,spin,ph)
                        j = self._index(jj,orb,spin,ph)
                        coeff = (-1)**ph
                        self.terms.append([i, j, *rij, pid, coeff])
        return self

    def add_hopsz(self, pid=None, rz=0.):
        if pid is None:
            pid = P.tz
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
        return self


    def add_interlayer(self, pid=None, dz=0., lam_decay=1., r_cut=None):
        """Add interlayer Slater-Koster hoppings for bilayer/moiré systems.

        The hopping amplitude follows an exponential decay:
            t_ij = tz * exp(-(dij - dz) / lam_decay)

        where dij = |ri - rj| is the 3D distance between sites, dz is the
        vertical interlayer separation, and lam_decay is the decay length.
        The amplitude tz is stored as params[pid].

        Args:
            pid: Parameter index for tz (default P.tz).
            dz: Vertical interlayer separation between the two layers.
            lam_decay: Exponential decay length controlling range of hoppings.
            r_cut: 3D distance cutoff (default: dz + 3*lam_decay).
        """
        if pid is None:
            pid = P.tz
        if r_cut is None:
            r_cut = dz + 3 * lam_decay

        hops = self.hopping.get_interlayer_hops(r_cut)
        if len(hops) == 0:
            return self

        for c in range(len(hops)):
            ii = int(hops[c, 3])
            jj = int(hops[c, 4])
            dij = hops[c, 5]
            rij = hops[c, 6:9]
            decay = np.exp(-(dij - dz) / lam_decay)
            for orb in range(self.norb):
                for spin in range(self.nspin):
                    for ph in range(self.nph):
                        i = self._index(ii, orb, spin, ph)
                        j = self._index(jj, orb, spin, ph)
                        coeff = decay * (-1)**ph
                        self.terms.append([i, j, *rij, pid, coeff])
        return self

    def add_peierls(self, pid=None):
        if pid is None:
            pid = P.phi
        if self.nspin == 2 or self.nph == 2:
            raise ValueError("This term requires nspin=1 and nph=1")
        for c in range(len(self.hops_nn)):
            ii = int(self.hops_nn[c, 3])
            jj = int(self.hops_nn[c, 4])
            rij = self.hops_nn[c,6:9]
            ri = self.rlist[ii]
            rj = self.rlist[jj]
            x_mean = 0.5 * (ri[0]+rj[0])
            dy     = rj[1]-ri[1]
            for orb in range(self.norb):
                for spin in range(self.nspin):
                    for ph in range(self.nph):
                        i = self._index(ii, orb, spin, ph)
                        j = self._index(jj, orb, spin, ph)
                        coeff = 2*np.pi * x_mean * dy
                        self.terms.append([i, j, *rij, pid, coeff])
        return self


    def add_mu(self, pid=None):
        if pid is None:
            pid = P.mu
        for ii in range(self.nsite):
            for orb in range(self.norb):
                for spin in range(self.nspin):
                    for ph in range(self.nph):
                        i = self._index(ii,orb,spin,ph)
                        sign = (-1)**ph
                        self.terms.append([i, i, 0, 0, 0, pid, sign])
        return self

    def add_bilayer_polarized_pot(self, pid=None):
        if pid is None:
            pid = P.Vbilayer
        for ii in range(self.nsite):
            coeff = -1 if np.isclose(self.rlist[ii, 2], 0.0) else 1
            for orb in range(self.norb):
                for spin in range(self.nspin):
                    for ph in range(self.nph):
                        i = self._index(ii,orb,spin,ph)
                        self.terms.append([i, i, 0, 0, 0, pid, coeff*(-1)**ph])
        return self


    def add_quasiperiodic_pot(self, pid_x=None, pid_y=None):
        if pid_x is None:
            pid_x = P.Vx_aah
        if pid_y is None:
            pid_y = P.Vy_aah
        beta = (np.sqrt(5)+1) * 0.5
        for ii in range(self.nsite):
            m,n,_ = self.sys.site_to_cell(ii)
            for orb in range(self.norb):
                for spin in range(self.nspin):
                    for ph in range(self.nph):
                        i = self._index(ii,orb,spin,ph)
                        coeff = np.cos(2*np.pi * beta * m) * (-1)**ph
                        self.terms.append([i, i, 0, 0, 0, pid_x, coeff])
                        coeff = np.cos(2*np.pi * beta * n) * (-1)**ph
                        self.terms.append([i, i, 0, 0, 0, pid_y, coeff])
        return self


    def add_impurity(self, pid=None):
        if pid is None:
            pid = P.Jimp
        o = self.sys.get_mid_site()
        for orb in range(self.norb):
            for spin in range(self.nspin):
                for ph in range(self.nph):
                    i = self._index(o,orb,spin,ph)
                    self.terms.append([i, i, 0, 0, 0, pid, (-1)**ph*(-1)**spin])
        return self


    def add_zeeman(self, pid=None):
        if pid is None:
            pid = P.hz
        if self.nspin != 2:
            raise ValueError("This term requires nspin=2")

        for ii in range(self.nsite):
            for orb in range(self.norb):
                for spin in range(self.nspin):
                    for ph in range(self.nph):
                        i = self._index(ii,orb,spin,ph)
                        sign = (-1)**ph * (-1)**spin
                        self.terms.append([i, i, 0, 0, 0, pid, sign])
        return self


    def add_zeeman_in_plane(self, pid_x=None, pid_y=None):
        if self.nspin != 2:
            raise ValueError("In-plane Zeeman requires nspin=2")

        for ii in range(self.nsite):
            for orb in range(self.norb):
                for ph in range(self.nph):
                    i_up = self._index(ii, orb, 0, ph)
                    i_dn = self._index(ii, orb, 1, ph)
                    sign = (-1)**ph

                    if pid_x is not None:
                        self.terms.append([i_up, i_dn, 0, 0, 0, pid_x, sign])
                        self.terms.append([i_dn, i_up, 0, 0, 0, pid_x, sign])

                    if pid_y is not None:
                        self.terms.append([i_up, i_dn, 0, 0, 0, pid_y, -1j*sign])
                        self.terms.append([i_dn, i_up, 0, 0, 0, pid_y,  1j*sign])
        return self


    def add_rashba(self, pid=None):
        if pid is None:
            pid = P.rsoc
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
                    if ph == 0:
                        coeff_up_down = 1j*(dy+1j*dx)
                        coeff_down_up = 1j*(dy-1j*dx)
                    else:
                        coeff_up_down = -np.conj(1j*(dy+1j*dx))
                        coeff_down_up = -np.conj(1j*(dy-1j*dx))
                    self.terms.append([i_up, j_dn, *rij, pid, coeff_up_down])
                    self.terms.append([i_dn, j_up, *rij, pid, coeff_down_up])
        return self


    def add_stag_pot(self, pid=None):
        if pid is None:
            pid = P.Vstag
        for ii in range(self.nsite):
            for orb in range(self.norb):
                for spin in range(self.nspin):
                    for ph in range(self.nph):
                        i = self._index(ii,orb,spin,ph)
                        coeff = (-1)**(i%2) * (-1)**ph
                        self.terms.append([i, i, 0, 0, 0, pid, coeff])
        return self


    def add_haldane(self, pid=None):
        if pid is None:
            pid = P.tH
        if self.nspin == 2:
            raise ValueError("This term requires nspin=1")

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
                    coeff = 1j * nu
                    if pid is not None:
                        self.terms.append([i, j, *rij, pid, coeff])
        return self


    def add_kmsoc(self, pid=None):
        if pid is None:
            pid = P.kmsoc
        if self.nspin != 2:
            raise ValueError("This term requires nspin=2")

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
                        coeff = 1j*(-1)**spin * nu
                        if pid is not None:
                            self.terms.append([i, j, *rij, pid, coeff])
        return self


    def add_s_wave(self, pid=None):
        if pid is None:
            pid = P.delta_s
        if self.nspin != 2 or self.nph != 2:
            raise ValueError("This term requires nspin=2 and nph=2")

        for ii in range(self.nsite):
            for orb in range(self.norb):
                i_el_up = self._index(ii, orb, 0, 0)
                i_hl_dn = self._index(ii, orb, 1, 1)
                i_el_dn = self._index(ii, orb, 1, 0)
                i_hl_up = self._index(ii, orb, 0, 1)

                self.terms.append([i_el_up, i_hl_dn, 0, 0, 0, pid, 1.0])
                self.terms.append([i_hl_dn, i_el_up, 0, 0, 0, pid, 1.0])
                self.terms.append([i_el_dn, i_hl_up, 0, 0, 0, pid, 1.0])
                self.terms.append([i_hl_up, i_el_dn, 0, 0, 0, pid, 1.0])
        return self

    def add_pip_pairing(self, pid=None):
        if pid is None:
            pid = P.delta_pip
        if self.nph != 2:
            raise ValueError("Requires nph=2")

        for c in range(len(self.hops_nn)):
            ii = int(self.hops_nn[c,3])
            jj = int(self.hops_nn[c,4])
            rij = self.hops_nn[c,6:9]
            theta = np.arctan2(rij[1], rij[0])
            phase = np.exp(1j * theta)

            for orb in range(self.norb):
                i_el_up = self._index(ii, orb, 0, 0)
                j_hl_up = self._index(jj, orb, 0, 1)
                self.terms.append([i_el_up, j_hl_up, *rij, pid,  phase])
                j_el_up = self._index(jj, orb, 0, 0)
                i_hl_up = self._index(ii, orb, 0, 1)
                self.terms.append([j_el_up, i_hl_up, *rij, pid, -phase])

                i_el_dn = self._index(ii, orb, 1, 0)
                j_hl_dn = self._index(jj, orb, 1, 1)
                self.terms.append([i_el_dn, j_hl_dn, *rij, pid,  phase])
                j_el_dn = self._index(jj, orb, 1, 0)
                i_hl_dn = self._index(ii, orb, 1, 1)
                self.terms.append([j_el_dn, i_hl_dn, *rij, pid, -phase])
        return self


    def add_multiorb_sk(self, amp_matrices, SK_params=None, pid_matrix=None):
        for c, amp_matrix in enumerate(amp_matrices):
            ii = int(self.hops_nn[c, 3])
            jj = int(self.hops_nn[c, 4])
            rij = self.hops_nn[c, 6:9]

            for orb_i in range(self.norb):
                for orb_j in range(self.norb):
                    if pid_matrix is not None:
                        pid = pid_matrix[c][orb_i, orb_j]
                        if pid is None:
                            continue
                        for spin in range(self.nspin):
                            for ph in range(self.nph):
                                i = self._index(ii, orb_i, spin, ph)
                                j = self._index(jj, orb_j, spin, ph)
                                self.terms.append([i, j, *rij, pid, (-1)**ph])
                    else:
                        amp = float(amp_matrix[orb_i, orb_j])
                        if amp == 0.0:
                            continue
                        for spin in range(self.nspin):
                            for ph in range(self.nph):
                                i = self._index(ii, orb_i, spin, ph)
                                j = self._index(jj, orb_j, spin, ph)
                                self.terms.append([i, j, *rij, P.const, amp * (-1)**ph])

        if SK_params is None or 'onsite' not in SK_params:
            return self

        onsite_dict = SK_params['onsite']
        onsite_orbs = ('s', 'px', 'py', 'pz')

        for site in range(self.nsite):
            for orb_idx, orb_name in enumerate(onsite_orbs):
                if orb_idx >= self.norb:
                    break
                amp = float(onsite_dict[orb_name])
                if amp == 0.0:
                    continue
                for spin in range(self.nspin):
                    for ph in range(self.nph):
                        i = self._index(site, orb_idx, spin, ph)
                        self.terms.append([i, i, 0.0, 0.0, 0.0, P.const, amp * (-1)**ph])
        return self


    def sk_table(self, orb_type, SK_params=None):
        hopping_matrices = []
        for c in range(len(self.hops_nn)):
            rij = np.array(self.hops_nn[c, 6:9], dtype=np.float64)
            rnorm = np.linalg.norm(rij)
            if rnorm == 0:
                raise ValueError("Bond vector cannot be zero")

            l, m, n = rij / rnorm

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
                amp_matrix[0,0] = Vss
                amp_matrix[0,1:4] = [ l*Vsp,  m*Vsp,  n*Vsp]
                amp_matrix[1:4,0] = [-l*Vsp, -m*Vsp, -n*Vsp]
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

    def add_fm(self, pid=None):
        if pid is None:
            pid = P.Jfm
        for ii in range(self.nsite):
            for orb in range(self.norb):
                for ph in range(self.nph):
                    i = self._index(ii,orb,0,ph)
                    coeff = (-1)**ph
                    self.terms.append([i, i, 0, 0, 0, pid, coeff])
        return self


    # ------------------------------------------------------------------
    # Finalise and build
    # ------------------------------------------------------------------

    def finalize(self):
        """Convert terms list to arrays, splitting off Peierls (phi) terms once."""
        terms = np.asarray(self.terms, dtype=np.complex128)
        phi_pid = P.phi
        mask = terms[:, 5].real == phi_pid
        self._terms_phi = terms[mask]
        self._terms = terms[~mask]
        return self

    def get_terms(self):
        return self.terms

    def build(self, k=None):
        """Build H(k) using stored self.params."""
        if k is None:
            k = np.zeros(3)
        k = np.asarray(k, dtype=np.float64)
        return ham_numba(self.nstate(), self._terms, self._terms_phi, self.params, k)

    def hamf(self, lam, k=None):
        """Build H(k) with an explicit parameter array (for parameter sweeps)."""
        if k is None:
            k = np.zeros(3)
        k = np.asarray(k, dtype=np.float64)
        return ham_numba(self.nstate(), self._terms, self._terms_phi, lam, k)


# ---------------------------------------------------------------------------
# Numba-compiled Hamiltonian assembler
# ---------------------------------------------------------------------------

@nb.njit
def ham_numba(nstate, terms, terms_phi, lam, k):
    """
    Assemble H(k) from pre-split term arrays.

    terms     : regular terms  → H[i,j] -= lam[pid] * coeff * e^{i r·k}
    terms_phi : Peierls terms  → H[i,j] -= e^{i lam[pid]*coeff} * e^{i r·k}
    """
    H = np.zeros((nstate, nstate), dtype=np.complex128)

    for c in range(terms_phi.shape[0]):
        i   = int(terms_phi[c, 0].real)
        j   = int(terms_phi[c, 1].real)
        dx  = terms_phi[c, 2].real
        dy  = terms_phi[c, 3].real
        dz  = terms_phi[c, 4].real
        pid = int(terms_phi[c, 5].real)
        coeff = terms_phi[c, 6]
        bloch = np.exp(1j * (dx*k[0] + dy*k[1] + dz*k[2]))
        H[i, j] -= np.exp(1j * lam[pid] * coeff) * bloch

    for c in range(terms.shape[0]):
        i   = int(terms[c, 0].real)
        j   = int(terms[c, 1].real)
        dx  = terms[c, 2].real
        dy  = terms[c, 3].real
        dz  = terms[c, 4].real
        pid = int(terms[c, 5].real)
        coeff = terms[c, 6]
        bloch = np.exp(1j * (dx*k[0] + dy*k[1] + dz*k[2]))
        H[i, j] -= lam[pid] * coeff * bloch

    return H


# ---------------------------------------------------------------------------
# Mean-field Hamiltonian (Hubbard Hartree)
# ---------------------------------------------------------------------------

class MeanFieldHamiltonian:
    """Hubbard Hartree mean-field wrapper around a finalized Hamiltonian.

    H_MF(k) = H_kin(k) + U Σ_{i,σ} <n_{i,−σ}> c†_{i,σ} c_{i,σ}

    The base Hamiltonian must have nspin=2 and be finalized.
    Fields are updated by MFSolver; this class only builds H_MF(k) efficiently.

    Parameters
    ----------
    ham   : Hamiltonian   base kinetic Hamiltonian (finalized, nspin=2)
    U     : float         on-site Hubbard repulsion (same units as hopping t)
    """

    def __init__(self, ham: 'Hamiltonian', U: float):
        if ham.nspin != 2:
            raise ValueError("MeanFieldHamiltonian requires nspin=2")
        if ham._terms is None:
            raise RuntimeError("Call ham.finalize() before creating MeanFieldHamiltonian")
        self.ham   = ham
        self.U     = float(U)
        self.nsite = ham.nsite

        # Precompute spin-up / spin-dn matrix indices for each site (orb=0, ph=0)
        self.idx_up = np.array([ham._index(i, 0, 0, 0) for i in range(ham.nsite)],
                               dtype=np.int64)
        self.idx_dn = np.array([ham._index(i, 0, 1, 0) for i in range(ham.nsite)],
                               dtype=np.int64)

    def build(self, k, fields):
        """Build H_MF(k) for the given density fields.

        Parameters
        ----------
        k      : (3,) k-vector in Cartesian coordinates
        fields : (nsite, 2) array — fields[i, 0]=<n_{i,↑}>, fields[i, 1]=<n_{i,↓}>

        Returns
        -------
        H : (nstate, nstate) complex128 Hamiltonian matrix
        """
        k      = np.asarray(k,      dtype=np.float64)
        fields = np.asarray(fields, dtype=np.float64)
        H = self.ham.build(k)
        _add_hubbard_mf(H, self.idx_up, self.idx_dn, self.U, fields)
        return H


@nb.njit
def _add_hubbard_mf(H, idx_up, idx_dn, U, fields):
    """Add U*<n_{−σ}> Hartree shift to the diagonal of H in place.

    fields[i, 0] = <n_{i,↑}>,  fields[i, 1] = <n_{i,↓}>
    Spin-up   diagonal: H[idx_up[i], idx_up[i]] += U * fields[i, 1]
    Spin-down diagonal: H[idx_dn[i], idx_dn[i]] += U * fields[i, 0]
    """
    nsite = idx_up.shape[0]
    for i in range(nsite):
        H[idx_up[i], idx_up[i]] += U * fields[i, 1]
        H[idx_dn[i], idx_dn[i]] += U * fields[i, 0]


@nb.njit
def ham_custom(nstate, terms, lam, k):
    H = np.zeros((nstate, nstate), dtype=np.complex128)
    lamb = 1.0

    for c in range(terms.shape[0]):
        i   = int(terms[c, 0].real)
        j   = int(terms[c, 1].real)
        dx  = terms[c, 2].real
        dy  = terms[c, 3].real
        dz  = terms[c, 4].real
        pid = int(terms[c, 5].real)
        coeff = terms[c, 6]
        bloch = np.exp(1j * (dx*k[0] + dy*k[1] + dz*k[2]))
        if pid == 16:
            H[i, j] -= np.exp(1j * lam[pid] * coeff) * bloch
        elif pid == 3:
            dij = np.sqrt(dx**2 + dy**2 + dz**2)
            exp_factor = np.exp(-(dij - dz) / lamb)
            H[i, j] -= lam[pid] * exp_factor * coeff * bloch
        else:
            H[i, j] -= lam[pid] * coeff * bloch
    return H
