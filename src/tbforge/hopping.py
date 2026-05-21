import numpy as np
import scipy as sp
import numba as nb
from typing import List, Tuple

try:
    from . import find_hopsf90 as _f90
    _HAS_F90 = True
except ImportError:
    _HAS_F90 = False


class Hopping:
    def __init__(self, sys=None):
        self.sys = sys

    def find_hops(self, sys=None, order=1, rz=0., d=1, tol=1e-3):
        if sys is not None:
            self.sys = sys
        rxy = self.sys.find_neighbor_dist(order)

        if _HAS_F90:
            return self._find_hops_f90(rxy, rz, d, tol)

        return self._find_hops_py(rxy, rz, d, tol)

    def _find_hops_f90(self, rxy, rz, d, tol):
        hopsxy, nhopsxy, hopsz, nhopsz = _f90.find_hops(
            np.ascontiguousarray(self.sys.basis_vecs, dtype=np.float64),
            np.ascontiguousarray(self.sys.lat_vecs,   dtype=np.float64),
            self.sys.bc.astype(np.int32),
            float(rxy),
            float(rz),
            int(d),
            float(tol),
        )
        hopsxy = hopsxy[:nhopsxy]
        hopsz  = hopsz[:nhopsz]
        hopsxy_sorted = hopsxy[np.argsort(hopsxy[:, 3])]
        return hopsxy_sorted, hopsz

    def _find_hops_py(self, rxy, rz, d, tol):
        rlist = self.sys.basis_vecs

        range_x = range(-d, d+1) if self.sys.bc[0] else [0]
        range_y = range(-d, d+1) if self.sys.bc[1] else [0]
        range_z = range(-d, d+1) if self.sys.bc[2] else [0]

        hopsxy = []
        hopsz = []
        for i in range_x:
            for j in range_y:
                for k in range_z:
                    for s in range(len(rlist)):
                        for sp in range(len(rlist)):
                            ri = rlist[s]
                            rj = rlist[sp] + i*self.sys.lat_vecs[0]\
                                           + j*self.sys.lat_vecs[1]\
                                           + k*self.sys.lat_vecs[2]
                            rij = ri - rj
                            dxy = np.linalg.norm(ri[:2] - rj[:2])
                            dist = np.linalg.norm(rij)

                            if abs(rxy - dxy) < tol and abs(ri[2] - rj[2]) < tol:
                                hopsxy.append([i, j, k, s, sp, dxy, rij[0], rij[1], rij[2]])
                            if rz > 0 and abs(dist - rz) < tol and abs(ri[2] - rj[2]) > tol:
                                hopsz.append([i, j, k, s, sp, dist, *rij])
        hopsxy = np.array(hopsxy)
        hopsz  = np.array(hopsz)
        hopsxy_sorted = hopsxy[np.argsort(hopsxy[:, 3])]
        return hopsxy_sorted, hopsz
    
    def get_hops_nn(self):
        return self.find_hops(order=1)[0]
    
    def print_nnhops(self):
        h = self.find_hops(order=1)[0]
        width = 50
        print("=" * width)
        print("Hoppings from i in Ri:(0,0,0) to j in Rj".center(width))
        print(f"periodicity: {self.sys.bc}".center(width))
        print("=" * width)
        for row in h:
            i, j, k, s, sp = map(int, row[:5])
            dxy = row[5]

            print(
                f"Rj:({i:3d},{j:3d},{k:3d}), "
                f"  i:{s:3d}, "
                f"  j:{sp:3d}, "
                f"  |dij|:{dxy:.3f}, "
            )
        # for row in h:
        #     print(f"site-i:{int(row[3])} site-j:{int(row[4])}")
    
    def get_hops_2nn(self):
        return self.find_hops(order=2)[0]
    
    def get_hopsz(self, rz):
        return self.find_hops(rz=rz)[1]

    def get_interlayer_hops(self, r_cut, tol=1e-3):
        """Find all interlayer site pairs (different z) within a 3D cutoff.

        Args:
            r_cut: Maximum 3D distance to include.
            tol: Tolerance for z-layer discrimination and distance bound.

        Returns:
            ndarray of shape (n_hops, 9): [n1, n2, n3, site_i, site_j, dist, dx, dy, dz].
        """
        rlist = self.sys.basis_vecs
        lat = self.sys.lat_vecs
        periodic = [i for i in range(3) if self.sys.bc[i]]
        if periodic:
            min_lat = min(np.linalg.norm(lat[i]) for i in periodic)
            d = max(2, int(np.ceil(r_cut / min_lat)) + 1)
        else:
            d = 0

        range_x = range(-d, d + 1) if self.sys.bc[0] else [0]
        range_y = range(-d, d + 1) if self.sys.bc[1] else [0]
        range_z = range(-d, d + 1) if self.sys.bc[2] else [0]

        hops = []
        for nx in range_x:
            for ny in range_y:
                for nz in range_z:
                    for s in range(len(rlist)):
                        for sp in range(len(rlist)):
                            ri = rlist[s]
                            rj = rlist[sp] + nx*lat[0] + ny*lat[1] + nz*lat[2]
                            rij = ri - rj
                            dist = np.linalg.norm(rij)
                            if abs(ri[2] - rj[2]) > tol and dist <= r_cut + tol:
                                hops.append([nx, ny, nz, s, sp, dist, rij[0], rij[1], rij[2]])
        return np.array(hops) if hops else np.empty((0, 9))
    
    def get_kmsign(self, full_map=False):
        self.hops_nn = self.get_hops_nn()
        self.hops_nnn = self.get_hops_2nn()

        #From NN list, make nn_list[i] that contains 
        # all NN hoppings of site i
        hops_dict = [[] for _ in range(self.sys.n_sites)]
        for c in range(len(self.hops_nn)):
            i = int(self.hops_nn[c,3])
            j = int(self.hops_nn[c,4])
            rij = self.hops_nn[c,6:9]   # NN displacement vector
            hops_dict[i].append((j, rij))

        #for each (i,j) pair in NNN list, find
        #intersection b/w nn_list[i] & nn_list[j]
        #this gives intermediate site k (i->k->j)
        im = []
        for c in range(len(self.hops_nnn)):
            i = int(self.hops_nnn[c,3])
            j = int(self.hops_nnn[c,4])
            dij = self.hops_nnn[c,6:9]
            for k, d1 in hops_dict[i]:
                for k2, d2 in hops_dict[k]:
                    if k2 == j and np.allclose(d1+d2, dij):
                        im.append([i, j, k, d1, d2])
                        break
                    
        im = np.array(im, dtype=object) 
        #KM sign is v = sgn(d/|d|), where d=|d1xd2|
        #where d1=rk-ri and d2=rj-rk
        d1 = np.vstack(im[:,3])
        d2 = np.vstack(im[:,4])
        d = np.cross(d1, d2)
        #list of KM sigs for all site i
        nu_list = np.sign(d[:,2])
        if full_map: 
            return np.array([[*a[:3],b] for a,b in zip(im,nu_list)])
        else:
            return nu_list