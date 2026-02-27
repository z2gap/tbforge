import numpy as np
import scipy as sp
import numba as nb
from typing import List, Tuple
from .finite import *

class Lattice:
    def __init__(self, latVecs, basisVecs, bc=[1,1,1]) -> None:
        self.latVecs = np.array(latVecs)     #lattice vecs 
        self.basisVecs = np.array(basisVecs) #sublattice vecs
        self.bc = np.array(bc)
        self.dim = len(latVecs)              #lattice dimension (1D/2D/3D)
        self.n_sites = len(basisVecs)


    def is_bulk(self): 
        return True
    

    def bz_area(self):
        b1, b2, _ = self.bzVecs()
        return b1[0]*b2[1] - b1[1]*b2[0] 


    def bzVecs(self) -> np.ndarray:
        a1, a2, a3 = self.latVecs
        V = np.dot(a1, np.cross(a2, a3))  # unit cell V
        b1 = 2 * np.pi * np.cross(a2, a3) / V
        b2 = 2 * np.pi * np.cross(a3, a1) / V
        b3 = 2 * np.pi * np.cross(a1, a2) / V
        return np.array([b1, b2, b3])
    

    def make_finite(self, shape, center=True):
        Lx, Ly, Lz = shape
        if center:
            mid = np.array([Lx//2, Ly//2, Lz//2])
        else:
            mid = np.zeros(3, dtype=int)
        positions = []
        cell_indices = []
        basis_indices = []

        for i in range(Lx):
            for j in range(Ly):
                for k in range(Lz):
                    R = (
                        (i - mid[0]) * self.latVecs[0] +
                        (j - mid[1]) * self.latVecs[1] +
                        (k - mid[2]) * self.latVecs[2]
                    )

                    for ib, tau in enumerate(self.basisVecs):
                        positions.append(R + tau)
                        cell_indices.append([i, j, k])
                        basis_indices.append(np.array(self.latVecs)* np.array(shape))

        return Finite(
            np.array(positions),
            np.array(basis_indices),
        )
            

    def find_neighbor_dist(self, hop_order=1, nx=3, ny=3, nz=1):
        bulk_coords = []
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    shift = (ix*np.array(self.latVecs[0]) +
                            iy*np.array(self.latVecs[1]) +
                            iz*np.array(self.latVecs[2]))
                    for tau in self.basisVecs:
                        bulk_coords.append(shift + np.array(tau))

        bulk_coords = np.array(bulk_coords)
        coords_xy = bulk_coords[:, :2]

        tree = sp.spatial.KDTree(coords_xy)
        distances, _ = tree.query(coords_xy, k=10)
        all_distances = np.unique(distances[:, 1:].round(8))
        if hop_order > len(all_distances):
            raise ValueError(
                f"hop_order={hop_order} exceeds available neighbors ({len(all_distances)})"
            )
        return all_distances[hop_order - 1]


    @classmethod
    def chain(cls, a=1.0):
        """1D linear chain"""
        return cls([[a]], [[0]])           
    
    @classmethod
    def square(cls, a=1.0, c=1.0):
        """Square lattice"""
        a1 = [a, 0, 0]
        a2 = [0, a, 0]
        a3 = [0, 0, c]
        basis = [[0, 0, 0]]
        return cls([a1, a2, a3], basis)
    
    @classmethod
    def lieb(cls, a=1.0, c=1.0):
        """Lieb lattice"""
        a1 = [a, 0, 0]
        a2 = [0, a, 0]
        a3 = [0, 0, c]
        basis = [
            [0, 0, 0],          # corner site
            [a/2, 0, 0],        # x-edge center
            [0, a/2, 0],        # y-edge center
        ]
        return cls([a1, a2, a3], basis)

    @classmethod
    def triangular(cls, a=1.0, c=1.0):
        """Triangular lattice"""
        a1 = [a, 0, 0]
        a2 = [a/2, a*np.sqrt(3)/2, 0]
        a3 = [0, 0, c]
        basis = [[0, 0, 0]]
        return cls([a1, a2, a3], basis)

    @classmethod
    def honeycomb(cls, a=1.0, c=1.0):
        """Honeycomb lattice with fractional basis"""
        import numpy as np

        # Lattice vectors
        a1 = np.array([a, 0, 0])
        a2 = np.array([a/2, a*np.sqrt(3)/2, 0])
        a3 = np.array([0, 0, c])

        # Basis in fractional coordinates
        fA = np.array([1/3, 1/3, 0])
        fB = np.array([2/3, 2/3, 0])

        # Convert to Cartesian
        tauA = fA[0]*a1 + fA[1]*a2 + fA[2]*a3
        tauB = fB[0]*a1 + fB[1]*a2 + fB[2]*a3

        return cls([a1, a2, a3], [tauA, tauB])
    
    @classmethod
    def honeycomb2(cls, a=1.0, c=1.0):
        """Honeycomb lattice with 4 atoms/cell (armchair)"""
        a1 = [a, 0, 0]
        a2 = [0, a*np.sqrt(3), 0]
        a3 = [0, 0, c]
        tau1 = [0, 0, 0]
        tau2 = [a/2, a*np.sqrt(3)/2, 0]
        tau3 = [a, 0, 0]
        tau4 = [3*a/2, a*np.sqrt(3)/2, 0]
        return cls([a1, a2, a3], [tau1, tau2, tau3, tau4])
    
    @classmethod
    def kagome(cls, a=1.0, c=1.0):
        """Kagome lattice"""
        a1 = [a, 0, 0]
        a2 = [a/2, a*np.sqrt(3)/2, 0]
        a3 = [0, 0, c]
        tauA = [0, 0, 0]
        tauB = [a/2, 0, 0]
        tauC = [a/4, a*np.sqrt(3)/4, 0]
        return cls([a1, a2, a3], [tauA, tauB, tauC])
    
    @classmethod
    def kagome2(cls, a=1.0, c=1.0):
        """Rectangular kagome lattice"""
        a1 = [2*a, 0, 0]
        a2 = [0, 2*a*np.sqrt(3), 0]
        a3 = [0, 0, c]
        tau1 = [0, 0, 0]
        tau2 = [a1[0]/2, a1[1]/2, 0]
        tau3 = [3*a/4, a*np.sqrt(3)/2, 0]
        tau4 = [(a1[0]+a2[0])/2, (a1[1]+a2[1])/2, 0]
        tau5 = [a2[0]/2, a2[1]/2, 0]
        tau6 = [a1[0]/4 + 3*a2[0]/4, 3*a2[1]/4, 0]
        return cls([a1, a2, a3], [tau1, tau2, tau3, tau4, tau5, tau6])
    
    
    @classmethod
    def bilayer_kagome(cls, a=1.0, c=1.0, h=[0.,0.]):
        a1 = [a, 0, 0]
        a2 = [a/2, a*np.sqrt(3)/2, 0]
        a3 = [0, 0, c]

        # Layer 1 basis (z=0)
        tauA1 = [0, 0, 0]
        tauB1 = [a/2, 0, 0]
        tauC1 = [a/4, a*np.sqrt(3)/4, 0]

        # Layer 2 basis (z=c/2)
        shift = np.array(a1) * h[0] + np.array(a2) * h[1]
        tauA2 = list(np.array(tauA1) + shift + np.array([0, 0, c/2]))
        tauB2 = list(np.array(tauB1) + shift + np.array([0, 0, c/2]))
        tauC2 = list(np.array(tauC1) + shift + np.array([0, 0, c/2]))

        basis = [tauA1, tauB1, tauC1, tauA2, tauB2, tauC2]
        return cls([a1, a2, a3], basis)
    

    def find_kgrid(self, pbc=[1,1,1], mesh=[5,5,1]) -> np.ndarray:
        b_vectors = self.bzVecs()  # returns [b1, b2, b3] for 3D
        kgrid = []
        nkx, nky, nkz = mesh
        for i in range(nkx):
            for j in range(nky):
                for k in range(nkz):
                    kvec = (i / nkx) * pbc[0] * b_vectors[0] \
                        + (j / nky) * pbc[1] * b_vectors[1] \
                        + (k / nkz) * pbc[2] * b_vectors[2]
                    kgrid.append(kvec)
        return np.array(kgrid)
    
    
    def find_kpath(self, kpath_labels=None, kpath_frac=None, n_kpts=120):
        b1, b2, b3 = self.bzVecs()

        # Determine default path if none provided
        if kpath_labels is None or kpath_frac is None:
            angle12 = np.arccos(np.clip(np.dot(b1, b2) / (np.linalg.norm(b1)*np.linalg.norm(b2)), -1, 1))
            angle23 = np.arccos(np.clip(np.dot(b2, b3) / (np.linalg.norm(b2)*np.linalg.norm(b3)), -1, 1))

            # Default paths based on BZ angles
            if np.isclose(angle12, np.pi/2, atol=1e-3) and np.isclose(angle23, np.pi/2, atol=1e-3):
                # Square / cubic
                kpath_labels = ["G", "M", "X", "G"]
                kpath_frac = np.array([[0,0,0], [0.5,0.5,0], [0.5,0,0], [0,0,0]])
            elif np.isclose(angle12, 2*np.pi/3, atol=1e-3):
                # Hexagonal
                kpath_labels = ["G", "M", "K", "G"]
                kpath_frac = np.array([[0,0,0], [0.5,0,0], [2/3,1/3,0], [0,0,0]])
            else:
                raise ValueError("Cannot infer default BZ path for this lattice geometry")

        # Convert fractional coordinates to Cartesian
        kpath_cart = np.array([p[0]*b1 + p[1]*b2 + p[2]*b3 for p in kpath_frac])

        # Segment lengths
        segment_lengths = np.linalg.norm(np.diff(kpath_cart, axis=0), axis=1)
        total_length = np.sum(segment_lengths)
        nk_list = [max(2, int(round(n_kpts * l / total_length))) for l in segment_lengths]

        # Build k-path
        kpath = [kpath_cart[0]]
        kpath_1d = [0.0]
        tick_locs = [0.0]
        tick_labels = [kpath_labels[0]]
        dist_accum = 0.0

        for i, nk in enumerate(nk_list):
            start = kpath_cart[i]
            end = kpath_cart[i+1]
            endpoint = (i == len(nk_list)-1)
            segment = np.linspace(start, end, nk, endpoint=endpoint)

            if not endpoint:
                segment = segment[1:]  # avoid duplicate at junction

            diffs = np.linalg.norm(np.diff(np.vstack([start, segment]), axis=0), axis=1)
            dist_segment = dist_accum + np.cumsum(diffs)
            dist_accum = dist_segment[-1]

            kpath.extend(segment)
            kpath_1d.extend(dist_segment)

            tick_locs.append(dist_accum)
            tick_labels.append(kpath_labels[i+1])

        ticks = [tick_locs, tick_labels]
        return np.array(kpath), np.array(kpath_1d), ticks




    