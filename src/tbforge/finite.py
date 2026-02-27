import numpy as np
import scipy as sp
import numba as nb
from typing import List, Tuple

class Finite:
    def __init__(self, lat, shape, bc=[0,0,0]):
        self.lat = lat
        self.shape = np.array(shape)
        self.bc = bc
        self.positions = self._generate_positions()
        self.latVecs = np.array([shape[0]*lat.latVecs[0],
                                shape[1]*lat.latVecs[1],
                                shape[2]*lat.latVecs[2]])
        
    def get_mid_site(self):
        center = self.positions.mean(axis=0)
        dists = np.linalg.norm(self.positions - center, axis=1)
        idx = np.argmin(dists)
        return idx
    
    def get_edge_sites(self):
        pos = self.positions
        tree = sp.spatial.KDTree(pos)
        #find NN dist
        cutoff = self.lat.find_neighbor_dist(1)
        # Count neighbors within cutoff for each site
        neighbor_counts = np.array([len(tree.query_ball_point(p, cutoff)) - 1 for p in pos])
        max_neighbors = neighbor_counts.max()
        # Edge sites have fewer neighbors than bulk maximum
        edge_sites = np.where(neighbor_counts < max_neighbors)[0]
        return edge_sites
    
    def site_to_cell(self, site):
        num_basis = len(self.lat.basisVecs)  # number of sites per unit cell
        cell_index = site // num_basis        # which unit cell in the flattened 3D grid

        # unravel the flat index to 3D indices
        ix = cell_index // (self.shape[1] * self.shape[2])
        remainder = cell_index % (self.shape[1] * self.shape[2])
        iy = remainder // self.shape[2]
        iz = remainder % self.shape[2]
        return (ix, iy, iz)
        
    def is_bulk(self): 
        return False

    def _generate_positions(self, cetered=True):
        if cetered is True:
            mid = self.shape//2
        else:
            mid = np.zeros(2)
        lat = self.lat
        positions = []
        for ix in range(self.shape[0]):
            for iy in range(self.shape[1]):
                for iz in range(self.shape[2]):
                    R = (
                          (ix-mid[0]) * lat.latVecs[0]
                        + (iy-mid[1]) * lat.latVecs[1]
                        + (iz-mid[2]) * lat.latVecs[2]
                    )
                    for tau in lat.basisVecs:
                        positions.append(R + tau)
        return np.array(positions)

    @property
    def n_sites(self):
        return self.positions.shape[0]
        
    def find_neighbor_dist(self, hop_order=1, nx=2, ny=2, nz=1):
        bulk_coords = []
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    shift = (ix*np.array(self.latVecs[0]) +
                            iy*np.array(self.latVecs[1]) +
                            iz*np.array(self.latVecs[2]))
                    for tau in self.positions:
                        bulk_coords.append(shift + np.array(tau))

        bulk_coords = np.array(bulk_coords)
        
        #layer at z=0
        layer0 = bulk_coords[bulk_coords[:,2]==0.0]
        tree = sp.spatial.KDTree(layer0)
        distances, _ = tree.query(layer0, k=10)
        all_distances = np.unique(distances[:, 1:].round(8))
        if hop_order > len(all_distances):
            raise ValueError(
                f"hop_order={hop_order} exceeds available neighbors ({len(all_distances)})"
            )
        return all_distances[hop_order - 1]
