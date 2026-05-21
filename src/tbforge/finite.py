import numpy as np
from scipy.spatial import KDTree


class Finite:
    def __init__(self, lat, shape, bc=None):
        self.lat = lat
        self.shape = np.array(shape)
        self.bc = np.array(bc if bc is not None else [0, 0, 0])
        self.positions = self._generate_positions()
        self.lat_vecs = np.array([shape[0]*lat.lat_vecs[0],
                                   shape[1]*lat.lat_vecs[1],
                                   shape[2]*lat.lat_vecs[2]])

    def get_mid_site(self):
        center = self.positions.mean(axis=0)
        dists = np.linalg.norm(self.positions - center, axis=1)
        return np.argmin(dists)

    def get_edge_sites(self):
        pos = self.positions
        tree = KDTree(pos)
        cutoff = self.lat.find_neighbor_dist(1)
        neighbor_counts = np.array([len(tree.query_ball_point(p, cutoff)) - 1 for p in pos])
        max_neighbors = neighbor_counts.max()
        return np.where(neighbor_counts < max_neighbors)[0]

    def site_to_cell(self, site):
        num_basis = len(self.lat.basis_vecs)
        cell_index = site // num_basis
        ix = cell_index // (self.shape[1] * self.shape[2])
        remainder = cell_index % (self.shape[1] * self.shape[2])
        iy = remainder // self.shape[2]
        iz = remainder % self.shape[2]
        return (ix, iy, iz)

    def is_bulk(self):
        return False

    def _generate_positions(self, centered=True):
        mid = self.shape // 2 if centered else np.zeros(3, dtype=int)
        lat = self.lat
        positions = []
        for ix in range(self.shape[0]):
            for iy in range(self.shape[1]):
                for iz in range(self.shape[2]):
                    R = (
                          (ix - mid[0]) * lat.lat_vecs[0]
                        + (iy - mid[1]) * lat.lat_vecs[1]
                        + (iz - mid[2]) * lat.lat_vecs[2]
                    )
                    for tau in lat.basis_vecs:
                        positions.append(R + tau)
        return np.array(positions)

    @property
    def n_sites(self):
        return self.positions.shape[0]

    @property
    def basis_vecs(self):
        return self.positions

    def find_neighbor_dist(self, hop_order=1, n_img=2):
        # Finite system: bc=[0,0,0], so all positions are already in self.positions.
        # Delegate to the underlying unit-cell lattice for tiling in periodic directions.
        return self.lat.find_neighbor_dist(hop_order=hop_order, n_img=n_img)
