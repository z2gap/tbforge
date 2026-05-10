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

    def find_neighbor_dist(self, hop_order=1, nx=2, ny=2, nz=1):
        ix, iy, iz = np.meshgrid(range(nx), range(ny), range(nz), indexing='ij')
        cell_indices = np.stack([ix, iy, iz], axis=-1).reshape(-1, 3)
        shifts = cell_indices @ self.lat_vecs
        bulk_coords = (shifts[:, None, :] + self.positions[None, :, :]).reshape(-1, 3)

        if np.allclose(bulk_coords[:, 2], bulk_coords[0, 2]):
            coords = bulk_coords[:, :2]
        else:
            coords = bulk_coords

        k = max(20, hop_order * 10 + 1)
        tree = KDTree(coords)
        distances, _ = tree.query(coords, k=min(k, len(coords)))
        all_distances = np.unique(distances[:, 1:].round(8))

        if hop_order > len(all_distances):
            raise ValueError(
                f"hop_order={hop_order} exceeds available neighbors ({len(all_distances)})"
            )
        return all_distances[hop_order - 1]
