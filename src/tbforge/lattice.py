import numpy as np
from functools import cached_property
from scipy.spatial import KDTree
from .finite import Finite


class Lattice:
    def __init__(self, lat_vecs, basis_vecs, bc=None) -> None:
        self.lat_vecs = np.array(lat_vecs, dtype=float)
        self.basis_vecs = np.array(basis_vecs, dtype=float)
        self.bc = np.array(bc if bc is not None else [1, 1, 1])
        self.dim = len(lat_vecs)
        self.n_sites = len(basis_vecs)

    def is_bulk(self):
        return True

    def bz_area(self):
        b1, b2 = self.bz_vecs[:2]
        return float(np.linalg.norm(np.cross(b1, b2)))

    @cached_property
    def bz_vecs(self) -> np.ndarray:
        a1, a2, a3 = self.lat_vecs
        V = np.dot(a1, np.cross(a2, a3))
        b1 = 2 * np.pi * np.cross(a2, a3) / V
        b2 = 2 * np.pi * np.cross(a3, a1) / V
        b3 = 2 * np.pi * np.cross(a1, a2) / V
        return np.array([b1, b2, b3])

    def make_finite(self, shape):
        return Finite(self, shape)

    def transform(self, matrix, bc=None) -> 'Lattice':
        """Construct a supercell via a transformation matrix.

        matrix: [nx, ny, nz]  — diagonal supercell (integer or float)
                (3,3) array M — new lattice vectors = M @ lat_vecs
                               M entries may be non-integer (fractional/moiré).
        bc:     boundary conditions for the new lattice, e.g. [1, 0, 0] for a
                ribbon periodic in x and open in y.  Defaults to self.bc.

        Common patterns
        ---------------
        periodic  : Lattice.honeycomb()                         # bc=[1,1,1] default
        slab/ribbon: lat.transform([1, 10, 1], bc=[1, 0, 0])   # periodic x, open y
        finite    : lat.transform([5, 5, 1],  bc=[0, 0, 0])    # fully open
        """
        M = np.array(matrix, dtype=float)
        if M.ndim == 1:
            if M.shape != (3,):
                raise ValueError("1-D matrix must be [nx, ny, nz]")
            M = np.diag(M)
        elif M.shape != (3, 3):
            raise ValueError("matrix must be shape (3,) or (3,3)")
        if abs(np.linalg.det(M)) < 1e-10:
            raise ValueError("Transformation matrix is singular")

        new_lat = M @ self.lat_vecs   # new lattice vectors (rows)
        M_inv = np.linalg.inv(M)

        # Fractional coords of basis atoms in the original cell
        # frac_super = (cell + tau_frac) @ inv(M)  — exact, no Cartesian roundtrip
        lat_inv = np.linalg.inv(self.lat_vecs)
        tau_frac = self.basis_vecs @ lat_inv   # (n_sites, 3)

        # Search range: large enough to cover the supercell in every direction
        max_range = int(np.ceil(np.abs(M).max())) + 1
        tol = 1e-6

        all_fracs = []
        for ix in range(-max_range, max_range + 1):
            for iy in range(-max_range, max_range + 1):
                for iz in range(-max_range, max_range + 1):
                    cell = np.array([ix, iy, iz], dtype=float)
                    for tau in tau_frac:
                        frac = (cell + tau) @ M_inv
                        frac_mod = frac % 1.0
                        # Map near-1 back to 0 to avoid boundary duplicates
                        frac_mod[np.abs(frac_mod - 1.0) < tol] = 0.0
                        if np.all(frac_mod >= 0.0) and np.all(frac_mod < 1.0 - tol):
                            all_fracs.append(frac_mod)

        if not all_fracs:
            raise RuntimeError("No atoms found — check transformation matrix")

        # Deduplicate in fractional space via rounding + np.unique
        all_fracs = np.array(all_fracs)
        _, unique_idx = np.unique(np.round(all_fracs, 8), axis=0, return_index=True)
        new_basis = all_fracs[unique_idx] @ new_lat

        # Sanity check: |det(M)| * n_sites should equal atom count for integer M
        det = np.linalg.det(M)
        if abs(det - round(det)) < 0.1:
            expected = int(abs(round(det))) * self.n_sites
            if len(new_basis) != expected:
                import warnings
                warnings.warn(
                    f"Expected {expected} atoms in supercell, found {len(new_basis)}. "
                    "Check transformation matrix."
                )

        result_bc = np.array(bc) if bc is not None else self.bc.copy()
        return Lattice(new_lat, new_basis, result_bc)

    def find_neighbor_dist(self, hop_order=1, n_img=2):
        # Tile only in periodic directions: finite directions (bc=0) already have
        # all relevant sites in basis_vecs via Lattice.transform(), so no images needed.
        ranges = [range(-n_img, n_img + 1) if self.bc[i] else [0] for i in range(3)]

        ix, iy, iz = np.meshgrid(*ranges, indexing='ij')
        cell_indices = np.stack([ix, iy, iz], axis=-1).reshape(-1, 3)
        shifts = cell_indices @ self.lat_vecs
        all_coords = (shifts[:, None, :] + self.basis_vecs[None, :, :]).reshape(-1, 3)

        coords = all_coords[:, :2] if np.allclose(all_coords[:, 2], all_coords[0, 2]) else all_coords

        k = max(20, 4 * self.n_sites * hop_order + 1)
        tree = KDTree(coords)
        distances, _ = tree.query(coords, k=min(k, len(coords)))
        all_distances = np.unique(distances[:, 1:].round(8))

        if hop_order > len(all_distances):
            raise ValueError(
                f"hop_order={hop_order} exceeds available neighbors ({len(all_distances)}). "
                f"Try increasing n_img."
            )
        return all_distances[hop_order - 1]

    @classmethod
    def chain(cls, a=1.0):
        """1D linear chain"""
        return cls([[a, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 0, 0]])

    @classmethod
    def square(cls, a=1.0, c=1.0):
        """Square lattice"""
        return cls([[a, 0, 0], [0, a, 0], [0, 0, c]], [[0, 0, 0]])

    @classmethod
    def lieb(cls, a=1.0, c=1.0):
        """Lieb lattice"""
        return cls(
            [[a, 0, 0], [0, a, 0], [0, 0, c]],
            [[0, 0, 0], [a/2, 0, 0], [0, a/2, 0]],
        )

    @classmethod
    def triangular(cls, a=1.0, c=1.0):
        """Triangular lattice"""
        return cls(
            [[a, 0, 0], [a/2, a*np.sqrt(3)/2, 0], [0, 0, c]],
            [[0, 0, 0]],
        )

    @classmethod
    def honeycomb(cls, a=1.0, c=1.0):
        """Honeycomb lattice"""
        a1 = np.array([a, 0, 0])
        a2 = np.array([a/2, a*np.sqrt(3)/2, 0])
        a3 = np.array([0, 0, c])
        tauA = a1/3 + a2/3
        tauB = 2*a1/3 + 2*a2/3
        return cls([a1, a2, a3], [tauA, tauB])

    @classmethod
    def honeycomb2(cls, a=1.0, c=1.0):
        """Honeycomb lattice with 4 atoms/cell (armchair)"""
        return cls(
            [[a, 0, 0], [0, a*np.sqrt(3), 0], [0, 0, c]],
            [[0, 0, 0], [a/2, a*np.sqrt(3)/2, 0], [a, 0, 0], [3*a/2, a*np.sqrt(3)/2, 0]],
        )

    @classmethod
    def kagome(cls, a=1.0, c=1.0):
        """Kagome lattice"""
        return cls(
            [[a, 0, 0], [a/2, a*np.sqrt(3)/2, 0], [0, 0, c]],
            [[0, 0, 0], [a/2, 0, 0], [a/4, a*np.sqrt(3)/4, 0]],
        )

    @classmethod
    def kagome2(cls, a=1.0, c=1.0):
        """Rectangular kagome lattice"""
        a1 = np.array([2*a, 0, 0])
        a2 = np.array([0, 2*a*np.sqrt(3), 0])
        return cls(
            [a1, a2, [0, 0, c]],
            [
                [0, 0, 0],
                [a1[0]/2, 0, 0],
                [3*a/4, a*np.sqrt(3)/2, 0],
                [(a1[0] + a2[0])/2, (a1[1] + a2[1])/2, 0],
                [0, a2[1]/2, 0],
                [a1[0]/4, 3*a2[1]/4, 0],
            ],
        )

    @classmethod
    def bilayer_kagome(cls, a=1.0, c=1.0, h=None):
        if h is None:
            h = [0.0, 0.0]
        a1 = np.array([a, 0, 0])
        a2 = np.array([a/2, a*np.sqrt(3)/2, 0])
        a3 = np.array([0, 0, c])
        layer1 = np.array([[0, 0, 0], [a/2, 0, 0], [a/4, a*np.sqrt(3)/4, 0]], dtype=float)
        shift = h[0]*a1 + h[1]*a2 + np.array([0, 0, c/2])
        layer2 = layer1 + shift
        return cls([a1, a2, a3], np.vstack([layer1, layer2]))

    @classmethod
    def stack(cls, layer: 'Lattice', n_layers: int, d: float,
              shifts: list | None = None) -> 'Lattice':
        """Stack n_layers copies of a monolayer with interlayer separation d.

        shifts: list of (3,) in-plane offset vectors, one per layer.
                Defaults to AA stacking (all zeros).
        """
        if shifts is not None and len(shifts) != n_layers:
            raise ValueError(f"len(shifts)={len(shifts)} must equal n_layers={n_layers}")
        if d <= 0:
            raise ValueError(f"d must be positive, got {d}")

        if shifts is None:
            shifts = [np.zeros(3)] * n_layers

        a1, a2, _ = layer.lat_vecs
        a3 = np.array([0., 0., n_layers * d])

        all_basis = []
        for i, shift in enumerate(shifts):
            z_offset = np.array([0., 0., i * d])
            for tau in layer.basis_vecs:
                all_basis.append(tau + np.asarray(shift, dtype=float) + z_offset)

        return cls([a1, a2, a3], all_basis)

    @classmethod
    def moire(cls, n, m, a=1.0, d=3.35, c=None) -> 'Lattice':
        """
        Commensurate moiré bilayer from two twisted honeycomb layers.

        n, m  : integer supercell indices with n > m >= 0
        a     : honeycomb in-plane lattice constant
        d     : interlayer separation
        c     : out-of-plane period (default 2*d)

        Supercell size : N = n² + nm + m² honeycomb unit cells per layer
        Twist angle    : cos θ = (n² + 4nm + m²) / (2·N)
        Total atoms    : 4·N  (2 sublattices × N cells × 2 layers)

        Smallest non-trivial case: n=2, m=1  →  N=7, θ≈21.8°
        """
        n, m = int(n), int(m)
        if n <= 0 or m < 0 or n <= m:
            raise ValueError("Require n > m >= 0")

        a1 = np.array([a, 0.0, 0.0])
        a2 = np.array([a / 2, a * np.sqrt(3) / 2, 0.0])

        T1 = n * a1 + m * a2
        T2 = -m * a1 + (n + m) * a2

        # Twist angle from commensurability: R_θ(m·a1 + n·a2) = n·a1 + m·a2
        v1 = (n * a1 + m * a2)[:2]
        v2 = (m * a1 + n * a2)[:2]
        theta = np.arctan2(v1[1], v1[0]) - np.arctan2(v2[1], v2[0])
        print(f"Twist angle:{np.rad2deg(theta):.2f}")
        
        Rmat = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta),  np.cos(theta)]])

        # Enumerate layer-1 atoms via fractional coords in the moiré supercell
        N_cells = n * n + n * m + m * m
        print(f"No of atoms: {N_cells}")
        M2d = np.array([[n, m], [-m, n + m]], dtype=float)
        M2d_inv = np.linalg.inv(M2d)
        a_mat = np.array([a1[:2], a2[:2]])

        tol = 1e-6
        max_r = int(np.ceil(max(abs(n), abs(n + m), abs(m)))) + 2
        tau_fracs = [np.array([1 / 3, 1 / 3]), np.array([2 / 3, 2 / 3])]
        T_mat = np.array([T1[:2], T2[:2]])

        def _enumerate_layer(M_inv, z):
            """Fold lattice points into [0,1)² moiré fractional coords and deduplicate."""
            fracs = []
            for ix in range(-max_r, max_r + 1):
                for iy in range(-max_r, max_r + 1):
                    cell = np.array([ix, iy], dtype=float)
                    for tau in tau_fracs:
                        fs = (cell + tau) @ M_inv
                        fm = fs % 1.0
                        fm[np.abs(fm - 1.0) < tol] = 0.0
                        fracs.append(fm)
            fracs = np.array(fracs)
            _, uid = np.unique(fracs.round(8), axis=0, return_index=True)
            fracs = fracs[uid]
            pos_xy = fracs @ T_mat
            return np.column_stack([pos_xy, np.full(len(fracs), z)])

        layer1 = _enumerate_layer(M2d_inv, 0.0)

        # Layer 2: independent enumeration on the rotated (b1,b2) lattice.
        # In the rotated frame the moiré supercell satisfies
        # T1 = m·b1 + n·b2,  T2 = −n·b1 + (m+n)·b2.
        M2d_top = np.array([[m, n], [-n, m + n]], dtype=float)
        M2d_top_inv = np.linalg.inv(M2d_top)

        if c is None:
            c = 2.0 * d

        layer2 = _enumerate_layer(M2d_top_inv, d)

        import warnings
        for lbl, lyr in (("layer 1", layer1), ("layer 2", layer2)):
            if len(lyr) != 2 * N_cells:
                warnings.warn(
                    f"Expected {2 * N_cells} atoms in {lbl}, found {len(lyr)}. "
                    "Check (n, m) indices."
                )

        all_basis = np.vstack([layer1, layer2])
        lat_vecs = np.array([
            [T1[0], T1[1], 0.0],
            [T2[0], T2[1], 0.0],
            [0.0,   0.0,   c  ],
        ])
        return cls(lat_vecs, all_basis)



    def save(self, filepath="POSCAR", species=None, fmt="vasp"):
        """Export the lattice to a DFT format file.

        filepath: output path (default "POSCAR")
        species:  str  → all atoms share that element label
                  list → one label per basis atom (len must equal n_sites)
                  None → all atoms labelled "X"
        fmt:      export format; only "vasp" (POSCAR) is currently supported
        """
        if species is None:
            labels = ["X"] * self.n_sites
        elif isinstance(species, str):
            labels = [species] * self.n_sites
        else:
            labels = list(species)
            if len(labels) != self.n_sites:
                raise ValueError(
                    f"len(species)={len(labels)} must equal n_sites={self.n_sites}"
                )

        if fmt == "vasp":
            self._write_poscar(filepath, labels)
        else:
            raise ValueError(f"Unknown format '{fmt}'. Supported: 'vasp'")

    def _write_poscar(self, filepath, labels):
        # Collect Cartesian positions per species, preserving insertion order
        groups = {}
        for lbl, pos in zip(labels, self.basis_vecs):
            groups.setdefault(lbl, []).append(pos)
        for lbl in groups:
            groups[lbl].sort(key=lambda p: p[2])

        lines = [
            "Generated by tbforge",
            "  1.0",
        ]
        for vec in self.lat_vecs:
            lines.append(f"  {vec[0]:>20.16f}  {vec[1]:>20.16f}  {vec[2]:>20.16f}")
        lines.append("  " + "  ".join(groups))
        lines.append("  " + "  ".join(str(len(v)) for v in groups.values()))
        lat_inv = np.linalg.inv(self.lat_vecs)
        lines.append("Direct")
        for coords in groups.values():
            for pos in coords:
                f = pos @ lat_inv
                lines.append(
                    f"  {f[0]:>20.16f}  {f[1]:>20.16f}  {f[2]:>20.16f}"
                )

        with open(filepath, "w") as f:
            f.write("\n".join(lines) + "\n")

    def find_kgrid(self, mesh=None, pbc=None) -> np.ndarray:
        if mesh is None:
            mesh = [5, 5, 1]
        if pbc is None:
            pbc = self.bc
        nkx, nky, nkz = mesh
        b = self.bz_vecs
        i, j, k = np.meshgrid(range(nkx), range(nky), range(nkz), indexing='ij')
        kgrid = (
            (i[..., None] / nkx) * pbc[0] * b[0] +
            (j[..., None] / nky) * pbc[1] * b[1] +
            (k[..., None] / nkz) * pbc[2] * b[2]
        )
        return kgrid.reshape(-1, 3)

    def find_kpath(self, kpath_labels=None, kpath_frac=None, n_kpts=120):
        b1, b2, b3 = self.bz_vecs

        if kpath_labels is None or kpath_frac is None:
            angle12 = np.arccos(np.clip(
                np.dot(b1, b2) / (np.linalg.norm(b1) * np.linalg.norm(b2)), -1, 1
            ))
            angle23 = np.arccos(np.clip(
                np.dot(b2, b3) / (np.linalg.norm(b2) * np.linalg.norm(b3)), -1, 1
            ))
            if np.isclose(angle12, np.pi/2, atol=1e-3) and np.isclose(angle23, np.pi/2, atol=1e-3):
                kpath_labels = ["G", "M", "X", "G"]
                kpath_frac = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0], [0, 0, 0]])
            elif np.isclose(angle12, 2*np.pi/3, atol=1e-3):
                kpath_labels = ["G", "M", "K", "G"]
                kpath_frac = np.array([[0, 0, 0], [0.5, 0, 0], [2/3, 1/3, 0], [0, 0, 0]])
            else:
                raise ValueError("Cannot infer default BZ path for this lattice geometry")

        kpath_cart = np.array([p[0]*b1 + p[1]*b2 + p[2]*b3 for p in kpath_frac])
        segment_lengths = np.linalg.norm(np.diff(kpath_cart, axis=0), axis=1)
        total_length = np.sum(segment_lengths)
        nk_list = [max(2, int(round(n_kpts * l / total_length))) for l in segment_lengths]

        # Each segment contributes nk points (excluding its start, including its end)
        kpath = [kpath_cart[0]]
        for i, nk in enumerate(nk_list):
            segment = np.linspace(kpath_cart[i], kpath_cart[i + 1], nk + 1)[1:]
            kpath.extend(segment)

        kpath = np.array(kpath)
        kpath_1d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(kpath, axis=0), axis=1))])

        tick_indices = np.concatenate([[0], np.cumsum(nk_list)])
        tick_locs = kpath_1d[tick_indices].tolist()
        ticks = [tick_locs, kpath_labels]
        return kpath, kpath_1d, ticks


    def find_kpts(self, direction=None, n_kpts=300, half_zone=False, **kwargs):
        """Generate a k-path, dispatching on boundary conditions.

        For bulk lattices (bc=[1,1,1]) delegates to find_kpath() and accepts
        its keyword arguments (kpath_labels, kpath_frac). For ribbon/slab
        geometries (exactly one periodic direction) generates a 1D path along
        that direction over one full BZ period [0, 2π/a].

        Returns the same (kpath, kpath_1d, ticks) format throughout so
        Solver.get_bands() and Plotter.plot_bands() work unchanged.

        Args:
            direction: Reciprocal axis index (0=b1, 1=b2, 2=b3). Ribbon only;
                       auto-detected from bc when omitted. Ignored for bulk.
            n_kpts: Number of k-points.
            half_zone: Ribbon only. If True, sweep [Γ → X] instead of
                       [Γ → X → Γ].
            **kwargs: Passed through to find_kpath() for bulk lattices.

        Returns:
            kpath: (n_kpts, 3) array of Cartesian k-vectors.
            kpath_1d: (n_kpts,) cumulative distances for the plot x-axis.
            ticks: [tick_positions, tick_labels] for Plotter.plot_bands().
        """
        if np.all(self.bc == 1):
            return self.find_kpath(n_kpts=n_kpts, **kwargs)

        if direction is None:
            periodic = np.where(self.bc == 1)[0]
            if len(periodic) == 0:
                raise ValueError("No periodic direction in bc; specify direction explicitly")
            direction = int(periodic[0])

        b = self.bz_vecs[direction]

        fracs = np.linspace(0.0, 0.5 if half_zone else 1.0, n_kpts)
        kpath = np.outer(fracs, b)
        kpath_1d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(kpath, axis=0), axis=1))])
        return kpath, kpath_1d, None
