import numpy as np
import scipy as sp
import numba as nb
import matplotlib.pyplot as plt 


class Plotter:
    def __init__(self, ax=None):
        self.ax = ax
        if self.ax is None:
            self.ax = plt.gca()

    def plot_lattice(self, lat, s=20, add_bond=False) -> None:
        rlist = lat.basisVecs
        if lat.dim == 1:
            self.ax.scatter(rlist, np.zeros(rlist), c='blue', marker='o', s=s)
        if lat.dim == 2:
            self.ax.scatter(rlist[:,0], rlist[:,1], c='blue', marker='o', s=s)
        if lat.dim == 3:
            self.ax.scatter(rlist[:,0], rlist[:,1], c='blue', marker='o', s=s)
        
        #Not working; needs fix
        if add_bond:
            tree = sp.spatial.KDTree(rlist)
            nn_dist = lat.findNNdist()
            hops = tree.query_ball_point(rlist, r=nn_dist+1e-3)
            for i, jlist in enumerate(hops):
                for j in jlist:
                    ri = rlist[i]
                    rj = rlist[j]
                    self.ax.plot([ri[0], rj[0]], [ri[1], rj[1]], color='gray', zorder=0)


    def plot_bands(self, bands, kpath_1d, ticks=None, color='blue', linewidth=1.5):
        nk, nstate = bands.shape
        for ib in range(nstate):
            self.ax.plot(kpath_1d, bands[:, ib], color=color, linewidth=linewidth)

        if ticks is not None:
            tick_locs, tick_labels = ticks
            self.ax.set_xticks(tick_locs)
            self.ax.set_xticklabels(tick_labels)
            # for pos in tick_locs:
            #     self.ax.axvline(pos, color='k', linestyle='--', linewidth=0.8)

        self.ax.set_xlim(kpath_1d[0], kpath_1d[-1])
        self.ax.set_ylabel('Energy')
        self.ax.grid(True)



    def plot_orb_bands( self, bands, w_orb, kpath_1d, ticks=None, orb_names=None,):
        nk, nstate, norb = w_orb.shape
        if orb_names is None:
            orb_names = [f"orb {i}" for i in range(norb)]
        if len(orb_names) != norb:
            raise ValueError("orb_names length must match number of orbitals in w_orb")

        cmap = plt.get_cmap('tab20b')

        # Scatter bands with orbital weights
        for o in range(norb):
            color = cmap(o / norb)
            for n in range(nstate):
                self.ax.scatter(
                    kpath_1d,
                    bands[:, n],
                    s=50 * w_orb[:, n, o],
                    color=color,
                    linewidths=0,
                )

        # High-symmetry points
        if ticks is not None:
            tick_locs, tick_labels = ticks
            self.ax.set_xticks(tick_locs)
            self.ax.set_xticklabels(tick_labels)
            for pos in tick_locs:
                self.ax.axvline(pos, color='k', linestyle='--', linewidth=0.8)

        self.ax.set_ylabel('Energy')
        self.ax.set_xlim(kpath_1d[0], kpath_1d[-1])
        self.ax.grid(True)

        # Legend
        handles = [
            plt.Line2D(
                [], [],
                color=cmap(o / norb),
                marker='o',
                linestyle='',
                label=orb_names[o]
            ) for o in range(norb)
        ]
        self.ax.legend(
            handles=handles,
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
        )

    def plot_particle_bands(self, bands, w_el, kpath_1d, ticks):
        nk, nstate = bands.shape

        # Plot bands with scatter sized by electron weight and color-coded by weight
        sc = None
        for n in range(nstate):
            sc = self.ax.scatter(kpath_1d, bands[:, n], 
                                s=3,
                                c=w_el[:, n],
                                cmap='gray')

        if ticks is not None:
            tick_locs, tick_labels = ticks
            self.ax.set_xticks(tick_locs)
            self.ax.set_xticklabels(tick_labels)
            # for pos in tick_locs:
            #     self.ax.axvline(pos, color='k', linestyle='--', linewidth=0.8)

        self.ax.set_xlim(kpath_1d[0], kpath_1d[-1])
        self.ax.set_ylabel('Energy')
        self.ax.grid(True)

        # Add colorbar on top
        fig = self.ax.figure
        cbar = fig.colorbar(sc, ticks=[], fraction=0.03, ax=self.ax, orientation='horizontal', pad=0.15)
        cbar.set_label('electron/hole')


    def plot_bands_1d(self, bands, kpath_1d, ticks=None, labels=None, color='blue', linewidth=1.5):
        """
        Plot energy bands along 1D k-path.
        """
        nk, nstate = bands.shape

        if len(kpath_1d) != nk:
            raise ValueError(f"kpath_1d length {len(kpath_1d)} != number of k-points {nk}")

        # Plot each band
        for ib in range(nstate):
            self.ax.plot(kpath_1d, bands[:, ib], color=color, linewidth=linewidth)

        # Plot high-symmetry points
        if ticks is not None:
            tick_positions = list(ticks.values())
            #tick_labels = list(ticks.keys())
            self.ax.set_xticks(tick_positions)
            #self.ax.set_xticklabels(tick_labels)
            for pos in tick_positions:
                self.ax.axvline(pos, color='k', linestyle='--', linewidth=0.8)

        if labels:
            self.ax.set_xticklabels(labels)
        self.ax.set_xlim(kpath_1d[0], kpath_1d[-1])
        self.ax.grid(zorder=0)


    def plot_dos(self, dos, ylim=[0,2]):
        self.ax.set_ylim(ylim[0], ylim[1])
        self.ax.set_xlim(dos[:,0][0], dos[:,0][-1])
        self.ax.set_ylabel('DOS')
        self.ax.set_xlabel('Eneregy')
        self.ax.plot(dos[:,0], dos[:,1], c='blue')
        self.ax.grid(zorder=0)

    