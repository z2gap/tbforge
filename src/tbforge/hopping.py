import numpy as np
import scipy as sp
import numba as nb
from typing import List, Tuple


class Hopping:
    def __init__(self):
        pass

    def find_hops(self, lat, order=1, rz=0., d=1, tol=1e-3) -> Tuple[np.ndarray, np.ndarray]:
        rlist = lat.basisVecs 
        hopsxy = []
        hopsz = []
        rxy = lat.find_neighbor_dist(order)
        for i in range(-d,d+1):
            for j in range(-d,d+1):
                for k in range(-d,d+1):
                    for s in range(len(rlist)):
                        for sp in range(len(rlist)):
                            ri = rlist[s]
                            rj = rlist[sp] + i*lat.latVecs[0]\
                                                + j*lat.latVecs[1]\
                                                + k*lat.latVecs[2]
                            rij = ri-rj
                            dxy = np.linalg.norm(ri[0:2]-rj[0:2])
                            dz = np.linalg.norm(rij)
                            #only retruns hoppings that are rxy distance away
                            if abs(rxy-dxy) < tol and abs(ri[2]-rj[2]) < tol:
                                hopsxy.append([i, j, k, s, sp, dxy, rij[0], rij[1], rij[2]])
                            #returns all hoppings with rz distance
                            if (rz-dz) > 0 and abs(ri[2]-rj[2]) > tol:
                                hopsz.append([i, j, k, s, sp, dz, rij[0], rij[1], rij[2]])
                                
        hopsxy = np.array(hopsxy).reshape(-1,9)
        hopsz = np.array(hopsz).reshape(-1,9)
        return np.array(hopsxy), np.array(hopsz)