import numpy as np
import scipy as sp
import numba as nb
from typing import List, Tuple


class Hopping:
    def __init__(self, sys):
        self.sys = sys

    def find_hops(self, order=1, rz=0., d=1, tol=1e-3):
        if self.sys.is_bulk():
            rlist = self.sys.basisVecs
        else:
            rlist = self.sys.positions 

        range_x = range(-d, d+1) if self.sys.bc[0] else [0]
        range_y = range(-d, d+1) if self.sys.bc[1] else [0]
        range_z = range(-d, d+1) if self.sys.bc[2] else [0]
        
        hopsxy = []
        hopsz = []
        rxy = self.sys.find_neighbor_dist(order)
        for i in range_x:
            for j in range_y:
                for k in range_z:
                    for s in range(len(rlist)):
                        for sp in range(len(rlist)):
                            ri = rlist[s]
                            rj = rlist[sp] + i*self.sys.latVecs[0]\
                                            + j*self.sys.latVecs[1]\
                                            + k*self.sys.latVecs[2]
                            rij = ri-rj
                            dxy = np.linalg.norm(ri[0:2]-rj[0:2])
                            dz = np.linalg.norm(rij)
                            #only retruns hoppings that are rxy distance away
                            
                            if abs(rxy-dxy) < tol and abs(ri[2]-rj[2]) < tol:
                                hopsxy.append([i, j, k, s, sp, dxy, rij[0], rij[1], rij[2]])
                            #returns all hoppings with rz distance
                            if (rz-dz) > 0 and abs(ri[2]-rj[2]) > tol:
                                hopsz.append([i, j, k, s, sp, dz, *rij])                
        return np.array(hopsxy), np.array(hopsz)
    
    def get_hops_nn(self):
        return self.find_hops(order=1)[0]
    
    def get_hops_2nn(self):
        return self.find_hops(order=2)[0]
    
    def get_hopsz(self, rz):
        return self.find_hops(rz=rz)[1]
    
    def get_kmsign(self, full_map=False):
        if self.sys.is_bulk():
            self.rlist = self.sys.basisVecs
        else:
            self.rlist = self.sys.positions 
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
                    if k2 == j and np.all(d1+d2==dij):
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