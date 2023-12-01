# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 20:57:00 2023


@author: Victor
"""

import kwant
import kwant.continuum
import numpy as np
#import matplotlib.pyplot as plt
import scipy
import h5py
import pandas as pd

paulix = np.array([[0,1],[1,0]])
pauliy = np.array([[0,-1j],[1j,0]])
pauliz = np.array([[1,0],[0,-1]])

taux4 = np.kron(paulix, np.eye(2))
tauz4 = np.kron(pauliz, np.eye(2))
sz_tauz = np.kron(pauliz, pauliz)

sx4 = np.kron(np.eye(2), paulix)
sy4 = np.kron(np.eye(2), pauliy)
sz4 = np.kron(np.eye(2), pauliz)

epsi = 1e-3


def Kitaev_syst(aa, LN, LS):
    """ This function builds a finite junction between a Kitaev lattice 
    and a normal metal using kwant"""
    # INPUT
    # aa: float:  lattice constant
    # LN: float:  size of the normal metal
    # LS: float:  size of the Kitaev lattice
    # RETURNS
    # syst: class: kwant system
    
     
        
    def interval_shape(pos):
        x, = pos
        return  -LN <= x <= LS 
    
    def onsite(site, mu_sc, mu_n):
        (x,) = site.pos
        if x >= 0:
            return -mu_sc*pauliz
        if x < 0:
            return -mu_n*pauliz
    
    def hop(site1, site2, t, Delta, V):
        (x,) = site1.pos
        if x > 0:
            return -t*pauliz  - 1j*Delta*pauliy
        elif x < 0:
            return -t*pauliz  
        elif x == 0:
            return -V*pauliz
    
    lat = kwant.lattice.chain(aa,norbs=2)
    syst = kwant.Builder()
    syst[lat.shape(interval_shape, (0, ))] = onsite
    syst[lat.neighbors()] = hop
    syst = syst.finalized()
    
    return syst

def FS_NW_syst(aa,LN,LS):
    """ This function builds a finite junction between a ferromagnetic-superconducting
    SOC nanowire and a normal metal with SOC using kwant"""
    # INPUT
    # aa: float:  lattice constant
    # LN: float:  size of the normal metal
    # LS: float:  size of the SOC nanowire
    # RETURNS
    # syst: class: kwant system

    
    def interval_shape(pos):
        x, = pos
        return  -LN <= x <= LS 

    def onsite(site, t, mu_sc, mu_n, Delta, m0, theta, phi, V):
        (x,) = site.pos
        if x >= 0:
            mx = m0*np.sin(theta*np.pi)*np.cos(phi*np.pi)
            my = m0*np.sin(theta*np.pi)*np.sin(phi*np.pi)
            mz = m0*np.cos(theta*np.pi)
            H_mag = mx*sx4 + my*sy4 + mz*sz4
            H_sc = Delta*taux4 
            H_chem = (2*t-mu_sc)*tauz4 
            return H_chem + H_sc + H_mag
        elif x < -aa:
            return (2*t-mu_n)*tauz4
        elif -1.1*aa < x <-0.9*aa:
            return V*tauz4

    def hop(site1, site2, t, alpha):
        return -t*tauz4 -1j*alpha*sz_tauz

    lat = kwant.lattice.chain(aa,norbs=4)
    syst = kwant.Builder()
    syst[lat.shape(interval_shape, (0, ))] = onsite
    syst[lat.neighbors()] = hop
    syst = syst.finalized()

    return syst

    
def diag_test(sel,spden,syst_sel,arr,kn,pardict,xleft,xright):
    # Sel: the key of the dictionary
    # spden: a string variable with two choices: sparse or dense
    # arr: array of the input variable
    # kn: number of eigenstates in the sparse diagonalization
    
    if syst_sel == 'kitaev':
        syst = Kitaev_syst(aa=0.4,LN=xleft,LS=xright)
    
    elif syst_sel == 'FSNW':
        syst = FS_NW_syst(aa=0.4,LN=xleft,LS=xright)

    else:
        print('Option for system not recognized')
        
    pardict[sel] = arr[0]     
    if spden == 'sparse':
        ham = syst.hamiltonian_submatrix(params=pardict, sparse=True)
        Emtx = scipy.sparse.linalg.eigsh(ham, k=kn, which='SM', return_eigenvectors=False)
    elif spden == 'dense':
        ham = syst.hamiltonian_submatrix(params=pardict, sparse=False)
        #print('m x n:',len(ham[0,::]),len(ham[::,0]))
        Emtx = scipy.linalg.eigh(ham, eigvals_only=True)
    else:
        print('Option not recognized')
    Emtx = np.sort(Emtx)
        
    
    for aa in arr[1::]:
        pardict[sel] = aa
        if spden == 'sparse':        
            ham = syst.hamiltonian_submatrix(params=pardict, sparse=True)
            evals = scipy.sparse.linalg.eigsh(ham, k=kn, which='SM', return_eigenvectors=False)
        elif spden == 'dense':
            ham = syst.hamiltonian_submatrix(params=pardict, sparse=False)
            evals = scipy.linalg.eigh(ham, eigvals_only=True)
        else:
            print('Option not recognized')        
        evals = np.sort(evals)
        Emtx = np.vstack((Emtx,evals))

    nameone = syst_sel+'_EE_'+sel+'_'+str.format('{:.1f}',arr[0])+'_'+str.format('{:.1f}',arr[-1])    

    if sel == 'V':
        parstr = pardict.get('mu_sc')
        nametwo = '_mu_sc'+str.format('{:.1f}',parstr)
    elif sel == 'mu_sc':
        parstr = pardict.get('V')
        nametwo = '_V'+str.format('{:.1f}',parstr)

    namethree = '_xi'+str.format('{:.1f}',-xleft)+'_xf'+str.format('{:.1f}',xright)+'.hdf5'
    
    flname = nameone+nametwo+namethree
    
    with h5py.File(flname, 'w') as f:
        f.create_dataset('array_1', data = arr)
        f.create_dataset('array_2', data = Emtx)
        
    return Emtx

def spect_plot(Ens,arrmu):

    fig, sub = plt.subplots(1, figsize=(5,4))

    nn = len(Ens[0,::])
    for ii in range(nn):
        sub.plot(arrmu,Ens[::,ii])    
    
    sub.set_ylim(-2,2)
    sub.set_xlabel(r'$\mu_{SC}/E_0$')
    sub.set_ylabel(r'$E/E_0$')
    
def main():
    
    A = pd.read_csv('input.dat', header=None)
    
    str_par_sel = A[0][7]
    str_diag_sel = A[0][8]
    str_syst_sel = A[0][9]
    
    B = A[0][0:7].astype(float)
    xl= B[3]
    xr= B[4]
    
    if str_syst_sel == 'kitaev':
        default = dict(t=1,mu_sc=B[5],mu_n=0.0,Delta=2.0, V=B[6])   

        #system = Kitaev_syst(aa=0.4,LN=xl,LS=xr)
    
    elif str_syst_sel == 'FSNW':
        default = dict(t=2.0, alpha=1, Delta=5.0, mu_n=0.0, mu_sc=B[5], V=B[6], 
                       m0=8.0, theta=0.5, phi=0.2)
        
        #system = FS_NW_syst(aa=0.4,LN=xl,LS=xr)

    else:
        print('Option for system not recognized')

    par_arr=np.linspace(B[0],B[1],int(B[2]), endpoint=False)

    Emtx = diag_test(str_par_sel,str_diag_sel,str_syst_sel,par_arr,30,default,xl,xr)
    
    #spect_plot(Emtx,muarr)
    
    
if __name__ == '__main__':
    main()
