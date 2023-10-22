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

tauy = np.array([[0,-1j],[1j,0]])
tauz = np.array([[1,0],[0,-1]])

def Kitaev_syst(aa, LN, LS):
    """ This function using kwant builds a finite junction of a Kitaev lattice 
    and a normal metal """
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
            return -mu_sc*tauz
        if x < 0:
            return -mu_n*tauz
    
    def hop(site1, site2, t, Delta, V):
        (x,) = site1.pos
        if x > 0:
            return -t*tauz  - 1j*Delta*tauy
        elif x < 0:
            return -t*tauz  
        elif x == 0:
            return -V*tauz
    
    lat = kwant.lattice.chain(aa,norbs=2)
    syst = kwant.Builder()
    syst[lat.shape(interval_shape, (0, ))] = onsite
    syst[lat.neighbors()] = hop
    syst = syst.finalized()
    
    return syst

def SOC_NW_syst(alat,Lleft,Lright):

    hamscatt = """
        (4*k_x**2 + varphi(x, x0, sigma, V) - mu(x, mu_L, mu_SC)) * kron(sigma_z, sigma_0)
        - 4 * k_x * kron(sigma_z, sigma_z)
        + Delta(x, Delta0) * kron(sigma_x, sigma_0)
        + m_x(x,m0,theta,phi) * kron(sigma_0, sigma_x)
        + m_y(x,m0,theta,phi) * kron(sigma_0, sigma_y)
        + m_z(x,m0,theta,phi) * kron(sigma_0, sigma_z)
    """
    
    hamiltonian = kwant.continuum.sympify(hamscatt)
    
    syst_template = kwant.continuum.discretize(hamiltonian, grid=alat)
    
    def interval_shape(x_min, x_max):
        def shape(site):
            return x_min <= site.pos[0] <= x_max
    
        return shape

    phy_syst = kwant.Builder()



    phy_syst.fill(
        syst_template,
        shape=interval_shape(Lleft, Lright),
        start=[0],
    )

    return phy_syst.finalized()

def delta_func(x, Delta0):
    """"Function to impose Delta on x>0"""
    if x < 0:
        return 0.0
    elif x >= 0:
        return Delta0
        
def mu_func(x, x0, dx, mu_L, mu_SC):
    """" A tanh-shaped potential to simulate a gradual change in 
    chemical potential."""
    return -(mu_L - mu_SC)*np.tanh((x - x0)/dx)/2 + (mu_L + mu_SC)/2

def mu_sharp(x, mu_L, mu_SC):

    if x >= 0:
        return mu_SC
    else:
        return mu_L

def potential(x, x0, sigma, V):
    """"Gaussian potential centered around x0, with amplitude V and smoothness sigma."""
    return V*np.exp(-(x - x0)**2/(2*sigma**2))

def mx_f(x, m0, theta, phi):
    """" The x component of the magnetization"""
    if x >= 0.0:
        return m0*np.sin(theta*np.pi)*np.cos(phi*np.pi)
    else:
        return 0.0
    
def my_f(x, m0, theta, phi):
    """" The x component of the magnetization"""
    if x >= 0.0:
        return m0*np.sin(theta*np.pi)*np.sin(phi*np.pi)
    else:
        return 0.0
    
def mz_f(x, m0, theta, phi):
    """" The x component of the magnetization"""
    if x >= 0.0:
        return m0*np.cos(theta*np.pi)
    else:
        return 0.0

    
def diag_test(sel,spden,arr,kn,pardict,xleft,xright):
    # Sel: the key of the dictionary
    # spden: a string variable with two choices: sparse or dense
    # arr: array of the input variable
    # kn: number of eigenstates in the sparse diagonalization
    
    #syst = system(alat=0.4,Lleft=xleft,Lright=xright)
    syst = Kitaev_syst(aa=0.4,LN=xleft,LS=xright)
    
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

    nameone = 'EE_'+sel+'_'+str.format('{:.1f}',arr[0])+'_'+str.format('{:.1f}',arr[-1])    

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
    
    B = A[0][0:7].astype(float)
    xl= B[3]
    xr= B[4]
    
#    default = dict(Delta0=20.0, mu_L=20, V=B[4], mu_SC=0.0, m0=40.0, theta=0.5, phi=0.2, 
#           Delta=delta_func, mu=mu_sharp, varphi=potential, m_x=mx_f, m_y=my_f, m_z=mz_f,
#           x0=-0.4, sigma=0.1)

    default = dict(t=1,mu_sc=B[5],mu_n=0.0,Delta=2.0, V=B[6])

    #kwant_spectrum(default,xl,xr)
    
    par_arr=np.linspace(B[0],B[1],int(B[2]), endpoint=False)
    Emtx = diag_test(str_par_sel,str_diag_sel,par_arr,30,default,xl,xr)
    #spect_plot(Emtx,muarr)
    
    
if __name__ == '__main__':
    main()
