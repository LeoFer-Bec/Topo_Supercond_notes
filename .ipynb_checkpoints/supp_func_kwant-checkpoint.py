# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 20:01:32 2023

@author: victor
"""

import numpy as np
import kwant.continuum
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize_scalar
from functools import partial
import matplotlib.pyplot as plt

def mx_homog(m0, theta, phi):
    """" The x component of the magnetization"""
    return m0*np.sin(theta*np.pi)*np.cos(phi*np.pi)
    
def my_homog(m0, theta, phi):
    """" The y component of the magnetization"""
    return m0*np.sin(theta*np.pi)*np.sin(phi*np.pi)
    
def mz_homog(m0, theta, phi):
    """" The z component of the magnetization"""
    return m0*np.cos(theta*np.pi)

def K_syst(leftL,rightL, alat, str_hamil, finite):
    """ This function builds the tight binding model of a
    continuous hamiltonian provided as a string variable"""
    # INPUT
    # leftL: float: starting point (x_i) of the lattice
    # rightL: float: final point (x_f) of the lattice
    # alat: float: lattice constant
    # str_hamil: string: the continuous hamiltonian
    # finite: boolean: if True (False) the lattice is finite (infinite)
    # RETURNS
    # syst (infisyst): the tight binding system
    
    def interval_shape(x_min, x_max):
        def shape(site):
            return x_min <= site.pos[0] <= x_max

        return shape
    
    scatt_template = kwant.continuum.discretize(str_hamil, 'x', grid=alat)
    

    if finite:
        syst = kwant.Builder()

        syst.fill(
            scatt_template,
            shape=interval_shape(leftL, rightL),
            start=[leftL],
        )

        return syst.finalized()
    else:
        syst = kwant.Builder(kwant.TranslationalSymmetry([alat]))

        syst.fill(
            scatt_template,
            shape=interval_shape(0, alat),
            start=[0],
        )
        infisyst = kwant.wraparound.wraparound(syst)
        return infisyst.finalized()

def mu_f(x, mu_N, mu_sc):
    """ Two piecewise chemical potential """
    if x >= 0:
        return mu_sc
    else:
        return mu_N
    
def mx_f(x, m0, theta, phi):
    """" The x component of the magnetization"""
    if x >= 0.0:
        return m0*np.sin(theta*np.pi)*np.cos(phi*np.pi)
    else:
        return 0.0
    
def my_f(x, m0, theta, phi):
    """" The y component of the magnetization"""
    if x >= 0.0:
        return m0*np.sin(theta*np.pi)*np.sin(phi*np.pi)
    else:
        return 0.0
    
def mz_f(x, m0, theta, phi):
    """" The z component of the magnetization"""
    if x >= 0.0:
        return m0*np.cos(theta*np.pi)
    else:
        return 0.0

def delta_f(x, Delta0):
    
    if x >= 0:
        return Delta0
    else:
        return 0.0


def V_rect(x, L_b, V0):
    """ Rectangular potential barrier """
    if -L_b <= x < 0: 
        return V0
    else:
        return 0.0
    
def mu_g(x, L, mu_N, mu_1, mu_2):
    """ Three piecewise chemical potential """
    if x < 0:
        return mu_N
    elif 0 <= x <= L:
        return mu_1
    elif x > L:
        return mu_2
    
def mx_g(x, L, m0, theta, phi):
    """" The x component of the magnetization"""
    if 0.0 <= x <= L:
        return m0*np.sin(theta*np.pi)*np.cos(phi*np.pi)
    else:
        return 0.0
    
def my_g(x, L, m0, theta, phi):
    """" The y component of the magnetization"""
    if 0.0 <= x <= L:
        return m0*np.sin(theta*np.pi)*np.sin(phi*np.pi)
    else:
        return 0.0
    
def mz_g(x, L, m0, theta, phi):
    """" The z component of the magnetization"""
    if 0.0 <= x <= L:
        return m0*np.cos(theta*np.pi)
    else:
        return 0.0    
    
def V_g(x, x0, sigma, V0):
    """ Gaussian-shape potential barrier """
    return V0*np.exp(-(x - x0)**2/(2*sigma**2))
    
def diag_solver(syst,sel,val,kn,pardict,vswitch):
    """ This function calculates the kn eigenvalues of the hamiltonian associated
    with a kwant system. The function optionally calculates the eigenstates"""
    # Sel: string: 'single' or a key of the pardict dictionary
    # val: float: value of a key in the pardict dictionary
    # kn: integer: number of eigenstates in the sparse diagonalization
    # vswitch: boolean: if False (True) the function doesn't return (returns) the
    #                   eigenstates
    
    pardict[sel] = val
    ham = syst.hamiltonian_submatrix(params=pardict, sparse=True)

    if vswitch:
    
        evals, evecs = eigsh(ham, k=kn, which='SM', return_eigenvectors=True)        
        return (evals, evecs)

    else:
    
        evals = eigsh(ham, k=kn, which='SM', return_eigenvectors=False)
        return evals
    
def ph_diagram(sub, D0, polar, xlim):
    """ This function plots the diagram of the homogeneous and infinite 
    Lutchyn-Oreg hamiltonian """
    # Sub: Axes object of matplotlib.pyplot: a subplot defined in the main code
    # D0: float: the value of the superconducting gap
    # polar: float: the value of the precessing cone angle
    # xlim: float: the max val along the x axis 'm0' 
    
    #gapped to gapless transition line
    if abs(polar - 0.0) < 1e-7:
        gapless = D0
        NTG_str = ' '
    else:
        gapless = D0/(np.cos(polar*np.pi) + 1e-7)
        NTG_str = 'NTG'
    
    if gapless < xlim:    
        vertline = np.linspace(-xlim,xlim,7)
        sub.plot(gapless*np.ones(len(vertline)),vertline, '--g', lw=2.2)

        #Splitting the interval [0, xlim] in three parts
        left_m0arr = np.linspace(0,0.99*D0,13)
        center_m0arr = np.linspace(D0,gapless,29)         #Constraint leading to potential bug but kept for simplicity
        right_m0arr = np.linspace(gapless,xlim,29)        #dim of center_m0arr & dim of right_m0arr should be equal 
    
        topline_left = xlim*np.ones(len(left_m0arr))
        topline_right = xlim*np.ones(len(center_m0arr))
        #Splitting the transition line beteeen topologically nontrivial and trivial phases into two parts 
        left_muarr = np.sqrt(center_m0arr**2-D0**2*np.ones(len(center_m0arr)))
        right_muarr = np.sqrt(right_m0arr**2-D0**2*np.ones(len(right_m0arr)))

        for ii in range(2):
            sub.plot(center_m0arr,left_muarr*(-1)**ii, '--b', lw=2)
            sub.plot(right_m0arr,right_muarr*(-1)**ii, '--b', lw=2)
    

        sub.fill_between(right_m0arr, right_muarr, topline_right, color='lime', alpha=0.5)
        sub.fill_between(right_m0arr, -right_muarr, -topline_right, color='lime', alpha=0.5)
        sub.fill_between(right_m0arr, -right_muarr, right_muarr, color='orange', alpha=0.5)
        sub.fill_between(center_m0arr, -left_muarr, left_muarr, color='purple', alpha=0.5)
        sub.fill_between(center_m0arr, left_muarr, topline_right, color='cyan', alpha=0.5)
        sub.fill_between(center_m0arr, -left_muarr, -topline_right, color='cyan', alpha=0.5)
        sub.fill_between(left_m0arr, -topline_left, topline_left, color='cyan', alpha=0.5)
        
        sub.text(1.05*gapless,0.0*xlim,'NTGL', fontsize=12, color='black')
        sub.text(1.05*gapless,0.85*xlim,'TGL', fontsize=12, color='black')
        sub.text(0.7*gapless,-0.15*xlim,NTG_str, fontsize=12, color='black')

        
    else:
        left_m0arr = np.linspace(0,0.99*D0,13)
        right_m0arr = np.linspace(D0,xlim,43)
        muarr = np.sqrt(right_m0arr**2-D0**2*np.ones(len(right_m0arr)))
        topline_left = xlim*np.ones(len(left_m0arr))
        topline_right = xlim*np.ones(len(right_m0arr))

        
        for ii in range(2):
            sub.plot(right_m0arr,muarr*(-1)**ii, '--b', lw=2)
            
        sub.fill_between(right_m0arr, -muarr, muarr, color='purple', alpha=0.5)
        sub.fill_between(right_m0arr, muarr, topline_right, color='cyan', alpha=0.5)
        sub.fill_between(right_m0arr, -muarr, -topline_right, color='cyan', alpha=0.5)
        sub.fill_between(left_m0arr, -topline_left, topline_left, color='cyan', alpha=0.5)
        
        sub.text(0.7*xlim,-0.15*xlim,'NTG', fontsize=12, color='black')


    sub.set_xlim(0,xlim)
    sub.set_ylim(-xlim,xlim)
    sub.text(0.5,0.5*xlim,'TG', fontsize=12, color='black')
    
    return sub

def cond_trans(syst, params, Elims, npts):
    """ This function calculates the spectral conductance of a kwant system """
    # INPUT
    # syst: object: tight binding system built by kwant
    # params: dictionary: parameters defining the system
    # Elims: array: energy limits
    # npts: integer: number of points defining the energy array
    #                with even spacing
    # RETURNS
    # Earr: array: energy array
    # Tee: list -> array: the normal reflection amplitude
    # The: list -> array: the Andreev reflection amplitude
    # Nmds: list -> array: the number of propating modes
    
    Earr = np.linspace(Elims[0],Elims[1],npts)
    Tee = []
    The = []
    Nmds = []
    for EE in Earr:
        Smtx = kwant.smatrix(syst, energy=EE, params=params)
        Nmds.append(Smtx.submatrix((0,0),(0,0)).shape[0])
        Tee.append(Smtx.transmission((0,0),(0,0)))
        The.append(Smtx.transmission((0,1),(0,0)))
    
    return Earr, np.array(Tee), np.array(The), np.array(Nmds)

def cond_1D(syst, params, m0arr, EE):
    
    Glist = []
    for m0 in m0arr:
        
        default['m0'] = m0 

        Smtx = kwant.smatrix(syst, energy=EE, params=params)

        Nmds = Smtx.submatrix((0,0),(0,0)).shape[0]
        Tee = Smtx.transmission((0,0),(0,0))
        The = Smtx.transmission((0,1),(0,0))
        Glist.append(Nmds - Tee + The)
    
    return np.array(Glist)


def f_Emin(params, alat, ham_str_homo, kx):
    """ This function computes the energy minimum of a homogeneous 
    hamiltonian """
    # INPUT
    # params: dictionary: parameters defining the system
    # alat: float: the lattice constant
    # ham_str_homo: string: the continuous hamiltonian
    # kx: float: the linear momentum
    # RETURNS
    # float: energy minimum
    
    params['k_x'] = kx
    perlat = K_syst(params['L1'],params['L2'], alat, str_hamil=ham_str_homo, finite=False)
    ham = perlat.hamiltonian_submatrix(params=params, sparse=False)
    Evals = eigh(ham, eigvals_only=True)
    Evals = Evals[Evals>0]
    return np.min(Evals)

def gap(params,alat,ham_str_homo):
    """ This function computes the gap of a homogeneous 
    hamiltonian. It requires the local function f_Emin() and the 
    function minimize_scalar() of scipy.optimize"""
    
    E_f = partial(f_Emin, params, alat, ham_str_homo)
    gone = minimize_scalar(E_f, bounds=(-1,-0.2), method='bounded')
    gtwo = minimize_scalar(E_f, bounds=(-0.2,0.2), method='bounded')
    gthree = minimize_scalar(E_f, bounds=(0.2,1.0), method='bounded')
    minimarr = np.array([gone.fun, gtwo.fun, gthree.fun])
    return np.min(minimarr)
        
def K_junct(info_lat, hamlist, updates, junct, sym_cons):
    """ This function builds the tight binding model of a
    juncttion from continuous hamiltonians provided as string variables.
    The function is a slight modification of the function K_syst()"""
    # INPUT
    # info_lat: list: [lattice spacing, (dimensions of the scattering region)] 
    # ham_list: list: [all homogeneous terms to form the hamiltonian]
    # updates: list: [terms to be updated (e.g homogeneous -> inhomogeneous)]
    # juct: string: NS or NSS
    # sym_cons: 2D array: Symmetry of the hamiltonian used to label scattering states
    # RETURNS
    # syst (infisyst): the tight binding system
    
    alat = info_lat[0]; leftL = info_lat[1]; rightL = info_lat[2] 
    
    def interval_shape(x_min, x_max):
        def shape(site):
            return x_min <= site.pos[0] <= x_max

        return shape

    def shape_lead(s): return s.pos[0] >= 0
    
    if junct == 'NS':
        mu_right_contact = 'mu_sc'
        h_right = hamlist[0]
        for ii in range(1,len(hamlist)-1):
            h_right = h_right+"""+"""+hamlist[ii]
    elif junct == 'NSS':
        mu_right_contact = 'mu_2'
        h_right = hamlist[0]+"""+"""+hamlist[1]
    else:
        print('junction option not recognized')

    h_total = hamlist[0]
    for ii in range(1,len(hamlist)):
        h_total = h_total+"""+"""+hamlist[ii]

    nupdates = int(0.5*len(updates))
    for ii in range(nupdates):
        h_total = h_total.replace(updates[2*ii], updates[2*ii+1])
    
    syst = kwant.Builder()
    scatt_template = kwant.continuum.discretize(h_total, 'x', grid=alat)
    syst.fill(
        scatt_template,
        shape=interval_shape(leftL, rightL),
        start=[leftL],
    )
            
    left_lead = kwant.Builder(kwant.TranslationalSymmetry([-alat]))
    template_left_lead = kwant.continuum.discretize(
        hamlist[0], 'x', grid=alat)
    left_lead.fill(template_left_lead, shape_lead, (0,))
    left_lead.conservation_law = sym_cons
    syst.attach_lead(left_lead)
    
    right_lead = kwant.Builder(kwant.TranslationalSymmetry([alat]))
    template_right_lead = kwant.continuum.discretize(
        h_right, 'x', grid=alat)
    right_lead.fill(template_right_lead.substituted(mu_N=mu_right_contact), shape_lead, (0,))
    syst.attach_lead(right_lead)

    return syst.finalized()

def spin_pumping(latsyst, parameters, par_arr, par_str, En):

    spump_list = []
    for par in par_arr: 
        parameters[par_str] = par
        
        Smatrix = kwant.smatrix(latsyst, energy=En, params=parameters).data


        dS_dphi = np.array([[0,-1j*Smatrix[0,1],0,-1j*Smatrix[0,3]],
                            [1j*Smatrix[1,0],0,1j*Smatrix[1,2],0],
                            [0,-1j*Smatrix[2,1],0,-1j*Smatrix[2,3]],
                            [1j*Smatrix[3,0],0,1j*Smatrix[3,2],0]])


        sz_op = np.diag([1,-1,1,-1]) 
        prod_mtx = dS_dphi @ sz_op @ (np.conj(Smatrix[:4:,:4:].T))
        d_sz = np.trace(prod_mtx)
        spump_list.append(0.5*d_sz.imag)
    
    return np.array(spump_list)