#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 20:30:52 2018

@author: Michaelray
"""
import os
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.optimize import minimize

def ell2alm(list_of_ells):
    """Parameters
    list_of_ells: Give a 1-D vector representing power at each ell.

    Returns
    A list of a_lm values which correspond directly to the ell values given
    in the input array list_of_ells"""
    Nout = int(np.sum(1.0 * np.arange(len(list_of_ells)) + 1.0))
    alm = np.zeros(Nout)
    counter = 0
    for i in range(len(list_of_ells)):
        alm[counter:(counter)+i+1] = list_of_ells[i]
        counter = counter + i + 1
    return alm

def signal_diag(inputTT, NSIDE = 512):
    """Parameters
    inputTT: Give a 1-D vector that represents the TT power spectrum. Only power at
        ell of 200 will be counted from inputTT because of the way this function
        selects for certain ells.
    NSIDE: Give a number representing the n-side of the map associated with the
        input TT power spectrum.

    Returns
    A 1-D array that represents the diagonal of the signal matrix in spherical
    harmonic space. This functions selects for power at ell of 200 because of the
    way the entries in the matrix are replaced."""
    ELLMAX = 3*NSIDE - 1
    sinput = hp.synfast(inputTT, NSIDE)
    alm = hp.map2alm(sinput)
    alm1 = np.abs(alm)
    alm2 = alm1**2
    Sdiag1 = np.ones(ELLMAX+1) * 1e-10
    Sdiag = ell2alm(Sdiag1)
    Sdiag[np.where(alm2 > 10e-6)] = 1.0
    return Sdiag

def iterate_pcg(Ndiag, i_q_u, B_pspec_cov):
    """
    Parameters
    
    Ndiag: Give a vector that is size NPIX*2 which is the noise covariance matrix
        in pixel space. This is usually two copies of the inverse mask on top of
        each other
    i_q_u: Give a 3 by NPIX array which represents measured data. 
    B_pspec_cov: Give a vector of length ELLMAX which will be used to create
        the signal covariance matrix.

    Returns
    A filtered map that is filtered using the preconditioned conjugate gradient
    technique. I have used the method and equations outlined in the Papez, 
    Grigori, Stompor paper and algorithm 10.2.1 from Golub/Van Loan 1996.
    This also returns a list of r_k, x_k, and p_k.
    """
    NPIX = len(i_q_u[0])
    NSIDE = int(np.sqrt(NPIX/12))
    ELLMAX = 3*NSIDE - 1
    ells = np.arange(0, ELLMAX + 1)
    data = np.concatenate((i_q_u[1],i_q_u[2]), axis=0)
    NSPH = int(np.sum(1.0 * np.arange(ELLMAX+1) + 1.0))
    Tdiag_pix = np.min(Ndiag)*np.ones(NPIX*2)
    Tdiag_sph = np.min(Ndiag)*np.ones(NSPH * 2) * (4.0 * np.pi) / NPIX
    Ntilde_inv = (Ndiag - 0.999*Tdiag_pix)**(-1)
    FWHM = 24.2
    sigma_rad = (FWHM/(np.sqrt(8*np.log(2))))*(np.pi/(60*180))
    B_ell_squared = [np.e**((-1)*(index**2)*(sigma_rad**2)) for index in range(ELLMAX)]
    B_pspec_cov[:30]=1e-10
    B_pspec_cov_beamed = B_pspec_cov*B_ell_squared
    B_cov = hp.almxfl(np.ones(NSPH), B_pspec_cov_beamed)
    S_B = np.concatenate((np.zeros(NSPH), B_cov), axis = 0)
    B_pseudo = np.concatenate((np.zeros(NSPH), (B_cov)**(-1)), axis = 0)
    B_pseudo[np.where(np.isnan(B_pseudo))] = 0.0
    C_mat = B_pseudo + Tdiag_sph**(-1)
    Ndiag_sph = hp.map2alm(Ndiag[:NPIX], lmax = ELLMAX)
    D_mat = Tdiag_pix**(-1) * (Ntilde_inv + Tdiag_pix**(-1))**(-1) * Tdiag_pix**(-1)
    D_mat_q = D_mat[:NPIX]
    D_mat_u = D_mat[NPIX:]
    D_mat_sph = np.concatenate((hp.map2alm(D_mat_q, lmax=ELLMAX), hp.map2alm(D_mat_u, lmax=ELLMAX)), axis=0)
    A_mat = np.ones(NPIX*2) - np.concatenate((hp.alm2map(C_mat[:NSPH]**(-1) * D_mat_sph[:NSPH], NSIDE), hp.alm2map(C_mat[NSPH:]**(-1) * D_mat_sph[NSPH:], NSIDE)), axis=0)
    b_vec = np.concatenate((hp.alm2map(C_mat[:NSPH] * Ndiag_sph, NSIDE), hp.alm2map(C_mat[NSPH:] * Ndiag_sph, NSIDE)), axis=0) * data
    k = 0
    list_rk = np.empty([2, NPIX*2])
    list_xk = np.empty([2, NPIX*2])
    list_pk = np.empty([2, NPIX*2])
    x_k = np.zeros(NPIX*2)
    r_k = b_vec - A_mat * x_k
    list_xk[1,:] = x_k
    list_rk[1,:] = r_k
    while np.any(np.abs(r_k) >= 1e-5):
        print('k = ', k)
        k += 1
        if k == 1:
            p_k = r_k
            list_pk[1, :] = p_k
        else:
            beta_k = np.sum(list_rk[1,:] * list_rk[1,:] / (list_rk[0,:] * list_rk[0,:]))
            p_k = list_rk[1,:] + beta_k * list_pk[1,:]
            print("beta_k = ", beta_k)
            print('p_k = ', p_k)
            print('list_rk[1,:] after pk = ', list_rk[1,:])
            print('list_pk[1,:] after pk before update = ', list_pk[1,:])
            list_pk[0,:] = list_pk[1,:]
            list_pk[1,:] = p_k
        alpha_k = np.sum(list_rk[1,:] * list_rk[1,:] / (p_k * A_mat * p_k))
        print('alpha_k = ', alpha_k)
        x_k = list_xk[1,:] + alpha_k * p_k
        print("x_k = ", x_k)
        r_k = list_rk[1,:] - alpha_k * A_mat * p_k
        print("r_k = ", r_k)
        list_xk[0,:] = list_xk[1,:]
        list_xk[1,:] = x_k
        list_rk[0,:] = list_rk[1,:]
        list_rk[1,:] = r_k
    final_sig = x_k
    return final_sig, list_rk, list_xk, list_pk

def iterate_withbeam(Ndiag, i_q_u, B_pspec_cov, lambdaone = 100, eta = 7/10):
    """
    Parameters

    Ndiag: Give a 1-D vector that represents the diagonal of the noise covariance
        matrix. In simulations, this was two copies of the inverse of the mask
        on top of each other. These served to mask the Q and U maps from the data
        vector independently. Ndiag should be size NPIX*2
    i_q_u: This is a 3 by NPIX size matrix with the first column of the matrix
        being an I map, the second being a Q map and the third being a U map.
        This should be measured data.
    B_pspec_cov: Give a 1-D vector that is size ELLMAX and represents the theoretical
        B power spectrum of the data.This is used to calculate the B covariance
        matrix.
    lambdaone: Give the starting value for lambda in the iterative
        algorithm. The default value is 100.
    eta: Give the factor that lambda is reduce by on each iteration. The default
        value is 7/10

    Returns

    A 1-D vector with size NSPH * 2. The first NSPH entries in this will be the
    alm's for the E mode. The last NSPH entries in this will be the alm's for the
    B mode. Also returns a list of chi-squared statistics calculated on each iteration.
    Also returns a list of iqu maps calculated on each iteration. Also returns a list
    of lambda values that represent the lambda used on each iteration.
    """
    NPIX = len(i_q_u[0])
    NSIDE = int(np.sqrt(NPIX/12))
    ELLMAX = 3*NSIDE - 1
    ells = np.arange(ELLMAX + 1)
    data = np.concatenate((i_q_u[1],i_q_u[2]), axis=0)
    NSPH = int(np.sum(1.0 * np.arange(ELLMAX+1) + 1.0))
    Tdiag_pix = np.min(Ndiag)*np.ones(NPIX*2)
    Tdiag_sph = np.min(Ndiag)*np.ones(NSPH * 2) * (4.0 * np.pi) / NPIX
    Ntilde_inv = (Ndiag - 0.999*Tdiag_pix)**(-1)
    FWHM = 24.2
    sigma_rad = (FWHM/(np.sqrt(8*np.log(2))))*(np.pi/(60*180))
    B_ell_squared = np.e**((-1)*(ells**2)*(sigma_rad**2))
    B_pspec_cov[:30]=1e-10
    B_pspec_cov_beamed = B_pspec_cov*B_ell_squared
    B_cov = hp.almxfl(np.ones(NSPH), B_pspec_cov_beamed)
    #B_cov = hp.almxfl(np.ones(NSPH), B_ell_squared)
    S_B = np.concatenate((np.zeros(NSPH), B_cov), axis = 0)
    #S_B = np.concatenate((np.zeros(NSPH), B_cov), axis=0)
    B_pseudo = np.concatenate((np.zeros(NSPH), (B_cov)**(-1)), axis = 0)
    B_pseudo[np.where(np.isnan(B_pseudo))] = 0.0
    s = np.zeros(NPIX*2)
    tpix = data
    bspec_list = []
    lambda_list = []
    i = 0
    lambdaone = 1300
    tpix = (Ntilde_inv * data + (lambdaone * Tdiag_pix)**(-1) * s) * (Ntilde_inv + (lambdaone * Tdiag_pix)**(-1))**(-1)
    tpix_q = tpix[:NPIX]
    tpix_u = tpix[NPIX:]
    t_e_b = hp.map2alm((i_q_u[0], tpix_q, tpix_u), lmax=ELLMAX, pol=True)
    tsph_e = t_e_b[1]
    tsph_b = t_e_b[2]
    tsph = np.concatenate((tsph_e, tsph_b), axis = 0)
    ssph = (B_pseudo + (lambdaone * Tdiag_sph)**(-1))**(-1) * (lambdaone * Tdiag_sph)**(-1) * tsph
    ssph_e = ssph[:NSPH]
    ssph_b = ssph[NSPH:]
    weiner_iqu = hp.alm2map((t_e_b[0], ssph_e, ssph_b), NSIDE, lmax=ELLMAX, pol=True)
    out_bspec_cl = hp.anafast(weiner_iqu, lmax = ELLMAX, pol=True)[2]
    s = np.concatenate((weiner_iqu[1], weiner_iqu[2]), axis = 0)
    bspec_list.append(out_bspec_cl * ells * (ells+1) / (2 * np.pi))
    lambda_list.append(lambdaone)
    print("lambdaone = ", lambdaone)
    i += 1
    lambdaone = 100
    while lambdaone >= 1:
        tpix = (Ntilde_inv * data + (lambdaone * Tdiag_pix)**(-1) * s) * (Ntilde_inv + (lambdaone * Tdiag_pix)**(-1))**(-1)
        tpix_q = tpix[:NPIX]
        tpix_u = tpix[NPIX:]
        t_e_b = hp.map2alm((i_q_u[0], tpix_q, tpix_u), lmax=ELLMAX, pol=True)
        tsph_e = t_e_b[1]
        tsph_b = t_e_b[2]
        tsph = np.concatenate((tsph_e, tsph_b), axis = 0)
        ssph = (B_pseudo + (lambdaone * Tdiag_sph)**(-1))**(-1) * (lambdaone * Tdiag_sph)**(-1) * tsph
        ssph_e = ssph[:NSPH]
        ssph_b = ssph[NSPH:]
        weiner_iqu = hp.alm2map((t_e_b[0], ssph_e, ssph_b), NSIDE, lmax=ELLMAX, pol=True)
        out_bspec_cl = hp.anafast(weiner_iqu, lmax = ELLMAX, pol = True)[2]
        s = np.concatenate((weiner_iqu[1], weiner_iqu[2]), axis = 0)
        bspec_list.append(out_bspec_cl * ells * (ells+1) / (2 * np.pi))
        lambdaone = lambdaone*eta
        print("lambdaone = ", lambdaone)
        lambda_list.append(lambdaone)
        i += 1
    lambdaone = 1
    i = 0
    while i < 5:
        tpix = (Ntilde_inv * data + (lambdaone * Tdiag_pix)**(-1) * s) * (Ntilde_inv + (lambdaone * Tdiag_pix)**(-1))**(-1)
        tpix_q = tpix[:NPIX]
        tpix_u = tpix[NPIX:]
        t_e_b = hp.map2alm((i_q_u[0], tpix_q, tpix_u), lmax=ELLMAX, pol=True)
        tsph_e = t_e_b[1]
        tsph_b = t_e_b[2]
        tsph = np.concatenate((tsph_e, tsph_b), axis = 0)
        ssph = (B_pseudo + (lambdaone * Tdiag_sph)**(-1))**(-1) * (lambdaone * Tdiag_sph)**(-1) * tsph
        ssph_e = ssph[:NSPH]
        ssph_b = ssph[NSPH:]
        weiner_iqu = hp.alm2map((t_e_b[0], ssph_e, ssph_b), NSIDE, lmax=ELLMAX, pol=True)
        s = np.concatenate((weiner_iqu[1], weiner_iqu[2]), axis = 0)
        out_bspec_cl = hp.anafast(weiner_iqu, lmax = ELLMAX, pol=True)[2]
        bspec_list.append(out_bspec_cl * ells * (ells+1) / (2 * np.pi))
        lambda_list.append(lambdaone)
        i += 1
        print("i = ", i)
    s_final = ssph
    return s_final, bspec_list, lambda_list

def iterate_fixedlam(Ndiag, i_q_u, B_pspec_cov, lambdaone = 100, NSIMS = 10):
    """                                                                                        Parameters                                                                                 Ndiag: Give a 1-D vector that represents the diagonal of the noise covariance
        matrix. In simulations, this was two copies of the inverse of the mask
        on top of each other. These served to mask the Q and U maps from the data
        vector independently. Ndiag should be size NPIX*2                
    i_q_u: This is a 3 by NPIX size matrix with the first column of the matrix
        being an I map, the second being a Q map and the third being a U map.
        This should be measured data.          
    B_pspec_cov: Give a 1-D vector that is size ELLMAX and represents the theoretical
        B power spectrum of the data.This is used to calculate the B covariance
        matrix.                                                                                lambdaone: Give the value for lambda in the iterative algorithm. The default 
        value is 100.    
    NSIMS: Give an integer that is the number of iterations that will run.
    
    Returns                                                                                                                                                       
    A 1-D vector with size NSPH * 2. The first NSPH entries in this will be the alm's for 
    the E mode. The last NSPH entries in this will be the alm's for the B mode. Also 
    returns a list of chi-squared statistics calculated on each iteration."""
    NPIX = len(i_q_u[0])
    NSIDE = int(np.sqrt(NPIX/12))
    ELLMAX = 3*NSIDE - 1
    data = np.concatenate((i_q_u[1],i_q_u[2]), axis=0)
    NSPH = int(np.sum(1.0 * np.arange(ELLMAX+1) + 1.0))
    Tdiag_pix = np.min(Ndiag)*np.ones(NPIX*2)
    Tdiag_sph = np.min(Ndiag)*np.ones(NSPH * 2) * (4.0 * np.pi) / NPIX
    Ntilde_inv = (Ndiag - 0.999*Tdiag_pix)**(-1)
    FWHM = 24.2
    sigma_rad = (FWHM/(np.sqrt(8*np.log(2))))*(np.pi/(60*180))
    B_ell_squared = [np.e**((-1)*(index**2)*(sigma_rad**2)) for index in range(ELLMAX)]
    B_pspec_cov[:30]=1e-10
    B_pspec_cov_beamed = B_pspec_cov*B_ell_squared
    B_cov = hp.almxfl(np.ones(NSPH), B_pspec_cov_beamed)
    B_cov = hp.almxfl(np.ones(NSPH), B_pspec_cov_beamed)
    S_B = np.concatenate((np.zeros(NSPH), B_cov), axis = 0)
    B_pseudo = np.concatenate((np.zeros(NSPH), (B_cov)**(-1)), axis = 0)
    B_pseudo[np.where(np.isnan(B_pseudo))] = 0.0
    s = np.zeros(NPIX*2)
    tpix = data
    chi_squared_list = []
    i = 0
    while i <= NSIMS:
        tpix = (Ntilde_inv * data + (lambdaone * Tdiag_pix)**(-1) * s) * (Ntilde_inv + (lambdaone * Tdiag_pix)**(-1))**(-1)
        tpix_q = tpix[:NPIX]
        tpix_u = tpix[NPIX:]
        t_e_b = hp.map2alm((i_q_u[0], tpix_q, tpix_u), lmax=ELLMAX, pol=True)
        tsph_e = t_e_b[1]
        tsph_b = t_e_b[2]
        tsph = np.concatenate((tsph_e, tsph_b), axis = 0)
        ssph = (B_pseudo + (lambdaone * Tdiag_sph)**(-1))**(-1) * (lambdaone * Tdiag_sph)**(-1) * tsph
        ssph_e = ssph[:NSPH]
        ssph_b = ssph[NSPH:]
        weiner_iqu = hp.alm2map((t_e_b[0], ssph_e, ssph_b), NSIDE, lmax=ELLMAX, pol=True)
        s = np.concatenate((weiner_iqu[1], weiner_iqu[2]), axis = 0)
        chi_squared = chi2(B_pseudo, Ndiag, Tdiag_pix, data, s, ssph, lambdaone)
        chi_squared_list.append(chi_squared)
        i += 1
    s_final = S_B * B_pseudo * ssph
    return s_final, chi_squared_list

def iterate_declam(Ndiag, i_q_u, B_pspec_cov, min_lam, max_lam, stepsize):
    """
    Parameters

    Ndiag: Give a 1-D vector that represents the diagonal of the noise covariance 
        matrix. In simulations, this was two copies of the inverse of the mask 
        on top of each other. These served to mask the Q and U maps from the data 
        vector independently. Ndiag should be size NPIX*2.
    i_q_u: This is a 3 by NPIX size matrix with the first column of the matrix 
        being an I map, the second being a Q map and the third being a U map. 
        This should be measured data.
    B_pspec_cov: Give a 1-D vector that is size ELLMAX and represents the theoretical
        B power spectrum of the data.This is used to calculate the B covariance matrix.
    min_lam: Give the minimum value of lambda
    max_lam: Give the maximum value of lambda
    stepsize: Give the size of step you want between successive values of lambda.
        For example, max_lam = 5, min_lam = 1, stepsize = 2 would give lambda values
        of 1, 3, 5.

    Returns

    A 1-D vector with size NSPH * 2. The first NSPH entries in this will be the
    alm's for the E mode. The last NSPH entries in this will be the alm's for the
    B mode. Also returns a list of chi-squared statistics calculated on each iteration.
    """
    NPIX = len(i_q_u[0])
    NSIDE = int(np.sqrt(NPIX/12))
    ELLMAX = 3*NSIDE - 1
    data = np.concatenate((i_q_u[1],i_q_u[2]), axis=0)
    NSPH = int(np.sum(1.0 * np.arange(ELLMAX+1) + 1.0))
    Tdiag_pix = np.min(Ndiag)*np.ones(NPIX*2)
    Tdiag_sph = np.min(Ndiag)*np.ones(NSPH * 2) * (4.0 * np.pi) / NPIX
    Ntilde_inv = (Ndiag - 0.999*Tdiag_pix)**(-1)
    FWHM = 24.2
    sigma_rad = (FWHM/(np.sqrt(8*np.log(2))))*(np.pi/(60*180))
    B_ell_squared = [np.e**((-1)*(index**2)*(sigma_rad**2)) for index in range(ELLMAX)]
    B_pspec_cov[:30]=1e-10
    B_pspec_cov_beamed = B_pspec_cov*B_ell_squared
    B_cov = hp.almxfl(np.ones(NSPH), B_pspec_cov_beamed)
    S_B = np.concatenate((np.zeros(NSPH), B_cov), axis = 0)
    B_pseudo = np.concatenate((np.zeros(NSPH), (B_cov)**(-1)), axis = 0)
    B_pseudo[np.where(np.isnan(B_pseudo))] = 0.0
    s = np.zeros(NPIX*2)
    tpix = data
    lambda_list = []
    chi_squared_list = []
    i = 0
    lambdaone = max_lam
    while i < 10:
        tpix = (Ntilde_inv * data + (lambdaone * Tdiag_pix)**(-1) * s) * (Ntilde_inv + (lambdaone * Tdiag_pix)**(-1))**(-1)
        tpix_q = tpix[:NPIX]
        tpix_u = tpix[NPIX:]
        t_e_b = hp.map2alm((i_q_u[0], tpix_q, tpix_u), lmax=ELLMAX, pol=True)
        tsph_e = t_e_b[1]
        tsph_b = t_e_b[2]
        tsph = np.concatenate((tsph_e, tsph_b), axis = 0)
        ssph = (B_pseudo + (lambdaone * Tdiag_sph)**(-1))**(-1) * (lambdaone * Tdiag_sph)**(-1) * tsph
        ssph_e = ssph[:NSPH]
        ssph_b = ssph[NSPH:]
        weiner_iqu = hp.alm2map((t_e_b[0], ssph_e, ssph_b), NSIDE, lmax=ELLMAX, pol=True)
        s = np.concatenate((weiner_iqu[1], weiner_iqu[2]), axis = 0)
        chi_squared = chi2(B_pseudo, Ndiag, Tdiag_pix, data, s, ssph, lambdaone)
        chi_squared_list.append(chi_squared)
        lambda_list.append(lambdaone)
        i += 1
        if i==9:
            i = 0
            lambdaone = lambdaone-stepsize
            continue
        if lambdaone < min_lam:
            break
    s_final = S_B * B_pseudo * ssph
    return s_final, chi_squared_list, lambda_list


def iterate_untilchi_steps(Ndiag, i_q_u, B_pspec_cov, min_lam, max_lam, stepsize, min_chi = 10**6):
    """
    Parameters                                                                                                                                                                                                                                                                                                                                                                                                                          
    Ndiag: Give a 1-D vector that represents the diagonal of the noise covariance                                                                              
        matrix. In simulations, this was two copies of the inverse of the mask                                                                                
        on top of each other. These served to mask the Q and U maps from the data                                                                             
        vector independently. Ndiag should be size NPIX*2                                                                                                     
    i_q_u: This is a 3 by NPIX size matrix with the first column of the matrix                                                                             
        being an I map, the second being a Q map and the third being a U map.                                                                                  
        This should be measured data.                                                                                                                         
    B_pspec_cov: Give a 1-D vector that is size ELLMAX and represents the theoretical                                                                          
        B power spectrum of the data.This is used to calculate the B covariance                                                                             
        matrix.                                                                                                                                                
    min_lam: Give the minimum value of lambda                                                                                                                 
    max_lam: Give the maximum value of lambda                                                                                                                 
    stepsize: Give the size of step you want between successive values of lambda.                                                                             
        For example, max_lam = 5, min_lam = 1, stepsize = 2 would give lambda values                                                                          
        of 1, 3, 5.                                                                                                                                           
    min_chi: Give the value at which the filter will lower the value of lambda when
        chi_squared * lambda gets lower than min_chi. Once lambda * chi_squared goes
        min_chi, the filter will lower the value of lambda.

    Returns                                                                                                                                                                                                               
    A 1-D vector with size NSPH * 2. The first NSPH entries in this will be the                                                                                
    alm's for the E mode. The last NSPH entries in this will be the alm's for the                                                                              
    B mode. Also returns a list of chi-squared statistics calculated on each iteration."""
    NPIX = len(i_q_u[0])
    NSIDE = int(np.sqrt(NPIX/12))
    ELLMAX = 3*NSIDE - 1
    data = np.concatenate((i_q_u[1],i_q_u[2]), axis=0)
    NSPH = int(np.sum(1.0 * np.arange(ELLMAX+1) + 1.0))
    Tdiag_pix = np.min(Ndiag)*np.ones(NPIX*2)
    Tdiag_sph = np.min(Ndiag)*np.ones(NSPH * 2) * (4.0 * np.pi) / NPIX
    Ntilde_inv = (Ndiag - 0.999*Tdiag_pix)**(-1)
    FWHM = 24.2
    sigma_rad = (FWHM/(np.sqrt(8*np.log(2))))*(np.pi/(60*180))
    B_ell_squared = [np.e**((-1)*(index**2)*(sigma_rad**2)) for index in range(ELLMAX)]
    B_pspec_cov[:30]=1e-10
    B_pspec_cov_beamed = B_pspec_cov*B_ell_squared
    B_cov = hp.almxfl(np.ones(NSPH), B_pspec_cov_beamed)
    B_cov = hp.almxfl(np.ones(NSPH), B_pspec_cov_beamed)
    S_B = np.concatenate((np.zeros(NSPH), B_cov), axis = 0)
    B_pseudo = np.concatenate((np.zeros(NSPH), (B_cov)**(-1)), axis = 0)
    B_pseudo[np.where(np.isnan(B_pseudo))] = 0.0
    s = np.zeros(NPIX*2)
    tpix = data
    lambda_list = []
    chi_squared_list = []
    iqu_list = []
    chi_squared = min_chi*2
    for lambdaone in np.flip(np.arange(min_lam, max_lam, stepsize)):
        tpix = (Ntilde_inv * data + (lambdaone * Tdiag_pix)**(-1) * s) * (Ntilde_inv + (lambdaone * Tdiag_pix)**(-1))**(-1)
        tpix_q = tpix[:NPIX]
        tpix_u = tpix[NPIX:]
        t_e_b = hp.map2alm((i_q_u[0], tpix_q, tpix_u), lmax=ELLMAX, pol=True)
        tsph_e = t_e_b[1]
        tsph_b = t_e_b[2]
        tsph = np.concatenate((tsph_e, tsph_b), axis = 0)
        ssph = (B_pseudo + (lambdaone * Tdiag_sph)**(-1))**(-1) * (lambdaone * Tdiag_sph)**(-1) * tsph
        ssph_e = ssph[:NSPH]
        ssph_b = ssph[NSPH:]
        weiner_iqu = hp.alm2map((t_e_b[0], ssph_e, ssph_b), NSIDE, lmax=ELLMAX, pol=True)
        s = np.concatenate((weiner_iqu[1], weiner_iqu[2]), axis = 0)
        chi_squared = chi2(B_pseudo, Ndiag, Tdiag_pix, data, s, ssph, lambdaone)
        chi_squared_list.append(chi_squared)
        iqu_list.append(weiner_iqu)
        lambda_list.append(lambdaone)
        while lambdaone*chi_squared > min_chi:
            tpix = (Ntilde_inv * data + (lambdaone * Tdiag_pix)**(-1) * s) * (Ntilde_inv + (lambdaone * Tdiag_pix)**(-1))**(-1)
            tpix_q = tpix[:NPIX]
            tpix_u = tpix[NPIX:]
            t_e_b = hp.map2alm((i_q_u[0], tpix_q, tpix_u), lmax=ELLMAX, pol=True)
            tsph_e = t_e_b[1]
            tsph_b = t_e_b[2]
            tsph = np.concatenate((tsph_e, tsph_b), axis = 0)
            ssph = (B_pseudo + (lambdaone * Tdiag_sph)**(-1))**(-1) * (lambdaone * Tdiag_sph)**(-1) * tsph
            ssph_e = ssph[:NSPH]
            ssph_b = ssph[NSPH:]
            weiner_iqu = hp.alm2map((t_e_b[0], ssph_e, ssph_b), NSIDE, lmax=ELLMAX, pol=True)
            s = np.concatenate((weiner_iqu[1], weiner_iqu[2]), axis = 0)
            chi_squared = chi2(B_pseudo, Ndiag, Tdiag_pix, data, s, ssph, lambdaone)
            chi_squared_list.append(chi_squared)
            iqu_list.append(weiner_iqu)
            lambda_list.append(lambdaone)
    i = 0
    lambdaone = 1
    while i < 5:
        tpix = (Ntilde_inv * data + (lambdaone * Tdiag_pix)**(-1) * s) * (Ntilde_inv + (lambdaone * Tdiag_pix)**(-1))**(-1)
        tpix_q = tpix[:NPIX]
        tpix_u = tpix[NPIX:]
        t_e_b = hp.map2alm((i_q_u[0], tpix_q, tpix_u), lmax=ELLMAX, pol=True)
        tsph_e = t_e_b[1]
        tsph_b = t_e_b[2]
        tsph = np.concatenate((tsph_e, tsph_b), axis = 0)
        ssph = (B_pseudo + (lambdaone * Tdiag_sph)**(-1))**(-1) * (lambdaone * Tdiag_sph)**(-1) * tsph
        ssph_e = ssph[:NSPH]
        ssph_b = ssph[NSPH:]
        weiner_iqu = hp.alm2map((t_e_b[0], ssph_e, ssph_b), NSIDE, lmax=ELLMAX, pol=True)
        s = np.concatenate((weiner_iqu[1], weiner_iqu[2]), axis = 0)
        chi_squared = chi2(B_pseudo, Ndiag, Tdiag_pix, data, s, ssph, lambdaone)
        chi_squared_list.append(chi_squared)
        iqu_list.append(weiner_iqu)
        lambda_list.append(lambdaone)
        i += 1
    s_final = S_B * B_pseudo * ssph
    return s_final, chi_squared_list, lambda_list, iqu_list


def iterate_chidiff_steps(Ndiag, i_q_u, B_pspec_cov, min_lam, max_lam, stepsize, init_chi = 2*10**7, chi_diff = 3500):
    """                                                                                                                                                          
    Parameters                                                                                                                                                                                                                                                                                                                                                                                               
    Ndiag: Give a 1-D vector that represents the diagonal of the noise covariance                                                                             
        matrix. In simulations, this was two copies of the inverse of the mask                                                                                
        on top of each other. These served to mask the Q and U maps from the data                                                                              
        vector independently. Ndiag should be size NPIX*2                                                                                                     
    i_q_u: This is a 3 by NPIX size matrix with the first column of the matrix                                                                                 
        being an I map, the second being a Q map and the third being a U map.                                                                                  
        This should be measured data.                                                                                                                         
    B_pspec_cov: Give a 1-D vector that is size ELLMAX and represents the theoretical                                                                        
        B power spectrum of the data.This is used to calculate the B covariance                                                                            
        matrix.                                                                                                                                                
    min_lam: Give the minimum value of lambda                                                                                                                
    max_lam: Give the maximum value of lambda                                                                                                                 
    stepsize: Give the size of step you want between successive values of lambda.                                                                              
        For example, max_lam = 5, min_lam = 1, stepsize = 2 would give lambda values                                                                           
        of 1, 3, 5.                                                                                                                                           
    init_chi: Give the initial value of chi_squared. This value doesn't matter very much
        as long as it is large enough such that the next chi_squared value is not within
        chi_diff of init_chi. If init_chi is closer than chi_diff to the next value
        of chi_squared, then the filter will break out of the loop and change lambda before
        it should have. Default value is 2*10**7.
    chi_diff: Give the difference between successive chi_squared values that indicates the 
        chi_squared has plateaued. When this difference between chi_squared values is hit, the
        filter will break out of the loop and change lambda. Default value is 3500.
                                                                                                                                                              
    Returns                                                                                                                                                
                                                                                                                                                              
    A 1-D vector with size NSPH * 2. The first NSPH entries in this will be the                                                                              
    alm's for the E mode. The last NSPH entries in this will be the alm's for the                                                                            
    B mode. Also returns a list of chi-squared statistics calculated on each iteration.
    Also returns a list of the difference in successive values of chi squared on each 
    iteration.
    """
    NPIX = len(i_q_u[0])
    NSIDE = int(np.sqrt(NPIX/12))
    ELLMAX = 3*NSIDE - 1
    data = np.concatenate((i_q_u[1],i_q_u[2]), axis=0)
    NSPH = int(np.sum(1.0 * np.arange(ELLMAX+1) + 1.0))
    Tdiag_pix = np.min(Ndiag)*np.ones(NPIX*2)
    Tdiag_sph = np.min(Ndiag)*np.ones(NSPH * 2) * (4.0 * np.pi) / NPIX
    Ntilde_inv = (Ndiag - 0.999*Tdiag_pix)**(-1)
    FWHM = 24.2
    sigma_rad = (FWHM/(np.sqrt(8*np.log(2))))*(np.pi/(60*180))
    B_ell_squared = [np.e**((-1)*(index**2)*(sigma_rad**2)) for index in range(ELLMAX)]
    B_pspec_cov[:30]=1e-10
    B_pspec_cov_beamed = B_pspec_cov*B_ell_squared
    B_cov = hp.almxfl(np.ones(NSPH), B_pspec_cov_beamed)
    B_cov = hp.almxfl(np.ones(NSPH), B_pspec_cov_beamed)
    S_B = np.concatenate((np.zeros(NSPH), B_cov), axis = 0)
    B_pseudo = np.concatenate((np.zeros(NSPH), (B_cov)**(-1)), axis = 0)
    B_pseudo[np.where(np.isnan(B_pseudo))] = 0.0
    s = np.zeros(NPIX*2)
    tpix = data
    lambda_list = []
    chi_squared_list = []
    iqu_list = []
    chi_squared = init_chi*2
    chi_squared_list.append(chi_squared)
    i = 0
    for lambdaone in np.flip(np.arange(min_lam, max_lam, stepsize)):
        tpix = (Ntilde_inv * data + (lambdaone * Tdiag_pix)**(-1) * s) * (Ntilde_inv + (lambdaone * Tdiag_pix)**(-1))**(-1)
        tpix_q = tpix[:NPIX]
        tpix_u = tpix[NPIX:]
        t_e_b = hp.map2alm((i_q_u[0], tpix_q, tpix_u), lmax=ELLMAX, pol=True)
        tsph_e = t_e_b[1]
        tsph_b = t_e_b[2]
        tsph = np.concatenate((tsph_e, tsph_b), axis = 0)
        ssph = (B_pseudo + (lambdaone * Tdiag_sph)**(-1))**(-1) * (lambdaone * Tdiag_sph)**(-1) * tsph
        ssph_e = ssph[:NSPH]
        ssph_b = ssph[NSPH:]
        weiner_iqu = hp.alm2map((t_e_b[0], ssph_e, ssph_b), NSIDE, lmax=ELLMAX, pol=True)
        s = np.concatenate((weiner_iqu[1], weiner_iqu[2]), axis = 0)
        iqu_list.append(weiner_iqu)
        chi_squared = chi2(B_pseudo, Ndiag, Tdiag_pix, data, s, ssph, lambdaone)
        chi_squared_list.append(chi_squared)
        lambda_list.append(lambdaone)
        i += 1
        while np.abs((chi_squared_list[i-1] - chi_squared_list[i])) > chi_diff:
            tpix = (Ntilde_inv * data + (lambdaone * Tdiag_pix)**(-1) * s) * (Ntilde_inv + (lambdaone * Tdiag_pix)**(-1))**(-1)
            tpix_q = tpix[:NPIX]
            tpix_u = tpix[NPIX:]
            t_e_b = hp.map2alm((i_q_u[0], tpix_q, tpix_u), lmax=ELLMAX, pol=True)
            tsph_e = t_e_b[1]
            tsph_b = t_e_b[2]
            tsph = np.concatenate((tsph_e, tsph_b), axis = 0)
            ssph = (B_pseudo + (lambdaone * Tdiag_sph)**(-1))**(-1) * (lambdaone * Tdiag_sph)**(-1) * tsph
            ssph_e = ssph[:NSPH]
            ssph_b = ssph[NSPH:]
            weiner_iqu = hp.alm2map((t_e_b[0], ssph_e, ssph_b), NSIDE, lmax=ELLMAX, pol=True)
            s = np.concatenate((weiner_iqu[1], weiner_iqu[2]), axis = 0)
            chi_squared = chi2(B_pseudo, Ndiag, Tdiag_pix, data, s, ssph, lambdaone)
            chi_squared_list.append(chi_squared)
            iqu_list.append(weiner_iqu)
            lambda_list.append(lambdaone)
            i += 1
    s_final = S_B * B_pseudo * ssph
    return s_final, chi_squared_list, lambda_list, iqu_list



def iterate_untilchi_eta(Ndiag, i_q_u, B_pspec_cov, max_lam, eta, min_chi = 10**6):
    """                                                                                                                                                        
    Parameters                                                                                                                                                
                                                                                                                                                           
    Ndiag: Give a 1-D vector that represents the diagonal of the noise covariance                                                                          
        matrix. In simulations, this was two copies of the inverse of the mask                                                                               
        on top of each other. These served to mask the Q and U maps from the data                                                                           
        vector independently. Ndiag should be size NPIX*2                                                                                                      
    i_q_u: This is a 3 by NPIX size matrix with the first column of the matrix                                                                                
        being an I map, the second being a Q map and the third being a U map.                                                                                 
        This should be measured data.                                                                                                                         
    B_pspec_cov: Give a 1-D vector that is size ELLMAX and represents the theoretical                                                                         
        B power spectrum of the data.This is used to calculate the B covariance                                                                              
        matrix.                                                                                                                                              
    max_lam: Give the starting value of lambda                                                                                                              
    eta: Give the value by which lambda is multiplied by each time lambda gets 
        altered.
    min_chi: Give a number which tells the filter when to decrease the value of lambda.
        When lambda times chi_squared goes below min_chi, the filter will lower the value
        of lambda, and then repeat the same process.
                                                                                                                                                             
    Returns                                                                                                                                          
                                                                                                                                                              
    A 1-D vector with size NSPH * 2. The first NSPH entries in this will be the                                                                             
    alm's for the E mode. The last NSPH entries in this will be the alm's for the                                                                             
    B mode. Also returns a list of chi-squared statistics calculated on each iteration."""
    NPIX = len(i_q_u[0])
    NSIDE = int(np.sqrt(NPIX/12))
    ELLMAX = 3*NSIDE - 1
    data = np.concatenate((i_q_u[1],i_q_u[2]), axis=0)
    NSPH = int(np.sum(1.0 * np.arange(ELLMAX+1) + 1.0))
    Tdiag_pix = np.min(Ndiag)*np.ones(NPIX*2)
    Tdiag_sph = np.min(Ndiag)*np.ones(NSPH * 2) * (4.0 * np.pi) / NPIX
    Ntilde_inv = (Ndiag - 0.999*Tdiag_pix)**(-1)
    FWHM = 24.2
    sigma_rad = (FWHM/(np.sqrt(8*np.log(2))))*(np.pi/(60*180))
    B_ell_squared = [np.e**((-1)*(index**2)*(sigma_rad**2)) for index in range(ELLMAX)]
    B_pspec_cov[:30]=1e-10
    B_pspec_cov_beamed = B_pspec_cov*B_ell_squared
    B_cov = hp.almxfl(np.ones(NSPH), B_pspec_cov_beamed)
    B_cov = hp.almxfl(np.ones(NSPH), B_pspec_cov_beamed)
    S_B = np.concatenate((np.zeros(NSPH), B_cov), axis = 0)
    B_pseudo = np.concatenate((np.zeros(NSPH), (B_cov)**(-1)), axis = 0)
    B_pseudo[np.where(np.isnan(B_pseudo))] = 0.0
    s = np.zeros(NPIX*2)
    tpix = data
    lambda_list = []
    chi_squared_list = []
    iqu_list = []
    chi_squared = min_chi*2
    lam = max_lam
    lambda_list.append(lam)
    while lam >= 1:
        lambda_list.append(lam)
        lam = lam*eta
    for lambdaone in lambda_list:
        tpix = (Ntilde_inv * data + (lambdaone * Tdiag_pix)**(-1) * s) * (Ntilde_inv + (lambdaone * Tdiag_pix)**(-1))**(-1)
        tpix_q = tpix[:NPIX]
        tpix_u = tpix[NPIX:]
        t_e_b = hp.map2alm((i_q_u[0], tpix_q, tpix_u), lmax=ELLMAX, pol=True)
        tsph_e = t_e_b[1]
        tsph_b = t_e_b[2]
        tsph = np.concatenate((tsph_e, tsph_b), axis = 0)
        ssph = (B_pseudo + (lambdaone * Tdiag_sph)**(-1))**(-1) * (lambdaone * Tdiag_sph)**(-1) * tsph
        ssph_e = ssph[:NSPH]
        ssph_b = ssph[NSPH:]
        weiner_iqu = hp.alm2map((t_e_b[0], ssph_e, ssph_b), NSIDE, lmax=ELLMAX, pol=True)
        s = np.concatenate((weiner_iqu[1], weiner_iqu[2]), axis = 0)
        chi_squared = chi2(B_pseudo, Ndiag, Tdiag_pix, data, s, ssph, lambdaone)
        chi_squared_list.append(chi_squared)
        iqu_list.append(weiner_iqu)
        while lambdaone*chi_squared > min_chi:
            tpix = (Ntilde_inv * data + (lambdaone * Tdiag_pix)**(-1) * s) * (Ntilde_inv + (lambdaone * Tdiag_pix)**(-1))**(-1)
            tpix_q = tpix[:NPIX]
            tpix_u = tpix[NPIX:]
            t_e_b = hp.map2alm((i_q_u[0], tpix_q, tpix_u), lmax=ELLMAX, pol=True)
            tsph_e = t_e_b[1]
            tsph_b = t_e_b[2]
            tsph = np.concatenate((tsph_e, tsph_b), axis = 0)
            ssph = (B_pseudo + (lambdaone * Tdiag_sph)**(-1))**(-1) * (lambdaone * Tdiag_sph)**(-1) * tsph
            ssph_e = ssph[:NSPH]
            ssph_b = ssph[NSPH:]
            weiner_iqu = hp.alm2map((t_e_b[0], ssph_e, ssph_b), NSIDE, lmax=ELLMAX, pol=True)
            s = np.concatenate((weiner_iqu[1], weiner_iqu[2]), axis = 0)
            chi_squared = chi2(B_pseudo, Ndiag, Tdiag_pix, data, s, ssph, lambdaone)
            chi_squared_list.append(chi_squared)
            iqu_list.append(weiner_iqu)
    i = 0
    lambdaone = 1
    while i < 10:
        tpix = (Ntilde_inv * data + (lambdaone * Tdiag_pix)**(-1) * s) * (Ntilde_inv + (lambdaone * Tdiag_pix)**(-1))**(-1)
        tpix_q = tpix[:NPIX]
        tpix_u = tpix[NPIX:]
        t_e_b = hp.map2alm((i_q_u[0], tpix_q, tpix_u), lmax=ELLMAX, pol=True)
        tsph_e = t_e_b[1]
        tsph_b = t_e_b[2]
        tsph = np.concatenate((tsph_e, tsph_b), axis = 0)
        ssph = (B_pseudo + (lambdaone * Tdiag_sph)**(-1))**(-1) * (lambdaone * Tdiag_sph)**(-1) * tsph
        ssph_e = ssph[:NSPH]
        ssph_b = ssph[NSPH:]
        weiner_iqu = hp.alm2map((t_e_b[0], ssph_e, ssph_b), NSIDE, lmax=ELLMAX, pol=True)
        s = np.concatenate((weiner_iqu[1], weiner_iqu[2]), axis = 0)
        chi_squared = chi2(B_pseudo, Ndiag, Tdiag_pix, data, s, ssph, lambdaone)
        chi_squared_list.append(chi_squared)
        iqu_list.append(weiner_iqu)
        lambda_list.append(lambdaone)
        i += 1
    s_final = S_B * B_pseudo * ssph
    return s_final, chi_squared_list, lambda_list, iqu_list


def iterate_cooling(Ndiag, i_q_u, B_pspec_cov, chi_min = 1.3*10**5):
    """                                                                                                                                                        
    Parameters                                                                                                                                                
                                                                                                                                                              
    Ndiag: Give a 1-D vector that represents the diagonal of the noise covariance                                                                            
        matrix. In simulations, this was two copies of the inverse of the mask                                                                               
        on top of each other. These served to mask the Q and U maps from the data                                                                             
        vector independently. Ndiag should be size NPIX*2                                                                                                     
    i_q_u: This is a 3 by NPIX size matrix with the first column of the matrix                                                                               
        being an I map, the second being a Q map and the third being a U map.                                                                                 
        This should be measured data.                                                                                                                         
    B_pspec_cov: Give a 1-D vector that is size ELLMAX and represents the theoretical                                                                         
        B power spectrum of the data.This is used to calculate the B covariance                                                                               
        matrix.                                                                                                                                                
    chi_min: Give a value that is the chi squared minimum which represents b in the 
        Elsner and Wandelt paper. This has to be a guess to start off with. Default 
        value is 10**5.
                                                                                                                                                              
    Returns                                                                                                                                                   
                                                                                                                                                               
    A 1-D vector with size NSPH * 2. The first NSPH entries in this will be the                                                                               
    alm's for the E mode. The last NSPH entries in this will be the alm's for the                                                                              
    B mode. Also returns a list of chi-squared statistics calculated on each iteration.                                                                        
    Also returns a list of the difference in successive values of chi squared on each                                                                         
    iteration. Also returns a list of BB power spectra in C_ell on each iteration.                                                                                            
    """
    NPIX = len(i_q_u[0])
    NSIDE = int(np.sqrt(NPIX/12))
    ELLMAX = 3*NSIDE - 1
    ells = np.arange(0,ELLMAX + 1)
    data = np.concatenate((i_q_u[1],i_q_u[2]), axis=0)
    NSPH = int(np.sum(1.0 * np.arange(ELLMAX+1) + 1.0))
    Tdiag_pix = np.min(Ndiag)*np.ones(NPIX*2)
    Tdiag_sph = np.min(Ndiag)*np.ones(NSPH * 2) * (4.0 * np.pi) / NPIX
    Ntilde_inv = (Ndiag - 0.999*Tdiag_pix)**(-1)
    FWHM = 24.2
    sigma_rad = (FWHM/(np.sqrt(8*np.log(2))))*(np.pi/(60*180))
    B_ell_squared = [np.e**((-1)*(index**2)*(sigma_rad**2)) for index in range(ELLMAX)]
    B_pspec_cov[:30]=1e-10
    B_pspec_cov_beamed = B_pspec_cov*B_ell_squared
    B_cov = hp.almxfl(np.ones(NSPH), B_pspec_cov_beamed)
    B_cov = hp.almxfl(np.ones(NSPH), B_pspec_cov_beamed)
    S_B = np.concatenate((np.zeros(NSPH), B_cov), axis = 0)
    B_pseudo = np.concatenate((np.zeros(NSPH), (B_cov)**(-1)), axis = 0)
    B_pseudo[np.where(np.isnan(B_pseudo))] = 0.0
    s = np.zeros(NPIX*2)
    tpix = data
    lambda_list = []
    chi_squared_list = []
    bspec_list = []
    i = 0
    iterfun_init = lambda lam: np.abs(chi2ofzero(Ndiag, Tdiag_pix, data, 0, lam) - chi_min)
    minfunc1 = minimize(iterfun_init, 1000, method = 'Nelder-Mead').x
    lambdaone = minfunc1[0]
    print("lambdaone = ", lambdaone)
    lambda_list.append(lambdaone)
    tpix = (Ntilde_inv * data + (lambdaone * Tdiag_pix)**(-1) * s) * (Ntilde_inv + (lambdaone * Tdiag_pix)**(-1))**(-1)
    tpix_q = tpix[:NPIX]
    tpix_u = tpix[NPIX:]
    t_e_b = hp.map2alm((i_q_u[0], tpix_q, tpix_u), lmax=ELLMAX, pol=True)
    tsph_e = t_e_b[1]
    tsph_b = t_e_b[2]
    tsph = np.concatenate((tsph_e, tsph_b), axis = 0)
    ssph = (B_pseudo + (lambdaone * Tdiag_sph)**(-1))**(-1) * (lambdaone * Tdiag_sph)**(-1) * tsph
    ssph_e = ssph[:NSPH]
    ssph_b = ssph[NSPH:]
    weiner_iqu = hp.alm2map((t_e_b[0], ssph_e, ssph_b), NSIDE, lmax=ELLMAX, pol=True)
    out_bspec_cl = hp.alm2cl(ssph_b, lmax = ELLMAX)
    s = np.concatenate((weiner_iqu[1], weiner_iqu[2]), axis = 0)
    chi_squared = chi2(B_pseudo, Ndiag, Tdiag_pix, data, s, ssph, lambdaone)
    chi_squared_list.append(chi_squared)
    bspec_list.append(out_bspec_cl * ells * (ells+1) / (2 * np.pi))
    i += 1
    iterfun_final = lambda lam: np.abs(chi2(B_pseudo, Ndiag, Tdiag_pix, data, s, ssph, lam) - chi_min)
    minfunc2 = minimize(iterfun_final, lambdaone, method = 'Nelder-Mead').x
    lambdaone = minfunc2[0]
    print("lambdaone = ", lambdaone)
    lambda_list.append(lambdaone)
    print("chi_squared (should be always close to chi_min) = ", chi2(B_pseudo, Ndiag, Tdiag_pix, data, s, ssph, lambdaone))
    while lambdaone > 1:
        tpix = (Ntilde_inv * data + (lambdaone * Tdiag_pix)**(-1) * s) * (Ntilde_inv + (lambdaone * Tdiag_pix)**(-1))**(-1)
        tpix_q = tpix[:NPIX]
        tpix_u = tpix[NPIX:]
        t_e_b = hp.map2alm((i_q_u[0], tpix_q, tpix_u), lmax=ELLMAX, pol=True)
        tsph_e = t_e_b[1]
        tsph_b = t_e_b[2]
        tsph = np.concatenate((tsph_e, tsph_b), axis = 0)
        ssph = (B_pseudo + (lambdaone * Tdiag_sph)**(-1))**(-1) * (lambdaone * Tdiag_sph)**(-1) * tsph
        ssph_e = ssph[:NSPH]
        ssph_b = ssph[NSPH:]
        weiner_iqu = hp.alm2map((t_e_b[0], ssph_e, ssph_b), NSIDE, lmax=ELLMAX, pol=True)
        out_bspec_cl = hp.alm2cl(ssph_b, lmax = ELLMAX)
        s = np.concatenate((weiner_iqu[1], weiner_iqu[2]), axis = 0)
        chi_squared = chi2(B_pseudo, Ndiag, Tdiag_pix, data, s, ssph, lambdaone)
        bspec_list.append(out_bspec_cl * ells * (ells+1) / (2 * np.pi))
        chi_squared_list.append(chi_squared)
        iterfun_loop = lambda lam: np.abs(chi2(B_pseudo, Ndiag, Tdiag_pix, data, s, ssph, lam) - chi_min)
        minfunc2 = minimize(iterfun_loop, lambdaone, method = 'Nelder-Mead').x
        lambdaone = minfunc2[0]
        lambda_list.append(lambdaone)
        print("lambdaone (in loop) = ", lambdaone)
        print("chi_squared (should be always close to chi_min) = ", chi2(B_pseudo, Ndiag, Tdiag_pix, data, s, ssph, lambdaone))
        if i == 0:
            print("i = ", i)
        elif i < 3:
            i += 1
            print("i = ", i)
        elif lambdaone > 2.1:
            i += 1
        else:
            min_chi_lam2p1 = 1*10**5
            chi_min = min_chi_lam2p1 + np.sum((lambdaone - 1) * min(Ndiag) * ssph * B_pseudo**2 * ssph)
            print("new chi_min = ", chi_min)
            i = 0
    i = 1
    delt_chi_squared = 100
    while delt_chi_squared > 10:
        print("delt_chi_squared = ", delt_chi_squared)
        lambdaone = 1
        tpix = (Ntilde_inv * data + (lambdaone * Tdiag_pix)**(-1) * s) * (Ntilde_inv + (lambdaone * Tdiag_pix)**(-1))**(-1)
        tpix_q = tpix[:NPIX]
        tpix_u = tpix[NPIX:]
        t_e_b = hp.map2alm((i_q_u[0], tpix_q, tpix_u), lmax=ELLMAX, pol=True)
        tsph_e = t_e_b[1]
        tsph_b = t_e_b[2]
        tsph = np.concatenate((tsph_e, tsph_b), axis = 0)
        ssph = (B_pseudo + (lambdaone * Tdiag_sph)**(-1))**(-1) * (lambdaone * Tdiag_sph)**(-1) * tsph
        ssph_e = ssph[:NSPH]
        ssph_b = ssph[NSPH:]
        weiner_iqu = hp.alm2map((t_e_b[0], ssph_e, ssph_b), NSIDE, lmax=ELLMAX, pol=True)
        out_bspec_cl = hp.alm2cl(ssph_b, lmax = ELLMAX)
        s = np.concatenate((weiner_iqu[1], weiner_iqu[2]), axis = 0)
        chi_squared = chi2(B_pseudo, Ndiag, Tdiag_pix, data, s, ssph, lambdaone)
        chi_squared_list.append(chi_squared)
        bspec_list.append(out_bspec_cl * ells * (ells+1) / (2 * np.pi))
        delt_chi_squared = np.abs(chi_squared_list[i] - chi_squared_list[i-1])
        print("lambdaone = ", lambdaone)
        print("i = ", i)
        i += 1
    s_final = S_B * B_pseudo * ssph
    return s_final, chi_squared_list, lambda_list, bspec_list

def chi2(S_inv, Ndiag, Tdiag_pix, data, x_pix, x_sph, lam):
    return np.sum(np.conj(x_sph) * S_inv * x_sph) + np.sum(np.conj((data - x_pix)) * (Ndiag + (lam - 1) * Tdiag_pix)**(-1) * (data - x_pix))

def chi2ofzero(Ndiag, Tdiag_pix, data, x_pix, lam):
    return np.sum(np.conj((data - x_pix)) * (Ndiag + (lam - 1) * Tdiag_pix)**(-1) * (data - x_pix))

def iterate(Ndiag, i_q_u, B_pspec_cov, lambdaone = 100, eta = 7/10):
    """                                  
    Parameters                                                            

    Ndiag: Give a 1-D vector that represents the diagonal of the noise covariance
        matrix. In simulations, this was two copies of the inverse of the mask
        on top of each other. These served to mask the Q and U maps from the data
        vector independently. Ndiag should be size NPIX*2
    i_q_u: This is a 3 by NPIX size matrix with the first column of the matrix
        being an I map, the second being a Q map and the third being a U map.          
        This should be measured data.
    B_pspec_cov: Give a 1-D vector that is size ELLMAX and represents the theoretical
        B power spectrum of the data.This is used to calculate the B covariance        
        matrix.
    lambdaone: Give the starting value for lambda in the iterative   
        algorithm. The default value is 100.
    eta: Give the factor that lambda is reduce by on each iteration. The default
        value is 7/10                                                                                                                                 
    Returns

    A 1-D vector with size NSPH * 2. The first NSPH entries in this will be the
    alm's for the E mode. The last NSPH entries in this will be the alm's for the
    B mode."""
    NPIX = len(i_q_u[0])
    NSIDE = int(np.sqrt(NPIX/12))
    ELLMAX = 3*NSIDE - 1
    data = np.concatenate((i_q_u[1],i_q_u[2]), axis=0)
    NSPH = int(np.sum(1.0 * np.arange(ELLMAX+1) + 1.0))
    Tdiag_pix = np.min(Ndiag)*np.ones(NPIX*2)
    Tdiag_sph = np.min(Ndiag)*np.ones(NSPH * 2) * (4.0 * np.pi) / NPIX
    Ntilde_inv = (Ndiag - 0.999*Tdiag_pix)**(-1)
    B_cov = hp.almxfl(np.ones(NSPH), B_pspec_cov)
    S_B = np.concatenate((np.zeros(NSPH), B_cov), axis = 0)
    B_pseudo = np.concatenate((np.zeros(NSPH), (B_cov)**(-1)), axis = 0)
    B_pseudo[np.where(np.isnan(B_pseudo))] = 0.0
    s = np.zeros(NPIX*2)
    tpix = data
    while lambdaone >= 1:
        tpix = (Ntilde_inv * data + (lambdaone * Tdiag_pix)**(-1) * s) * (Ntilde_inv + (lambdaone * Tdiag_pix)**(-1))**(-1)
        tpix_q = tpix[:NPIX]
        tpix_u = tpix[NPIX:]
        t_e_b = hp.map2alm((i_q_u[0], tpix_q, tpix_u), lmax=ELLMAX, pol=True)
        tsph_e = t_e_b[1]
        tsph_b = t_e_b[2]
        tsph = np.concatenate((tsph_e, tsph_b), axis = 0)
        ssph = (B_pseudo + (lambdaone * Tdiag_sph)**(-1))**(-1) * (lambdaone * Tdiag_sph)**(-1) * tsph
        ssph_e = ssph[:NSPH]
        ssph_b = ssph[NSPH:]
        weiner_iqu = hp.alm2map((t_e_b[0], ssph_e, ssph_b), NSIDE, lmax=ELLMAX, pol=True)
        s = np.concatenate((weiner_iqu[1], weiner_iqu[2]), axis = 0)
        lambdaone = lambdaone*eta
    lambdaone = 1
    tpix = (Ntilde_inv * data + (lambdaone * Tdiag_pix)**(-1) * s) * (Ntilde_inv + (lambdaone * Tdiag_pix)**(-1))**(-1)
    tpix_q = tpix[:NPIX]
    tpix_u = tpix[NPIX:]
    t_e_b = hp.map2alm((i_q_u[0], tpix_q, tpix_u), lmax=ELLMAX, pol=True)
    tsph_e = t_e_b[1]
    tsph_b = t_e_b[2]
    tsph = np.concatenate((tsph_e, tsph_b), axis = 0)
    ssph = (B_pseudo + (lambdaone * Tdiag_sph)**(-1))**(-1) * (lambdaone * Tdiag_sph)**(-1) * tsph
    ssph_e = ssph[:NSPH]
    ssph_b = ssph[NSPH:]
    weiner_iqu = hp.alm2map((t_e_b[0], ssph_e, ssph_b), NSIDE, lmax=ELLMAX, pol=True)
    s = np.concatenate((weiner_iqu[1], weiner_iqu[2]), axis = 0)
    s_final = S_B * B_pseudo * ssph
    return s_final


def clean_b(iqu, noise_bias, e_b_leakage, input_sup_BB, output_sup_BB, mask, B_pspec_cov, NSIDE, lambda1 = 100, iterateeta = 7/10):
    """Parameters
    iqu: Give a 3 by NPIX size matrix that represents the I map, Q map, and U map
        of your measured data.
    noise_bias: Give a 1-D vector that is size ELLMAX = 3*NSIDE - 1 and a function
        of ell. Alternatively, give a multidimensional vector of size N by ELLMAX where
        N is the number of simulations run in order to obtain the noise bias.
        This represents the noise bias in the data as a function of ell.
        This vector will be calculated through many simulations using random signal.
    e_b_leakage: Give a 1-D vector that is size ELLMAX = 3*NSIDE - 1 and is a function
        of ell. Alternatively, give a multidimensional vector of size N by ELLMAX where
        N is the number of simulations run in order to obtain the eb leakage.
        This represents the leakage from E to B modes that result from looking
        at a finite portion of the sky. This vector will be calculated through
        many simulations using E only signal.
    input_sup_BB: Give a vector of size ELLMAX = 3*NSIDE - 1 and is a function
        of ell. Alternatively, give a multidimensional matrix of size N by ELLMAX where
        N is the number of simulations run in order to obtain the suppression factors.
        This represents the input signal for a given simulation run.
    output_sup_BB: Give a matrix of size N by ELLMAX where N is the number of simulations
        run in order to obtain suppression factors. This is a list of the BB spectrum
        after the iterate function was run on spectra with power in E modes only. This
        combined with the input_sup_BB will be used to calculated suppression factors.
    B_pspec_cov: Give a 1-D vector that is length ELLMAX and represents the theoretical
        BB power spectrum for the data. This is used to calculate the B covariance
        matrix.
    NSIDE: Give a number that is representative of the n-side of the map associated
        with the alm's given for T_alm, E_alm, and B_alm
    lambda1: Give a number that is the starting number for lambda in the iterate
        function
    iterateeta: Give a number that is the factor by which lambda is reduced on each
        iteration in the iterate function

    Returns
    A cleaned BB power spectrum. This function corrects for noise bias, e to b leakage,
    and signal suppression in an uncorrected map.

    Also returns the BB power spectrum of the input iqu's.

    Also returns the uncleaned BB power spectrum.
    """
    C_ell = []
    iqupspeclist = []
    C_elltwid = []
    NSIDE = int(NSIDE)
    ELLMAX = 3*NSIDE - 1
    NPIX = 12*NSIDE**2
    NSPH = int(np.sum(1.0 * np.arange(ELLMAX+1) + 1.0))
    iqupspecs = hp.anafast(iqu, lmax = ELLMAX, pol = True)
    eb_ctwid = iterate(mask, iqu, B_pspec_cov, lambdaone = lambda1, eta = iterateeta)
    ctwid = hp.alm2cl(eb_ctwid[NSPH:], lmax = ELLMAX)
    C_elltwid.append(ctwid)
    iqupspeclist.append(iqupspecs[2])
    noise = np.array(noise_bias)
    leakage = np.array(e_b_leakage)
    avg_b_pspec = np.array(output_sup_BB[:ELLMAX+1])
    avg_input_b_pspec = np.array(input_sup_BB[:ELLMAX+1])
    print('ELLMAX', ELLMAX)
    print('Size of avg_b_pspec', np.shape(avg_b_pspec))
    print('Size of avg_input_b_pspec', np.shape(avg_input_b_pspec))
    sig_sup = avg_b_pspec / avg_input_b_pspec
    sig_sup = np.nan_to_num(sig_sup)
    sig_sup[np.where(sig_sup==0)]=1e-15
    #for i in range(ELLMAX):
    print('Size of ctwid',np.shape(ctwid))
    print('Size of noise',np.shape(noise))
    print('Size of leakage',np.shape(leakage))
    print("Size of sig_sup",np.shape(sig_sup))
    C_ell.append((ctwid - noise - leakage)/sig_sup)
    #C_ell.append((second[2][ELLMAX-1] - noise[ELLMAX-1] - leakage[ELLMAX-1])/sig_sup[ELLMAX-1])
    return C_ell, iqupspeclist, C_elltwid

def calculate_noise_bias(mask, NSIMS, B_pspec_cov):
    """Parameters
    mask: Give a 1-D vector that is the mask which will be used in the calculating
        of the noise bias. This should be size NPIX or the number of pixels that are
        generated in the map you put together.
    NSIMS: Give a number which is the number of simulations you want to run when
        calculating the noise bias. The higher the number of simulations you run,
        the more accurate the noise bias will be, but the time it takes to complete
        running the code will increase.
    B_pspec_cov: Give a 1-D vector of length ELLMAX which represents the BB power
        spectrum of what you think your data will look like based on theory. This
        is used to calculate the BB covariance matrix for the iterate function.

    Returns
    A matrix of size NSIMS by ELLMAX that is the noise bias associated with the given
    mask as a function of ell. This a list of power spectra of random noise that
    is fed through the cleaning process to attempt to ascertain just how much power
    in our cleaned map isl coming from noise.

    Also returns a list of length NSIMS which are IQU maps which are the IQU maps
    calculated using random noise as the signal. These are the maps that get
    passed through the iterate function.

    Also returns a list of alms that are the output of the iterate function.
    This will be size NSIMS with each entry being size 2 * NSPH. The first NSPH
    entries in each list piece of the array will be all zeros corresponding
    to the E part of the alm's. These are zero because we have applied a
    projection at the end of the iterate function. The last NSPH entries
    will be the B alms.

    Also returns a an array of size NSIMS x ELLMAX. These are the power spectrums
    of the input maps.
    """
    Ndiag = np.concatenate(((mask)**(-1), (mask)**(-1)), axis = 0)
    NPIX = len(mask)
    NSIDE = int(np.sqrt(NPIX/12))
    ELLMAX = 3*NSIDE - 1
    NSPH = int(np.sum(1.0 * np.arange(ELLMAX+1) + 1.0))
    list_of_noise_b_spectra1 = []
    list_of_i_q_u1 = []
    list_of_noise_alms1 = []
    iqu_pspecs1 = []
    for x in range(NSIMS):
        i_q_u = [np.random.randn(NPIX) * 0.19 / np.sqrt(mask), np.random.randn(NPIX) * 0.19 / np.sqrt(mask), np.random.randn(NPIX) * 0.19 / np.sqrt(mask)]
        list_of_i_q_u1.append(i_q_u)
        pspecs = hp.anafast(i_q_u, pol = True)
        iqu_pspecs1.append(pspecs[2])
        t_e_b = hp.map2alm(i_q_u, lmax = ELLMAX, pol = True)
        first = iterate(mask, i_q_u, B_pspec_cov)
        list_of_noise_alms1.append(first[NSPH:])
        sph2map = hp.alm2map((t_e_b[0], first[:NSPH], first[NSPH:]), NSIDE, lmax = ELLMAX, pol = True)
        second = hp.anafast(sph2map, lmax = ELLMAX, pol = True)
        list_of_noise_b_spectra1.append(second[2])
    list_of_noise_b_spectra = np.array(list_of_noise_b_spectra1)
    list_of_i_q_u = np.array(list_of_i_q_u1)
    list_of_noise_alms = np.array(list_of_noise_alms1)
    iqu_pspecs = np.array(iqu_pspecs1)
    return list_of_noise_b_spectra, list_of_i_q_u, list_of_noise_alms, iqu_pspecs

def calculate_eb_leakage(E_pspec, NSIMS, mask, B_pspec_cov):
    """Parameters
    E_pspec: Give a 1-D vector that is the E mode power spectrum. This will be used
        to create an E mode only map within the function. This power spectrum should
        be a simulated spectrum based on what you think the E mode power spectrum
        is for your data.
    NSIMS: Give a number that is the number of simulations run. The higher the number
        of sims is the more accurate the measure of e to b leakage is, but the longer
        the code will take to run.
    Ndiag: Give a 1-D vector of length NSPH which represents the diagonal of the
        noise covariance matrix.
    B_pspec_cov: Give a 1-D vector of length ELLMAX which represents the BB power
        spectrum that will be used to calculate the B covariance matrix in the
        iterate function.

    Returns
    A matrix of size NSIMS by ELLMAX that represents the leakage from E modes to B modes
    as a result of looking at a finite portion of the sky. Each row of the matrix
    represents a power spectrum of e to b leakage for a given simulation.

    Also returns a numpy array of size NSIMS by NPIX. These are the input IQU
    maps that were put through the iterate function.

    Also returns a numpy array of size NSIMS by ELLMAX. These are the BB power
    spectra for the input IQU maps.

    Also returns a numpy array of size NSIMS by NSPH. These are the alms for the
    B modes after filtering has been completed.
    """
    Ndiag = np.concatenate(((mask)**(-1), (mask)**(-1)), axis = 0)
    NPIX = len(mask)
    NSIDE = int(np.sqrt(NPIX/12))
    ELLMAX = int(3 * NSIDE - 1)
    NSPH = int(np.sum(1.0 * np.arange(ELLMAX+1) + 1.0))
    list_of_b_pspec21 = []
    list_of_iqu1 = []
    iqu_pspecs1 = []
    b_alms1 = []
    for i in range(NSIMS):
        i_q_u = hp.synfast((np.zeros(ELLMAX), E_pspec, np.zeros(ELLMAX), np.zeros(ELLMAX)), NSIDE, new = True)
        list_of_iqu1.append(i_q_u)
        iqu_ana = hp.anafast(i_q_u, pol=True, lmax = ELLMAX)
        iqu_pspecs1.append(iqu_ana[2])
        t_e_b = hp.map2alm(i_q_u, lmax = ELLMAX, pol = True)
        first = iterate(mask, i_q_u, B_pspec_cov)
        b_alms1.append(first[NSPH:])
        second = hp.alm2cl((t_e_b[0],first[:NSPH],first[NSPH:]), lmax = ELLMAX)
        list_of_b_pspec21.append(second[2])
    list_of_b_pspec2 = np.array(list_of_b_pspec21)
    list_of_iqu = np.array(list_of_iqu1)
    iqu_pspecs = np.array(iqu_pspecs1)
    b_alms = np.array(b_alms1)
    return list_of_b_pspec2, list_of_iqu, iqu_pspecs, b_alms

def calculate_supfacs(B_pspec_sim ,B_pspec_cov, NSIMS, mask):
    """Parameters
    B_pspec_sim: Give a 1-D vector of length EllMAX. This represents the power spectrum
        of the B mode and will be used to create a B only map. This power spectrum should
        be a simulated spectrum based on what you think the B mode power spectrum
        is for your data.
    B_pspec_cov: B power used to create B covariance matrix. This should be a theoretical
        spectrum of what you think your data looks like for a BB power spectrum.
    NSIMS: Give a number that represents the number of simulations that will be run
        when calculating the suppression factors as a function of ell. The higher
        the number of simulations run, the more accurate the suppression factors
        are, but the longer the code will take to run.
    Ndiag: Give a 1-D vector of length NSPH which represents the diagonal of the
        noise covariance matrix.

    Returns
    Two matrices, the first matrix of size NSIMS by ELLMAX which represents the
    suppression of the signal due to the filter as a function of ell. The second
    matrix is also size NSIMS by ELLMAX and this matrix represents the input
    BB power spectra. These two will be used to find suppression factors in the analysis.
    """
    Ndiag = np.concatenate(((mask)**(-1), (mask)**(-1)), axis = 0)
    NPIX = len(mask)
    NSIDE = int(np.sqrt(NPIX/12))
    ELLMAX= 3*NSIDE - 1
    NSPH = int(np.sum(1.0 * np.arange(ELLMAX+1) + 1.0))
    list_of_b_pspec1 = []
    list_of_input_b_pspec1 = []
    input_iqu1 = []
    supfacs_alms1 = []
    for j in range(NSIMS):
        i_q_u = hp.synfast((np.zeros(ELLMAX), np.zeros(ELLMAX), B_pspec_sim, np.zeros(ELLMAX)), 512, new = True)
        input_iqu1.append(i_q_u)
        t_e_b = hp.map2alm(i_q_u, lmax = ELLMAX, pol = True)
        first = iterate(mask ,i_q_u, B_pspec_cov)
        supfacs_alms1.append(first[NSPH:])
        second = hp.alm2cl((t_e_b[0], first[:NSPH], first[NSPH:]), lmax = ELLMAX)
        list_of_b_pspec1.append(second[2])
        list_of_input_b_pspec1.append(hp.anafast(i_q_u, lmax = ELLMAX, pol = True)[2])
    list_of_b_pspec = np.array(list_of_b_pspec1)
    list_of_input_b_pspec = np.array(list_of_input_b_pspec1)
    input_iqu = np.array(input_iqu1)
    supfacs_alms = np.array(supfacs_alms1)
    return list_of_b_pspec, list_of_input_b_pspec, input_iqu, supfacs_alms

def bin_pspec(pspec, first_ell, last_ell, binsize):
    """Paramters
    pspec: Give a 1-D vector of size ELLMAX that is the power spectrum wishing
        to be binned.
    NBINS: Give a number that represents the number of bins that you wish to bin
        the power spectrum into. In other words, this will be the number of entries
        in your ouptut binned power spectrum

    Returns
    A 1-D vector that is size NBINS. The code will average values in the power
    spectrum within the bin ranges. For example, if we consider a bin consisting
        of elements zero through ten of a power spectrum, then we can say that the
        first element of the output binned power spectrum will be the average
        of elements zero through ten of the input power spectrum.
    """
    ELLMAX = len(pspec)
    bins = range(first_ell, last_ell, binsize)
    NBINS = int(len(bins))
    binned_pspec = []
    for i in range(NBINS-1):
        binned_pspec.append(np.average(pspec[bins[i]:bins[i+1]]))
    binned_pspec.append(np.average(pspec[bins[-1]:]))
    return binned_pspec

def comb_files(directory):
    """Parameters:
    directory: Give the full file path to the directory which the files are in.

    returns: a list that is all of the data files in the directory combined into one list
    
    note: the data files must be of the form 00i.npz where i is an integer between 0 and the number of files in the directory. This also only works for power spectra that are size ELLMAX = 1536."""
    wd = os.getcwd()
    os.chdir(directory)
    files = np.empty([len(os.listdir('.'))*10, 1536])
    for ind,data in enumerate(os.listdir('{}'.format(directory))):
        files[10*ind:10*ind+10,:] = np.load("{}".format(data))['arr_0']
    return files


def comb_and_reducefiles(directory,sims_per_file,reducer):
    """Parameters:                                                                                       
    directory: Give a string that is the directory containing the files you wish to combine.
    sims_per_file: Give an integer that is the number of simulations run per file that is in the directory
    reducer: Give an integer that is the ell value you wish your output arrays to go up to.

    returns: The function will put together all of the files in the directory into one array, then it will take just the first 500 (if the reducer is 500) of each of the spectra and then bin that spectra into bins of size 20.           
    note: This only works for power spectra that are size ELLMAX = 1536."""
    wd = os.getcwd()
    os.chdir(directory)
    files = np.empty([len(os.listdir('{}'.format(directory)))*sims_per_file, reducer])
    counter = -1
    for ind,data in enumerate(os.listdir('{}'.format(directory))):
        x = np.load("{}".format(data))['arr_0']
        xred = x[:,:reducer]
        for i in range(sims_per_file):
            counter+=1
            y = xred[i,:]
            files[counter,:] = y
    return files
    
def gen_figures(directories):
    """Parameters:
    directories: Give a tuple of size three which is the name of each of the directories that you need to plot. The directories should contain data files which will be averaged together before they are graphed.

    Returns:
    3 plots, that are presumably noise, leakage, and suppression factor.
    Formally returns the three full lists of noise, leakage, and eb leakage. These
    lists will be NSIMS by 1536. Also returns an averaged version of the noise, leakage, and suppression factor. These averaged versions will be size one by 1536."""
    x = comb_files("{}".format(directories[0]))
    os.chdir("/users/PES0740/ucn3066/messenger")
    y = comb_files("{}".format(directories[1]))
    os.chdir("/users/PES0740/ucn3066/messenger")
    z = comb_files("{}".format(directories[2]))
    os.chdir("/users/PES0740/ucn3066/messenger")
    avgx = np.mean(x,axis=0)
    avgy = np.mean(y,axis=0)
    avgz = np.mean(z,axis=0)
    plt.figure()
    plt.semilogy(avgx)
    plt.title(directories[0])
    plt.xlabel('$\ell$')
    plt.ylabel('pspec')
    plt.figure()
    plt.semilogy(avgy)
    plt.title(directories[1])
    plt.xlabel('$\ell$')
    plt.ylabel('pspec')
    plt.figure()
    plt.semilogy(avgz)
    plt.title(directories[2])
    plt.xlabel('$\ell$')
    plt.ylabel('pspec')
    return x,y,z,avgx,avgy,avgz

def comb_bpwf_files(path, NBINS, NSIMS):
    """
    Paramters
    path: Give the full file path of the directory containing the files needing to be combined.
    NBINS: Give a number that is the number of bins. This should be the same as the length of any of the individual files in the file path. This in our original case is 25.

    Returns:
    A list which is 3 dimensional and is length 500 by 25 by NSIMS corresponding to the 25 bins present and 500 ell values we are looking at as well as the number of sims run.
    """

    os.chdir("{}".format(path))
    empty = np.empty([500, NBINS, NSIMS])
    for index,data  in enumerate(os.listdir(path)):
        x = np.load("{}".format(data))['arr_0']
        empty[:,:,index] = x
    return empty

def comb_bpwf_files_dlfix(path, NBINS, NSIMS):
    """                                                                                                              
    Paramters                                                                                                        
    path: Give the full file path of the directory containing the files needing to be combined.                      
    NBINS: Give a number that is the number of bins. This should be the same as the length of any of the individual \
files in the file path. This in our original case is 25.                                                             
                                                                                                                     
    Returns:                                                                                                         
    A list which is 3 dimensional and is length 500 by 25 by NSIMS corresponding to the 25 bins present and 500 ell \
values we are looking at as well as the number of sims run.                                                          
    """

    os.chdir("{}".format(path))
    empty = np.empty([500, NBINS, NSIMS])
    for index,data  in enumerate(os.listdir(path)):
        x = np.load("{}".format(data))['arr_0']
        empty[:,:,index] = [(2*np.pi/((i+1)*(i+2)))*item for i,item in enumerate(x)]
    return empty

def integrate_bpwf_bb2bb(NBINS):
    """
    Parameters
    NBINS: Give a number that is the number of bins that you have. That should be equal to the number of band power window functions you have.

    Returns:
    A list which has the values of the integrals. This list will have length NBINS.
"""
    os.chdir('/users/PES0740/ucn3066/messenger/bpwf/Results')
    x = np.load('Averaged BPWF BB2BB D_ell Fixed.npz')['arr_0']
    print('shape of x', np.shape(x))
    empty = np.empty(NBINS)
    for i in range(NBINS):
        def_integral = np.sum(x[:,i])
        empty[i] = def_integral
    return empty

def cell2dell(path):
    """Parameters
    path: Give a string that is the path to the directory in which the files are that need to be transformed from c_ell to d_ell

    Returns:
    A numpy array which has the data from all the files moved to d_ell instead of c_ell.
"""
    os.chdir(path)
    empty = np.empty([len(os.listdir(path)),500])
    for ind,data in enumerate(os.listdir(path)):
        x = np.load('{}'.format(data))['arr_0']
        y = [datapoint*(index)*(index+1)/(2*np.pi) for index,datapoint in enumerate(x)]
        empty[ind,:] = y
    return empty
        
def comb_bin_files(path, sims_per_file):
    """
    Parameters
    path: Give a string that is a path to the directory containing the files you want combined and binned
    sims_per_file: Give an integer that is the number of simulations run for each file in the directory specified by the path

    Returns: A numpy array that is length 25 by sims_per_file by the number of files in the directory specificed by the path. This is designed to convert noise and leakage simulations into d_ell values and then to bin the spectra, however it can be generalized for use in other endeavors"""
    os.chdir('{}'.format(path))
    empty = np.empty([25,sims_per_file*len(os.listdir('{}'.format(path)))])
    counter = -1
    for ind,data in enumerate(os.listdir('{}'.format(path))):
        for i in range(sims_per_file):
            counter+=1
            x = np.load('{}'.format(data))['arr_0']
            y = [datapoint*(index+1)*(index+2)/(2*np.pi) for index,datapoint in enumerate(x[i,:])]
            y500 = y[:500]
            binned_y = bin_pspec(y500,0,500,20)
            empty[:,counter] = binned_y
    return empty

