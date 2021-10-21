# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 14:46:03 2020

Forward/inverse Abel transformation based on convolution with an Abel matrix.
Image can be 1D or 2D, but the left-most pixel must correspond to the
symmetry axis.

The Abel matrix contains the information how each pixel contributes to the 
Abel forward transformation, so that the Abel forward transformation is a 
simple multiplication of Abel matrix A with pixel values x for each image row:
    
    b = A*x
    
The inverse Abel transformation (Abel inversion) is solved using a 
non-negative least-squares algorithm (scipy.optimize.nnls), where the
Abel matrix A and the convoluted (and noisy) data b are the inputs:
    
    minimize |A*x - b|^2 for x >= 0

A: Abel transformation matrix.
x: Pixel values of the deconvoluted data
b: Pixel values of the convoluted data

Functions:

1) abel_matrix(imax)
    
    Calculates the Abel transformation matrix A.
    
    Input:
        imax: the maximum number of pixel values in a row of the image
    Returns:
        Upper triangular matrix (imax*imax shape) of convolution factors

2) example_data(lamda)
    
    Calculates some example data to be used for first tests.
    
    Input:
        lamda: Value for Poisson noise. Default is 0.01. Set higher for more noise.
    Returns:
        1D array with Abel-transformed and noisy data.
    
3) abel_transform(image, direction='inverse', maxiter=None)
    
    Calculates the forward/inverse Abel transformation.
    
    Input:
        image: 1D or 2D array of pixel values (symmetry axis at left-most pixel)
    
    Parameters:
        
        direction:  string, optional
                    Either "inverse" for an Abel inversion (default)
                    or anything else (e.g. "forward") for an forward Abel transform.
                    
        maxiter:    int, optional
                    Maximum number of iterations of the NNLS solver.
                    Default is 3*imax, i.e. the number of pixels in one image row.
    
    Returns:
        Array of the same shape as the input array with transformed data.


Example how to use the functions:
    
    import matplotlib.pyplot as plt
    
    b = example_data(lamda=0.05)     # generate example data with noise
    x = abel_transform(b)   # Abel inversion of noisy example data b
    b_again = abel_transform(x, direction='forward')    # Abel forward transformation
    
    x_ideal = abel_transform(example_data(lamda=0.0001))
    
    plt.plot(x)
    plt.plot(x_ideal)

You do not need to use the abel_matrix function for anything.

@author: David S. Vogt
"""

import numpy as np
from scipy.optimize import nnls
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
from scipy.optimize import Bounds
from scipy.optimize import curve_fit

def abel_matrix(imax):
    indx = np.arange(imax)
    matrix = np.diag(2*np.sqrt(2*indx + 1))
    for i in range(imax):
        matrix[:i,i] = 2*(np.sqrt((i+1)**2 - indx[:i]**2) - np.sqrt(i**2 - indx[:i]**2))
    return matrix

def example_data(lamda=0):
    x = np.arange(500)
    g1 = 16*np.exp(-(x - 20)**2/400)
    g2 = 12*np.exp(-(x - 70)**2/1000)
    g3 = 10*np.exp(-(x - 200)**2/4000)
    g4 = 6*np.exp(-(x - 300)**2/700)
    yt = abel_transform(g1 + g2 + g3 + g4, direction='forward')
    if lamda > 0: yt = np.random.poisson(yt/lamda)*lamda
    return yt

def abel_transform(image, direction='inverse', maxiter=None):
    # In case the input is a 1D array, transform it to be 2D:
    data = np.atleast_2d(image)
    rows, cols = data.shape
    
    # Calculate the Abel transformation matrix applied to each row:
    conv_matrix = abel_matrix(cols)
    
    # Create empty array to store the transformed data
    abel_trans = np.empty_like(data)
    
    if direction == 'inverse':
        # Abel inversion by solving the NNLS equation
        for row in range(rows):
            fit = nnls(conv_matrix, data[row,:], maxiter=maxiter)
            abel_trans[row,:] = fit[0]
    else:
        # Abel forward transformation by matrix multiplication
        for row in range(rows):
            abel_trans[row,:] = np.dot(conv_matrix, data[row,:])
        
    if rows == 1:
        abel_trans = abel_trans[0,:]
    
    return abel_trans

def abelinv(image, maxiter=None):
    return abel_transform(image, direction='inverse', maxiter=maxiter)

def abelfwd(image, maxiter=None):
    return abel_transform(image, direction='fwd', maxiter=maxiter)

def abelinv_poly(image, deg):
    # In case the input is a 1D array, transform it to be 2D:
    data = np.atleast_2d(image)
    rows, cols = data.shape
    
    # Calculate the Abel transformation matrix applied to each row:
    conv_matrix = abel_matrix(cols)
    
    # Create empty array to store the transformed data
    abel_trans = np.empty_like(data)
    
    def fitfun(x, *p):
        poly = sum((a*x**i for i,a in enumerate(p)))
        return np.dot(conv_matrix, poly)
    
    
    x = np.arange(cols)
    
    # Abel inversion by fitting an Abel-transformed polynomial to the data
    for row in range(rows):
        y = data[row,:]
        p0 = np.polyfit(x, y, deg)
        popt, pcov = curve_fit(fitfun, x, y, p0=p0)
        abel_trans[row,:] = sum((a*x**i for i,a in enumerate(popt)))
        
    if rows == 1:
        abel_trans = abel_trans[0,:]
    
    return abel_trans


def SnegA(x, *args):
    f_j = x
    A = args
    f_j[f_j < 1e-6] = 1e-6
    return np.sum(f_j*(np.log(f_j/A) - 1))

def SnegA_deriv(x, *args):
    f_j = x
    A = args
    f_j[f_j < 1e-6] = 1e-6
    return np.log(f_j) - np.log(A)
    # return np.log(f_j)

def SnegA_hess_p(x, p, *args):
    f_j = x
    f_j[f_j < 1e-6] = 1e-6
    return p/f_j

def abel_inversion_MEM(image, xtol=1e-3, gtol=1e-3, maxiter=1000, sig=0, A=1):
    # In case the input is a 1D array, transform it to be 2D:
    data = np.atleast_2d(image)
    rows, cols = data.shape
    
    # Calculate the Abel transformation matrix applied to each row:
    R_kj = abel_matrix(cols)
    R_kj_T = R_kj.transpose()
    
    # Create empty array to store the transformed data
    abel_inv_MEM = np.empty_like(data)
    
    C_min = sig**2*cols
    C_max = sig**2*(cols + 3.29*np.sqrt(cols))
    bounds = Bounds(1e-6, np.inf)
    
    # Abel inversion by solving the NNLS equation
    for row in range(rows):
        
        D_k = data[row,:]
        
        C_fun = lambda f_j: np.sum((np.dot(R_kj, f_j) - D_k)**2)
        C_jac = lambda f_j: np.dot(R_kj_T, 2*(np.dot(R_kj, f_j) - D_k))
        C_hes = lambda f_j, lam: lam*R_kj
        
        cons_C = NonlinearConstraint(C_fun, C_min, C_max, jac=C_jac, hess=C_hes)
        
        x0 = nnls(R_kj, D_k)[0]
        
        res = minimize(SnegA, x0, args=(A),
                        method='trust-constr',
                        jac=SnegA_deriv, hessp=SnegA_hess_p,
                        constraints=[cons_C], bounds=bounds,
                        options={'xtol': xtol, 'gtol': gtol, 
                                 'verbose': 2, 'maxiter': maxiter})
        
        abel_inv_MEM[row,:] = res.x
        
    if rows == 1:
        abel_inv_MEM = abel_inv_MEM[0,:]
    
    return abel_inv_MEM