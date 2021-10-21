# abel-transform
Python code for Abel transformations (forward and inverse) of one- or two-dimensional data.

Forward/inverse Abel transformation based on convolution with an Abel matrix. Image can be 1D or 2D, but the left-most pixel must correspond to the
symmetry axis.
The Abel matrix contains the information how each pixel contributes to the Abel forward transformation, so that the Abel forward transformation is a simple multiplication of Abel matrix A with pixel values x for each image row:
    
    b = A*x
    
The inverse Abel transformation (Abel inversion) is solved using a non-negative least-squares algorithm (scipy.optimize.nnls), where the Abel matrix A and the convoluted (and noisy) data b are the inputs:
    
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
