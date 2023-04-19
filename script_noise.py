
import argparse
import numpy as np
import random
import pycsou.operator.linop.nufft as nt
from pycsou.opt.solver.pgd import PGD
import pycsou.operator.func.norm as nm
import pycsou.opt.stop as pycos
import datetime as dt
import matplotlib.pyplot as plt
import pycsou.util as pycu
import IRL1Norm as ir
from matplotlib.colors import Normalize
rng = np.random.default_rng(1)


def createSparceSource(size, sparsity, scaleExp=1.0):
    """ 
    Create a sparse source with a given sparsity. The sparsity is the percentage of non-zero values in the source.
    The source can be uniform, normal or exponential.

    Parameters
    ----------
    n : int, the height of the image
    m : int, the width of the image
    sparsity : float, The percentage of non-zero values in the source

    Returns
    -------
    sparseSource : np.array, the sparse source
    """
    nb_indices = int(size * sparsity)
    
    # Chose nb_indices indices that will be non-zero
    indices = np.random.choice(size, nb_indices, replace=False)

    # Create a sparse source
    sparseSource = np.zeros(size)

    # Create unfiform source with high dynamic range

    sparseSource[indices] = rng.uniform(low=1.0, high=1e5, size=nb_indices)
    # Create exponential source
    #sparseSource[indices] = rng.exponential(scale=scaleExp, size=nb_indices)
    
    return sparseSource



def compute_phi(n,m, L):
    """ 
    Compute the phi operator for the NUFFT

    Parameters
    ----------
    n : int, the height of the image
    m : int, the width of the image
    L : int, the number of samples

    Returns
    -------
    phi : pycsou.abc.operator.LinOp, the phi operator
    """

    # Sample L random 2D indices
    choices = [(x1, x2) for x1 in range(n) for x2 in range(m)]
    random_indices = rng.choice(choices, L, replace = False)

    # Compute the phi operator
    X = np.asarray([(2 * np.pi * x1 / n, 2 * np.pi * x2 / m) for x1, x2 in random_indices])
    phi = nt.NUFFT.type2(X, (n,m), isign=-1, eps=1e-3, real=True)

    return phi


def reconstruction_pgd(size, data_fid, regul, min_iter, tmax):
    """
    Reconstruction using PGD

    Parameters
    ----------
    size : int, the size of the image
    data_fid : pycsou.operator.func.fid.DataFidelity, the data fidelity
    regul : pycsou.operator.func.norm.Norm, the regularization
    min_iter : int, the minimum number of iterations of the reconstruction 
    tmax : int, the maximum time in seconds of the reconstruction

    Returns
    -------
    pgd.solution() : np.array, the reconstructed image
    """
    pgd = PGD(data_fid, regul, show_progress=False, verbosity=size)
    pgd.fit(x0=np.zeros(size),
            stop_crit=(min_iter & pgd.default_stop_crit()) | pycos.MaxDuration(t=dt.timedelta(seconds=tmax)),
            track_objective=True,
            )
    return pgd.solution()


def compute_mse(I_true, I_pred):
    """
    Compute the mean square error between two images

    Parameters
    ----------
    I_true : np.array, the original image
    I_pred : np.array, the reconstructed image

    Returns
    -------
    np.square(np.subtract(I_true,I_pred)).mean() : float, the mean square error
    """
    return np.square(np.subtract(I_true,I_pred)).mean()


def add_noise(signal, desired_snr):
    """
    Add noise to a signal

    Parameters
    ----------
    signal : np.array, the signal
    desired_snr : float, the desired signal to noise ratio (in dB

    Returns
    -------
    signal + noise : np.array, the noisy signal
    """
    # Compute signal power 
    signal_power = np.sum(signal**2) / len(signal)
    noise_power = signal_power / (10**(desired_snr/10))

    # Generate normal noise with given power
    noise = np.random.normal(scale=np.sqrt(noise_power), size=len(signal))

    return signal + noise




def plot_qq(original, reconstructed):
    """
    Plot the original and reconstructed images

    Parameters
    ----------
    original : np.array, the original image
    reconstructed : np.array, the reconstructed image
    """

    # Select only non-zero values
    original_nonzero = original[original != 0]
    reconstructed_nonzero = reconstructed[original != 0]
    
    # Normalize original and reconstructed non-zero values to [0, 1]
    original_norm = original_nonzero 
    reconstructed_norm = reconstructed_nonzero

    # Create QQ plot
    fig, ax = plt.subplots()
    ax.scatter(reconstructed_norm, original_norm)
    ax.plot([0, np.max(original_norm)], [0, np.max(original_norm)], 'k--')
    ax.set_ylabel('Original Intensity (normalized)')
    ax.set_xlabel('Reconstructed Intensity (normalized)')
    ax.set_title('QQ Plot of Reconstructed vs. Original Intensity')
    plt.show()


def plot_image(original, reconstructed, n, m):
    # resize the images 
    original = original.reshape(n, m)
    reconstructed = reconstructed.reshape(n, m)

    # Find the minimum and maximum values across both images
    min_val = min(original.min(), reconstructed.min())
    max_val = max(original.max(), reconstructed.max())

    # Normalize the color mapping
    norm = Normalize(vmin=min_val, vmax=max_val)

    # plot two images on the same figure
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    im1 = ax[0].imshow(original, cmap='viridis', norm = norm)
    ax[0].set_title('Original Image')
    fig.colorbar(im1, ax=ax[0])

    im2 = ax[1].imshow(reconstructed, cmap='viridis', norm = norm)
    ax[1].set_title('Reconstructed Image')
    fig.colorbar(im2, ax=ax[1])
    plt.show()


def plot_image_mean_pixels(original, reconstructed, n, m):
    
    # resize images
    original = original.reshape(n, m)
    reconstructed = reconstructed.reshape(n, m)

    # for each pixel that is not null in the original image, compute the mean of the 9 pixels around it
    original_mean = np.zeros((n, m))
    reconstructed_mean = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            if original[i, j] != 0:
                original_mean[i, j] = np.mean(original[max(0, i-1):min(n, i+2), max(0, j-1):min(m, j+2)])
                reconstructed_mean[i, j] = np.mean(reconstructed[max(0, i-1):min(n, i+2), max(0, j-1):min(m, j+2)])


    # Find the minimum and maximum values across both images
    min_val = min(original_mean.min(), reconstructed_mean.min())
    max_val = max(original_mean.max(), reconstructed_mean.max())

    # Normalize the color mapping
    norm = Normalize(vmin=min_val, vmax=max_val)

    # plot two images on the same figure
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    im1 = ax[0].imshow(original_mean, cmap='viridis', norm = norm)
    ax[0].set_title('Original Image')
    fig.colorbar(im1, ax=ax[0])

    im2 = ax[1].imshow(reconstructed_mean, cmap='viridis', norm = norm)
    ax[1].set_title('Reconstructed Image')
    fig.colorbar(im2, ax=ax[1])
    plt.show()


def iterative_reweighting_l1(V, phi, size, i_max=8, epsilon=1, lambda_ = 1, tol = 1e-3):
    """
    Iterative reweighting algorithm for l1 minimization

    Parameters
    ----------
    V : np.array, the measurements
    phi : pycsou.abc.operator.LinOp, the phi operator
    size : int, the size of the image
    i_max : int, the maximum number of iterations
    epsilon : float, the epsilon parameter
    tot : float, the tolerance

    Returns
    -------
    result.x : np.array, the reconstructed image
    """

    # Set up initial guess and initial weights
    I = np.zeros(size)
    weights = lambda_ * np.ones(size)
    
    # Set up reconstruction parameters
    min_iterations = pycos.MaxIter(n= 3 * size)
    tmax = 30
    data_fid = .5 * ir.SquaredL2Norm().asloss(V) * phi # F
    
    relative_change = 1
    iteration = 0 
    while iteration < i_max and relative_change > tol:
        regul = ir.IRL1Norm_2(W=weights) # G

        result = reconstruction_pgd(size, data_fid, regul, min_iterations, tmax)
        
        # Update weights
        if iteration != 0:
            relative_change = np.linalg.norm(result - I) / np.linalg.norm(I)
        I = result
        weights = lambda_ / (epsilon + np.abs(I))

        print("rel_change on iteration", iteration, ":", relative_change)
        iteration += 1
        
    return I




def main():
    
    def positive_int(value):
        value = float(value)
        if value != int(value):
            raise argparse.ArgumentTypeError("%s is not an integer" % value)
        if value <= 0:
            raise argparse.ArgumentTypeError("%s is not a positive integer" % value)
        return int(value)

    def sparsity(value):
        value = float(value)
        if value < 0 or value > 1:
            raise argparse.ArgumentTypeError("%s is not a valid threshold value" % value)
        return value

    parser = argparse.ArgumentParser(description='Reconstruct an image using mutliple methods')
    parser.add_argument('-n', type=positive_int, default=64, help='height of the matrix' )
    parser.add_argument('-m', type=positive_int, default=64, help='width of the matrix')
    parser.add_argument('-s', type=sparsity, default = 0.01, help='Sparsity of the source as a float')
    parser.add_argument('-r', type=str, choices=['PGD', 'IRL1', 'BOTH'], required=True, help='Reconstruction method')
    parser.add_argument('-SNR', type=float, default=20, help='Signal to noise ratio')
    parser.add_argument('-p', type=float, default=0.3, help='Percentage of measurements to use')

    args = parser.parse_args()
    
    # Set up parameters
    n = args.n
    m = args.m
    size = n * m
    reconstruction = args.r
    sparsity = args.s
    SNR_dB = args.SNR
    percentage = args.p
    
    # Create sparse source 
    sparse_mat = createSparceSource(size, sparsity)
    phi = compute_phi(n,m, int(percentage * size))
    
    # Create measurements 
    y = phi.apply(sparse_mat)
    y_noised = add_noise(y, SNR_dB)

    
    #Reconstruction using PGD
    if reconstruction == 'PGD' or reconstruction == 'BOTH':
        # define the regularization parameter
        lambda_factor = .1
        lambda_ = lambda_factor * np.linalg.norm(phi.adjoint(y_noised), np.infty)

        # define parameters for the PGD algorithm
        min_iter = pycos.MaxIter(n=3 * size)
        tmax = 30
        data_fid = .5 * nm.SquaredL2Norm().asloss(y_noised) * phi  # F
        regul = lambda_ * nm.L1Norm()  # G

        # Reconstruction using PGD
        sol = reconstruction_pgd(size, data_fid, regul, min_iter, tmax)

        # Compute evaluation metrics
        mse = compute_mse(sparse_mat, sol)
        print(f'mse PGD: {mse:_}')
        plot_qq(sparse_mat, sol)
        plot_image(sparse_mat, sol, n, m) 

    # Reconstruction using IRL1
    if reconstruction == 'IRL1' or reconstruction == 'BOTH':
        # define the regularization and epsilon parameters
        epsilon = np.mean(sparse_mat[sparse_mat != 0])
        lambda_ = .1 * np.linalg.norm(phi.adjoint(y_noised), np.infty)

        # Reconstruction using IRL1
        I_test = iterative_reweighting_l1(y_noised, phi, size = size, lambda_ = lambda_,  epsilon = epsilon)
        
        print("number of values in range [0, 100] in the original image:", np.sum((sparse_mat > 0) & (sparse_mat < 1000)), "out of", size)
        #print number 
        print("number of values in range [0, 100] in the reconstructed image:", np.sum((I_test > 0) & (I_test < 1000)), "out of", size)
        # Compute evaluation metrics
        mse = compute_mse(sparse_mat, I_test)
        print(f'mse IRL1: {mse:_}')
        plot_qq(sparse_mat, I_test)
        plot_image(sparse_mat, I_test, n, m)
        plot_image_mean_pixels(sparse_mat, I_test, n, m)


if __name__ == "__main__":
    main()
    
