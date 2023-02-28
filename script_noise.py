
import argparse
import numpy as np
import random
import pycsou.operator.linop.nufft as nt
from pycsou.opt.solver.pgd import PGD
import pycsou.operator.func.norm as nm
import pycsou.opt.stop as pycos
import datetime as dt
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)


def createSparceSource(n, m, sparsity, minUniform=0.0, maxUniform=10.0, meanNormal=0.0, varNormal=1.0, scaleExp=1.0, distribution="uniform"):
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
    size = n * m
    nb_indices = int(size * sparsity)
    
    # Chose nb_indices indices that will be non-zero
    indices = np.random.choice(size, nb_indices, replace=False)

    # Create a sparse source
    sparseSource = np.zeros(size)

    if (distribution == "uniform"):
        assert(minUniform <= maxUniform),  "minimum is bigger than maximum"
        sparseSource[indices] = rng.uniform(low=minUniform, high=maxUniform, size=nb_indices)

    elif (distribution == "normal"):
        sparseSource[indices] = rng.normal(loc=meanNormal, scale=varNormal, size=nb_indices)

    elif (distribution == "exponential"):
        sparseSource[indices] = rng.exponential(scale=scaleExp,size=nb_indices)  
    else:
        raise Exception("this distribution does not exist")

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
    size = n * m
    sampling = rng.choice(size, L, replace=False)
    X = 2 * np.pi * sampling / size
    phi = nt.NUFFT.type2(X, size, isign=-1, eps=1e-3, real=True)
    print(type(phi))
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




def plot1(original, reconstructed):
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
    original_norm = original_nonzero / np.max(original_nonzero)
    reconstructed_norm = reconstructed_nonzero / np.max(reconstructed_nonzero)

    # Create QQ plot
    fig, ax = plt.subplots()
    ax.scatter(reconstructed_norm, original_norm)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_ylabel('Original Intensity (normalized)')
    ax.set_xlabel('Reconstructed Intensity (normalized)')
    ax.set_title('QQ Plot of Reconstructed vs. Original Intensity')
    plt.show()







def main():
    
    #def positive_int(value):
        #value = float(value)
        #if value != int(value):
            #raise argparse.ArgumentTypeError("%s is not an integer" % value)
        #if value <= 0:
            #raise argparse.ArgumentTypeError("%s is not a positive integer" % value)
        #return int(value)

    def threshold(value):
        value = float(value)
        if value < 0 or value > 1:
            raise argparse.ArgumentTypeError("%s is not a valid threshold value" % value)
        return value

    parser = argparse.ArgumentParser(description='Process some integers.')
    #parser.add_argument('-n', type=positive_int, help='height of the matrix' )
    #parser.add_argument('-m', type=positive_int, help='width of the matrix')
    parser.add_argument('-t', type=threshold, help='a threshold as a float')
    parser.add_argument('-d', type=str, help='distribution of the sources')
    args = parser.parse_args()

    n = 64
    m = 64
    threshold = args.t
    SNR_dB = 10

    sparse_mat = createSparceSource(n,m, threshold, distribution=args.d) 
    phi = compute_phi(n,m, n*m)
    y = phi(sparse_mat)
    y_noised = add_noise(y, SNR_dB)


    min_iterations = 3 * n * m
    tmax = 30
    # Chose 10 values using logspace between 0 and 1
    #xs = np.logspace(-1,0,10)
    #mses = []
    #for x in xs:

    lambda_factor = .1

    lambda_ = lambda_factor * np.linalg.norm(phi.adjoint(y_noised), np.infty)
    min_iter = pycos.MaxIter(n=min_iterations)
    data_fid = .5 * nm.SquaredL2Norm().asloss(y_noised) * phi  # F
    regul = lambda_ * nm.L1Norm()  # G

    sol = reconstruction_pgd(n*m, data_fid, regul, min_iter, tmax)
    

    mse = compute_mse(sparse_mat, sol)
    #mses.append(mse)

    print(mse)
    min_iterations = 3 * n * m
    plot1(sparse_mat, sol)

if __name__ == "__main__":
    main()
    
