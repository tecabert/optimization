
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
    
    size = n * m
    nb_indices = int(size * sparsity)

    indices = np.random.choice(size, nb_indices, replace=False)# choisit k indices  entre 0 et size
    sparseSource = np.zeros(size)

    if (distribution == "uniform"):
        assert(minUniform <= maxUniform),  "minimum is bigger than maximum"
        sparseSource[indices] = rng.uniform(low=minUniform, high=maxUniform, size=nb_indices)

    elif (distribution == "normal"):
        sparseSource[indices] = rng.normal(loc=meanNormal, scale=varNormal, size=nb_indices)

    elif (distribution == "exponential"):
        sparseSource[indices] = rng.exponential(scale=scaleExp,size=nb_indices)  # create random real number between min and max with an exponential distribution
    else:
        raise Exception("this distribution does not exist")

    #return np.resize(sparseSource, (n, m))
    return sparseSource



def compute_phi(n,m, L):
    size = n * m
    sampling = rng.choice(size, L, replace=False)
    X = 2 * np.pi * sampling / size
    phi = nt.NUFFT.type2(X, size, isign=-1, eps=1e-3, real=True)
    return phi


def reconstruction_pgd(size, data_fid, regul, min_iter, tmax):

    pgd = PGD(data_fid, regul, show_progress=False, verbosity=size)

    pgd.fit(x0=np.zeros(size),
            stop_crit=(min_iter & pgd.default_stop_crit()) | pycos.MaxDuration(t=dt.timedelta(seconds=tmax)),
            track_objective=True,
            )
    return pgd.solution()

def compute_mse(I_true, I_pred):
    return np.square(np.subtract(I_true,I_pred)).mean()


def plot1(original, reconstructed):
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
    
    def positive_int(value):
        value = float(value)
        if value != int(value):
            raise argparse.ArgumentTypeError("%s is not an integer" % value)
        if value <= 0:
            raise argparse.ArgumentTypeError("%s is not a positive integer" % value)
        return int(value)

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

    sparse_mat = createSparceSource(n,m, threshold, distribution=args.d) 
    


    phi = compute_phi(n,m, n*m)

    
    y = phi(sparse_mat)
    lambda_factor = .2
    min_iterations = 3 * n * m
    tmax = 30
    lambda_ = lambda_factor * np.linalg.norm(phi.adjoint(y), np.infty)
    min_iter = pycos.MaxIter(n=min_iterations)
    data_fid = .5 * nm.SquaredL2Norm().asloss(y) * phi  # F
    regul = lambda_ * nm.L1Norm()  # G

    sol = reconstruction_pgd(n*m, data_fid, regul, min_iter, tmax)
    

    mse = compute_mse(sparse_mat, sol)
    print(mse)
    plot1(sparse_mat, sol)

if __name__ == "__main__":
    main()
    
