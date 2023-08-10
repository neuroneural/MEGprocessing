import numpy as np 
from ica import ica1, pca_whiten, infomax1


def ica_localv(x_raw, ncomp, verbose=False):
    '''
    Single modality Independent Component Analysis
    '''
    if verbose:
        print("Whitening data...")
    x_white, _, dewhite = pca_whiten(x_raw, ncomp)
    if verbose:
        print('x_white shape: %d, %d' % x_white.shape)
        print('d_white shape: %d, %d' % dewhite.shape)
        print("Done.")
    if verbose:
        print("Running INFOMAX-ICA ...")
    mixer, sources, unmixer = infomax1(x_white, verbose)
    mixer = np.dot(dewhite, mixer)

    scale = sources.std(axis=1).reshape((-1, 1))
    sources = sources / scale
    scale = scale.reshape((1, -1))
    mixer = mixer * scale

    if verbose:
        print("Done.")
    return (mixer, sources, unmixer)

def do_test_ica(Nobs=1000, Nvars=50000, Ncomp=100):

    # Nobs = 1000  # Number of observation

    # Nvars = 50000  # Number of variables
    # Ncomp = 100  # Number of components


    # Simulated true sources
    S_true = np.random.logistic(0, 1, (Ncomp, Nvars))
    # Simulated true mixing
    A_true = np.random.normal(0, 1, (Nobs, Ncomp))
    # X = AS
    X = np.dot(A_true, S_true)
    # add some noise
    X = X + np.random.normal(0, 1, X.shape)
    print(f'After generating signal:\nX:{X.shape}, A:{A_true.shape}, S: {S_true.shape}')

    mixer, sources, unmixer = ica_localv(X, Ncomp, verbose=True)
    print(f'After Doing ICA:\nmixer:{mixer.shape}, sources:{sources.shape}, unmixer: {unmixer.shape}')


if __name__ == "__main__":

    do_test_ica(Nobs=5124, Nvars=265505, Ncomp=200)

