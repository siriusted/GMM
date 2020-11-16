import numpy as np
from scipy.special import iv
from scipy.optimize import fsolve

def vonmises_density(x,mu,kappa):
    """
    Calculate the von Mises density for a series x (a 1D numpy.array).
    Input : 
        x : a 1D numpy.array of size L
        mu : a 1D numpy.array of size n, the mean of the von Mises distributions
        kappa : a 1D numpy.array of size n, the dispersion of the von Mises distributions
    Output : 
        a (L x n) numpy array, L is the length of the series, and n is the size of the array containing the parameters. Each row of the output corresponds to a density
    """
    res = []
    for i in x:
        f = np.exp(kappa*np.cos(i-mu))
        n = 2*np.pi*iv(0,kappa)
        res.append(f/n)
    return(np.array(res))

def vonmises_pdfit(series):
    """
    Calculate the estimator of the mean and deviation of a sample, for a von Mises distribution
    Input : 
        series : a 1D numpy.array
    Output : 
        the estimators of the parameters mu and kappa of a von Mises distribution, in an list [mu, kappa]
    See https://en.wikipedia.org/wiki/Von_Mises_distribution 
    for more details on the von Mises distribution
    """
    s0 = np.mean(np.sin(series))
    c0 = np.mean(np.cos(series))
    mu = np.arctan2(s0,c0)
    var = 1-np.sqrt(s0**2+c0**2)
    k = lambda kappa: 1-iv(1,kappa)/iv(0,kappa)-var
    kappa = fsolve(k, 0.0)[0]
    return([mu,kappa])

def mixture_vonmises_pdfit(series, n=2, threshold=1e-3):
    """
    Find the parameters of a mixture of von Mises distributions, using an EM algorithm.
    Input : 
        series : a 1D numpy array, representing the stochastic perdioci process
        n : an int, the number of von Mises distributions in th emixture
        threshold : a float, correspond to the euclidean distance between the old parameters and the new ones
    Output : a (3 x n) numpy-array, containing the probability amplitude of the distribution, and the mu and kappa parameters on each line.
    """
    # initialise the parameters and the distributions
    pi = np.random.random(n)
    mu = np.random.vonmises(0.0,0.0,n)
    kappa = np.random.random(n)
    t = pi*vonmises_density(series,mu,kappa)
    s = np.sum(t, axis=1)
    t = (t.T/s).T
    thresh = 1.0
    # calculate and update the coefficients, untill convergence
    while thresh > threshold:
        new_pi = np.mean(t, axis=0)
        new_mu = np.arctan2(np.sin(series)@t,np.cos(series)@t)      
        c = np.cos(series)@(t*np.cos(new_mu))+np.sin(series)@(t*np.sin(new_mu))
        k = lambda kappa: (c-iv(1,kappa)/iv(0,kappa)*np.sum(t, axis=0)).reshape(n)
        new_kappa = fsolve(k, np.zeros(n))
        thresh = np.sum((pi-new_pi)**2+(mu-new_mu)**2+(kappa-new_kappa)**2)
        pi = new_pi
        mu = new_mu
        kappa = new_kappa
        t = pi*vonmises_density(series,mu,kappa)
        s = np.sum(t, axis=1)
        t = (t.T/s).T
    res = np.array([pi,mu,kappa])
    # in case there is no mixture, one fits the data using the estimators
    if n == 1:
        res = vonmises_pdfit(series)
        res = np.append(1.0,res)
        res = res.reshape(3,1)
    return(res)
