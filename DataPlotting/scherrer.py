import numpy as np
import pandas as pd
import scipy.optimize as so
import scipy.stats as ss
import scipy.special as ssp
import matplotlib.pyplot as plt
import emcee
import tarmac
import numba as nb


@nb.jit
def deg_to_rad(a):
    return a*np.pi/180


@nb.jit
def scherrer_fwhm(center, fwhm, sto_fwhm):
    K = 0.94
    L = 1.54056
    theta = deg_to_rad(center/2)
    beta = deg_to_rad(fwhm - sto_fwhm)
    return K*L/(deg_to_rad(np.sqrt(fwhm**2-sto_fwhm**2))*np.cos(deg_to_rad(center/2)))


@nb.jit
def scherrer_fwhm_uncertainty(center, fwhm, sto_fwhm, center_uncertainty, fwhm_uncertainty, sto_fwhm_uncertainty):
    K = 0.94
    L = 1.54056
    theta = deg_to_rad(center/2)
    d_theta = deg_to_rad(center_uncertainty/2)
    beta = fwhm - sto_fwhm
    d_beta = np.sqrt(fwhm_uncertainty**2 + sto_fwhm_uncertainty**2)
    return np.sqrt(K*L/(beta**2 * np.cos(theta))*d_beta**2 + (K*L*np.sin(theta)/(beta*np.cos(theta)**2))*d_theta**2)


@nb.jit
def lorentz(x, A, c, w, offset):
    return offset+A/(1+((c-x)/(0.5*w))**2)


@nb.jit
def gaussian(x, A, c, w, offset):
    return offset+A*np.exp(-(((x-c)**2)/((w**2)/4*np.log(2))))


@nb.jit
def jeffreys_log_prior(sigma):
    return -np.log(sigma)


@nb.jit
def flat_log_prior(value, bounds):
    if bounds[0] < value < bounds[1]:
        return 0
    else:
        return -np.inf


def cov_c_w(chains, ndim):
    parameters = chains.reshape((-1, ndim)).T
    return np.cov(parameters)[2, 3]


# @nb.jit
def log_likelihood(theta, x, y):
    A, c, w, offset, sigma = theta
    return -0.5*np.sum(np.log(2*np.pi*3**2) + (y - lorentz(x, A, c, w, offset))**2/3**2)


@nb.jit
def log_prior(theta, bounds):
    A, c, w, offset, sigma = theta
    if sigma <= 0:
        return -np.inf
    else:
        return jeffreys_log_prior(sigma)+flat_log_prior(A, bounds[0])+flat_log_prior(c, bounds[1])+flat_log_prior(w, bounds[2])+flat_log_prior(offset, bounds[3])


# @nb.jit
def log_posterior(theta, x, y, bounds):
    return log_prior(theta, bounds) + log_likelihood(theta, x, y)


def fit_curve_MCMC(x, y, bounds, inits):

    ndim = len(inits)
    nwalkers = 50
    nburn = 4000
    nsteps = 1000
    ntemps = 20

    starting_guesses = np.array([[[ss.norm(init, init*0.05).rvs() for init in inits] for _ in range(nwalkers)] for _ in range(ntemps)])
    # starting_guesses = np.array([[ss.norm(init, init*0.05).rvs() for init in inits] for _ in range(nwalkers)])


    # Initialize MCMC
    # sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[x, y, bounds])
    sampler = emcee.PTSampler(ntemps, nwalkers, ndim, log_likelihood, log_prior, loglargs=(x, y), logpargs=([bounds]))
    sampler.run_mcmc(starting_guesses, nsteps+nburn)
    # chain = sampler.chain[:, nburn:, :]
    chain = sampler.chain[0, :, nburn:, :]
    return chain


def extract_pars(chains, ndim):
    chains = chains.reshape((-1, ndim))
    parameters_mean = np.mean(chains, axis=0)
    parameters_2std = 2*np.std(chains, axis=0)

    return parameters_mean, parameters_2std


def plot_data_and_fit(ax, x, y, chains, label, ndim=5):

    pars_mean, pars_2std = extract_pars(chains, ndim)

    ax.plot(x, y, '-b')
    x_fit = np.linspace(np.min(x), np.max(x), 1000)

    print(pars_mean, pars_2std)
    y_fit = lorentz(x_fit, pars_mean[0], pars_mean[1], pars_mean[2], pars_mean[3])

    # print('A = {}$\\pm${}, c = {}$\\pm${}, w = {}$\\pm${}, offset = {}$\\pm${}'.format(pars_mean[0], pars_2std[0],
    #                                                                                pars_mean[1], pars_2std[1],
    #                                                                                pars_mean[2], pars_2std[2],
    #                                                                                pars_mean[3], pars_2std[3]))

    ax.plot(x_fit, y_fit, '-r')
    ax.set_xlabel('$2\\theta$ (deg.)')
    ax.set_ylabel('Intensity (a. u.)')
    ax.set_yscale('log', nonposy='clip')

    return


def fit_film(data: pd.DataFrame) -> dict:

    chains_film = {}
    print('Fitting 002 YBCO peaks...')
    for index, tth, i, key in zip([0, 1, 2, 3],
                                  ['ag_tth', '3nm_tth', '7nm_tth', '20nm_tth'],
                                  ['ag_i', '3nm_i', '7nm_i', '20nm_i'],
                                  ['ag', '3nm', '7nm', '20nm']):

        print(key)

        mask = np.logical_and(data[tth] <= 20, data[tth] >= 10)
        x = data[tth][mask].values
        y = data[i][mask].values

        chains_film[key] = fit_curve_MCMC(x,
                                          y,
                                          bounds=((0, 1e4), (10, 20), (0, 1), (0, 1e2), (0.1, 10)),
                                          inits=(1e2, 15, 0.5, 10, 5))

        break

    return chains_film


def fit_sto(data: pd.DataFrame) -> dict:

    chains_sto = {}

    # Fit STO peak
    print('Fitting STO peaks...')
    for index, tth, i, key in zip([0, 1, 2, 3],
                                  ['ag_tth', '3nm_tth', '7nm_tth', '20nm_tth'],
                                  ['ag_i', '3nm_i', '7nm_i', '20nm_i'],
                                  ['ag', '3nm', '7nm', '20nm']):

        mask = np.logical_and(data[tth] <= 22.8, data[tth] >= 22.5)
        x = data[tth][mask].values
        y = data[i][mask].values

        chains_sto[key] = fit_curve_MCMC(x,
                                         y,
                                         bounds=((1e3, 1e7), (22.5, 28), (0.01, 0.5), (0, 1e2), (0.1, 1e2)),
                                         inits=(1e5, 22.7, 0.2, 10, 5))

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        plot_data_and_fit(ax, x, y, chains_sto[key], key, ndim=5)
        break

    return chains_sto


def fit_peaks(data):
    # fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(24, 4))

    chains_film = fit_film(data)
    chains_sto = fit_sto(data)

    # fig = plt.figure(figsize=(10, 10))
    # tarmac.corner_plot(fig, chains, labels=['A', 'c', 'w', 'offset', 'sigma'])
    # fig = plt.figure(figsize=(10, 10))
    # tarmac.walker_trace(fig, chains, labels=['A', 'c', 'w', 'offset', 'sigma'])

    # plot_data_and_fit(ax[index], x, y, chains, key)
    # pars_mean, pars_2std = extract_pars(chains, ndim)

    # center_film[key] = pars_mean[1]
    # fwhm_film[key] = pars_mean[2]
    # center_film_uncertainty[key] = pars_2std[1]
    # fwhm_film_uncertainty[key] = pars_2std[2]

    # fig = plt.figure(figsize=(10, 10))
    # tarmac.corner_plot(fig, chains, labels=['A', 'c', 'w', 'offset', 'sigma'])
    # fig = plt.figure(figsize=(10, 10))
    # tarmac.walker_trace(fig, chains, labels=['A', 'c', 'w', 'offset', 'sigma'])

    # plot_data_and_fit(ax[index], x, y, chains, key)
    # pars_mean, pars_2std = extract_pars(chains, ndim)

    # fwhm_sto[key] = pars_mean[2]
    # fwhm_sto_uncertainty = pars_2std[2]

    return chains_film, chains_sto


@nb.jit
def d_spacing(theta, L=1.54056):
    """Calculate d-spacing from wavelenth L and bragg angle theta."""
    return L/(2*np.sin(deg_to_rad(theta)))


@nb.jit
def calculate_c_axis(chains, ndim=5):
    """Calculate average d-spacing and 2*standard deviation from a set of MCMC chains. Chains[:, 1] corresponds
    to the parameter which marks the 2Theta center of the peak."""
    _chains = chains.reshape((-1, ndim))
    d = d_spacing(_chains[:, 1]/2)
    return np.mean(d), 2*np.std(d)


if __name__ == '__main__':
    data = pd.read_csv('C:/Users/pdmurray/Desktop/peyton/Projects/YBCO_Getters/Data/Combined Datasets/XRD.csv',
                       names=['ag_tth', 'ag_i', '3nm_tth', '3nm_i', '7nm_tth',
                              '7nm_i', '20nm_tth', '20nm_i', 'ta_tth', 'ta_i'],
                       skiprows=2)

    # chains_film, chains_sto = fit_peaks(data, ndim=5)
    # chains_film = fit_film(data)

    # for key in chains_film.keys():
    #     d_mean, d_std = calculate_c_axis(chains_film[key])
    #     print('{} c-axis from (002) peak: {}±{} Å (2 std deviations)'.format(key, 2*d_mean, 2*d_std))

    chains_sto = fit_sto(data)
    for key, chains in chains_sto.items():
        d_mean, d_std = calculate_c_axis(chains)
        print('{} c-axis from STO (001) peak: {}±{} Å (2 std deviations)'.format(key, d_mean, d_std))

        fig = plt.figure(figsize=(10, 10))
        tarmac.corner_plot(fig, chains, labels=['A', 'c', 'w', 'offset', 'sigma'])
        fig = plt.figure(figsize=(10, 10))
        tarmac.walker_trace(fig, chains, labels=['A', 'c', 'w', 'offset', 'sigma'])

    plt.show()
