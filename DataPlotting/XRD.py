import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np
import pandas as pd
from scipy.ndimage import convolve
from scipy.optimize import curve_fit, differential_evolution
from scipy.stats import norm, uniform
from emcee import EnsembleSampler, PTSampler
import tarmac
from multiprocessing import Pool
from time import time
import dill


def readData(fname):
	return pd.read_csv(fname, sep=',', skiprows=2, names=["$2\Theta$ (deg.)", "Intensity (a.u.)", "$2\Theta$ (deg.)", "Intensity (a.u.)", "$2\Theta$ (deg.)", "Intensity (a.u.)", "$2\Theta$ (deg.)", "Intensity (a.u.)", "$2\Theta$ (deg.)", "Intensity (a.u.)"])

def lorentzian(theta, tth):
	tth0, w, A, B = theta[:4]
	wsq = w**2
	return B+A*wsq/(wsq + (tth - tth0)**2)

def gaussian(tth, tth0, w, A, B):
	return B+A*np.exp(-(((tth-tth0)**2)/((w**2)/4*np.log(2))))

# def doublelorentz(tth, tth0_1, w_1, A_1, tth0_2, w_2, A_2, B):
	# return B+lorentzian(tth, tth0_1, w_1, A_1, 0)+lorentzian(tth, tth0_2, w_2, A_2, 0)

# def doublegausslorentz(tth, tth0_1, w_1, A_1, eta_1, tth0_2, w_2, A_2, eta_2, B):
	# return B+gausslorentz(tth, tth0_1, w_1, A_1, 0, eta_1)+gausslorentz(tth, tth0_2, w_2, A_2, 0, eta_2)

# def gausslorentz(tth, tth0, w, A, B, eta):
	# return B+eta*lorentzian(tth, tth0, w, A, 0)+(1-eta)*gaussian(tth, tth0, w, A, 0)

def plotFits(xdata, ydata, colors, ax, popsize=100, maxiter=10000):

	peaks = []
	widths = []
	scalings = []
	offsets = []
	weights = []
	count = 0
	for xvals, yvals, color in zip(xdata, ydata, colors):

		yvals = yvals[np.where(np.logical_not(np.isnan(xvals)))]
		xvals = xvals[np.where(np.logical_not(np.isnan(xvals)))]

		x = xvals[np.where(np.logical_and(12<xvals, xvals<18))]
		y = yvals[np.where(np.logical_and(12<xvals, xvals<18))]


		if count != 2:
			lsqlorentzian = lambda pars: np.sum((lorentzian(x, *pars) - y)**2)
			lsqgaussian = lambda pars: np.sum((gaussian(x, *pars) - y)**2)
			lsqgausslorentz = lambda pars: np.sum((gausslorentz(x, *pars) - y)**2)

			# res = differential_evolution(func=lsqlorentzian, bounds=((12,18), (0,6), (0,1e12), (0,1e12)), tol=0.001, popsize=30)
			# res = differential_evolution(func=lsqgaussian, bounds=((14,16), (0.00001,6), (0,1e12), (0,1e12)), popsize=30)
			res = differential_evolution(func=lsqgausslorentz, bounds=((12,18), (0.00001,6), (0,1e12), (0,1e12), (0,1)), popsize=popsize, maxiter=maxiter)
			popt = res.x

			peaks.append(popt[0])
			widths.append(popt[1])
			scalings.append(popt[2])
			offsets.append(popt[3])
			weights.append(popt[4])

			if ax is not None:
				# ax.plot(x, lorentzian(x, *popt), color=color)
				# ax.plot(x, gaussian(x, *popt), color=color)
				ax.plot(x, gausslorentz(x, *popt), color=color)

				# ax.text(popt[0], lorentzian(popt[0], *popt), "{:03.3f}".format(popt[0]), color=color, horizontalalignment='center')
				# ax.plot([popt[0], popt[0]], [ax.get_ylim()[0], lorentzian(popt[0], *popt)], '-', color=color, alpha=0.4)

				# ax.text(popt[0], gaussian(popt[0], *popt), "{:03.3f}".format(popt[0]), color=color, horizontalalignment='center')
				# ax.plot([popt[0], popt[0]], [ax.get_ylim()[0], gaussian(popt[0], *popt)], '-', color=color, alpha=0.4)

				ax.text(popt[0], gausslorentz(popt[0], *popt), "{:03.3f}".format(popt[0]), color=color, horizontalalignment='center')
				ax.plot([popt[0], popt[0]], [ax.get_ylim()[0], gausslorentz(popt[0], *popt)], '-', color=color, alpha=0.4)

		else:
			lsqdoublelorentz = lambda pars: np.sum((doublelorentz(x, *pars)-y)**2)
			lsqdoublegausslorentz = lambda pars: np.sum((doublegausslorentz(x, *pars)-y)**2)

			# res = differential_evolution(func=lsqdoublelorentz, bounds=((12,18), (0,6), (0,1e12), (12,18), (0,6), (0,1e12), (0,1e12)), tol=0.001, popsize=30)
			res = differential_evolution(func=lsqdoublegausslorentz, bounds=((12,18), (0.00001,6), (0,1e12), (0,1), (12,18), (0.00001,6), (0,1e12), (0,1), (0,1e12)), popsize=popsize, maxiter=maxiter)
			popt = res.x
			# ax.plot(x, doublelorentz(x, *popt), color=color)
			# ax.text(popt[0], doublelorentz(popt[0], *popt), "{:03.3f}".format(popt[0]), color=color, horizontalalignment='center')
			# ax.text(popt[3], doublelorentz(popt[3], *popt), "{:03.3f}".format(popt[3]), color=color, horizontalalignment='center')
			# ax.plot([popt[0], popt[0]], [ax.get_ylim()[0], doublelorentz(popt[0], *popt)], '-', color=color, alpha=0.4)
			# ax.plot([popt[3], popt[3]], [ax.get_ylim()[0], doublelorentz(popt[3], *popt)], '-', color=color, alpha=0.4)

			peaks += [popt[0], popt[4]]
			widths += [popt[1], popt[5]]
			scalings += [popt[2], popt[6]]
			offsets.append(popt[8])
			weights += [popt[3], popt[7]]

			if ax is not None:
				ax.plot(x, doublegausslorentz(x, *popt), color=color)
				ax.text(popt[0], doublegausslorentz(popt[0], *popt), "{:03.3f}".format(popt[0]), color=color, horizontalalignment='center')
				ax.text(popt[4], doublegausslorentz(popt[4], *popt), "{:03.3f}".format(popt[4]), color=color, horizontalalignment='center')
				ax.plot([popt[0], popt[0]], [ax.get_ylim()[0], doublegausslorentz(popt[0], *popt)], '-', color=color, alpha=0.4)
				ax.plot([popt[4], popt[4]], [ax.get_ylim()[0], doublegausslorentz(popt[4], *popt)], '-', color=color, alpha=0.4)

		count += 1

	return [peaks, widths, scalings, offsets, weights]

def main(fname):


	cmap = get_cmap('inferno')
	# colors = [cmap(0.0), cmap(0.3), cmap(0.5), cmap(0.6)]
	colors = ['black','saddlebrown','darkgoldenrod',"olivedrab"]

	data = readData(fname)
	xdata, ydata, ydata_raw = reduceData(data, scaling=[2e3, 6e1, 1e1, 0.15, 0.000001])

	fig = plt.figure(figsize=(10, 6))
	axMain = fig.add_subplot(111)

	# Plot the data
	axMain.plot(xdata[0], ydata[0], linestyle='-', color=colors[0], label="As Grown")
	axMain.plot(xdata[1], ydata[1], linestyle='-', color=colors[1], label="Gd (3 nm)")
	axMain.plot(xdata[2], ydata[2], linestyle='-', color=colors[2], label="Gd (7 nm)")
	axMain.plot(xdata[3], ydata[3], linestyle='-', color=colors[3], label="Gd (20 nm)")
	# axMain.plot(xdata[4], ydata[4], linestyle='-', color='k', label="Ta (20 nm)")

	axMain.set_yscale('log', nonposy='mask')

	axMain.set_xlim([10, 60])
	axMain.set_ylim([0.6, 3e9])

	textSizeMain = 'xx-large'
	textSizeInset = 'small'
	textSizeMainLabel = "xx-large"

	# axMain.set_xlabel(r"$2\Theta$ (deg.)", size=textSizeMain)
	# axMain.set_ylabel(r"Intensity (a.u.)", size=textSizeMain)
	axMain.set_yticks([])
	# axMain.text(x=12.3, y=7e5, s=r"YBCO (002)", fontsize=textSizeMainLabel)

	# axMain.legend(loc=(0.8, 0.82))
	plt.tight_layout()

	# fits = plotFits(xdata, ydata, colors, ax=None)

	# plt.savefig('XRD.png', dpi=300, bbox_inches='tight')
	return

def reduceData(data, scaling=None):

	if scaling is None:
		scaling = [1,1,1,1,1]

	xdata_raw = [data.iloc[:, 0].values, data.iloc[:, 2].values, data.iloc[:, 4].values, data.iloc[:, 6].values, data.iloc[:, 8].values]
	ydata_raw = [data.iloc[:, 1].values, data.iloc[:, 3].values, data.iloc[:, 5].values, data.iloc[:, 7].values, data.iloc[:, 9].values]

	xdata, ydata = [], []

	# Only plot data for 10 < TTh < 60, since that's the only range where all samples were measured.
	for x, y in zip(xdata_raw, ydata_raw):
		xdata.append(x[np.where(np.logical_and(x >= 10, x <= 60))])
		ydata.append(y[np.where(np.logical_and(x >= 10, x <= 60))])

	ydata_smooth = []
	kernel = np.ones(3) / 3
	for i, data in enumerate(ydata):
		ydata_smooth.append(convolve(data, kernel, mode='nearest')*scaling[i])

	return xdata, ydata_smooth, ydata

def negativeLogLikelihood(*args):
	return -1*logLikelihood(*args)

def fitData(xdata, ydata, nwalkers=20, nburn=1000, nsteps=1000, model=lorentzian, bounds=None, inits=None, do_DE=True):

	ntemps = 20

	if bounds is None:
		bounds = [[13,17], [0,6], [0,1e12], [0,1e12], [0,None]]

	if inits is None:
		inits = [[15,2], [2,2], [1e6,1e3], [1e6,1e6], [1e1,1]]

	ndim = len(bounds)

	if do_DE:
		#Find starting positions for the walkers by minimizing negative log likelihood
		result = differential_evolution(negativeLogLikelihood, bounds=bounds, args=(xdata, ydata, model), popsize=15, maxiter=1000)
		# starting_guesses = [result.x + np.array([np.random.normal(init[0], init[1]) for init in inits]) for _ in range(nwalkers)]
		#TODO This is really misleading.... the init vector is weird here
	else:
		# starting_guesses = [inits[:,0] + np.array([np.random.normal(init[0], init[1]) for init in inits]) for _ in range(nwalkers)]
		# starting_guesses = [np.array([norm(init[0], init[1]).rvs() for init, bound in zip(inits, bounds)]) for _ in range(nwalkers)]
		# starting_guesses = [np.array([uniform(loc=bound[0], scale=bound[1]-bound[0]).rvs() for bound in inits]) for _ in range(nwalkers)]

		starting_guesses = np.array([[[norm(init[0], init[1]).rvs() for init, bound in zip(inits, bounds)] for _ in range(nwalkers)] for _ in range(ntemps)])


	#Run MCMC
	# sampler = EnsembleSampler(nwalkers=nwalkers, dim=ndim, lnpostfn=logPosterior, args=(xdata, ydata, bounds, model))
	sampler = PTSampler(ntemps=20, nwalkers=nwalkers, dim=ndim, logl=logLikelihood, logp=logPrior, loglargs=(xdata, ydata, model), logpargs=[bounds])
	sampler.run_mcmc(starting_guesses, nsteps+nburn)

	# np.save("pt_fit", sampler.chain)
	# emcee_chain = np.load("pt_fit.npy")
	# emcee_chain = emcee_chain.reshape((-1, nsteps+nburn, ndim))
	emcee_chain = sampler.chain[0, :, nburn:, :]
	# emcee_chain = sampler.chain[:, nburn:, :]

	return emcee_chain

def flatLogPrior(value, low, high):
	if low < value < high:
		return 0
	else:
		return -np.inf

def jeffreysPrior(value, cutoff=None):
	if value <= 0:
		return -np.inf

	if cutoff is None or value < cutoff:
		return -np.log(value)
	else:
		return -np.inf

def logPrior(theta, bounds):
	sigma = theta[-1]
	if sigma <= 0:
		return -np.inf
	else:
		return jeffreysPrior(sigma, cutoff=bounds[-1][-1]) + np.sum([flatLogPrior(parameter, *bound) for parameter, bound in zip(theta[:-1], bounds[:-1])])

def logPosterior(theta, xdata, ydata, bounds, model):
	return logPrior(theta, bounds) + logLikelihood(theta, xdata, ydata, model)

def logLikelihood(theta, xdata, ydata, model):
	sigma = theta[-1]
	return -0.5*np.sum(np.log(2*np.pi*sigma**2) + ((ydata - model(theta, xdata))**2)/sigma**2)

def runFits(fname):

	data = readData(fname)
	xdata, ydata, ydata_raw = reduceData(data, scaling=None)

	#Chop off the last entry, that's the Ta sample
	xdata = xdata[:-1]
	ydata = ydata[:-1]

	cmap = get_cmap('inferno')

	models = [lorentzian, lorentzian, lorentzian, lorentzian]
	bounds = [[[12,18], [0,6], [0,1e4], [0,1e4], [0,100]] for _ in models]

	# inits = np.array([list(zip(*fitLorentzian(xdata[i], ydata[i])))+[[10,2]] for i in range(len(models))])
	inits = np.array([[[15,0.5], [3,0.01], [400,10], [10,0.5], [1,0.001]] for _ in models])
	# inits = np.array([[[14,16], [-5,5], [-1000,1000], [-1000,1000], [2,8]] for _ in models])

	labels = [[r"$2\theta_0$", r"$w$", r"$A$", r"$B$", r"$\sigma$"] for _ in models]
	colors = [cmap(0.0), cmap(0.3), cmap(0.5), cmap(0.6)]

	chains = []

	nwalkers = 20
	nburn = 2000
	nsteps = 1000

	# pars = [[x, y, nwalkers, nburn, nsteps, m, b, i, False] for x, y, m, b, i in zip(xdata, ydata, models, bounds, inits)]
	# pool = Pool()
	# chains = list(pool.starmap(fitData, pars))
	# pool.close()
	# pool.join()


	for x, y, m, b, i in zip(xdata, ydata, models, bounds, inits):
		print('starting fit')
		chains.append(fitData(x, y, nwalkers=nwalkers, nburn=nburn, nsteps=nsteps, model=m, bounds=b, inits=i, do_DE=False))

	for i in range(len(models)):
		fig = plt.figure()
		tarmac.corner_plot(fig, chains[i], labels=labels[i])
		fig = plt.figure()
		tarmac.walker_trace(fig, chains[i], labels=labels[i])
		fig = plt.figure()
		plotDataAndFit(fig, xdata[i], ydata[i], models[i], chains[i], color=colors[i], ndim=len(labels[i]), labels=labels[i])

	return

def plotDataAndFit(fig, x, y, model, chain, color, ndim, labels):
	chain = chain.reshape((-1, ndim))
	parameters_mean = np.mean(chain, axis=0)
	parameters_2std = 2*np.std(chain, axis=0)

	txt = ""
	for l, m, d in zip(labels, parameters_mean, parameters_2std):
		txt += "{}:{:4.4f}±{:4.4f}   ".format(l,m,d)

	ax = fig.add_subplot(111)
	ax.plot(x, y, '-', color=color)
	yfits = model(parameters_mean, x)
	ax.plot(x, yfits, ':', alpha=0.3, color=color)
	ax.set_title(txt)
	return

def fitLorentzian(xdata, ydata):
	popt, pcov = curve_fit(lorentzian, xdata, ydata, p0=(15, 0.8, 200, 100), bounds=((12, 0, 0, 0), (18, 6, 1e4, 1e4)), maxfev=1000000)
	print("tth0: {}, w: {}, A: {}, B: {}\n".format(*popt))
	return popt, np.diagonal(pcov)

def testBayesianMachinery():
	x = np.linspace(0,100, 1000)
	y = 10 + 3*x + np.random.normal(0, 100, size=1000)

	model = lambda theta, x: theta[0]*x + theta[1]
	bounds = [[-1000,1000],[-1000,1000],[0,1000]]
	labels = ["slope", "intercept", "sigma"]

	nwalkers = 20
	nsteps = 2000
	nburn = 1000

	starting_guesses = [np.array([uniform(loc=bound[0], scale=bound[1]-bound[0]).rvs() for bound in bounds]) for _ in range(nwalkers)]

	sampler = EnsembleSampler(nwalkers=nwalkers, dim=len(bounds), lnpostfn=logPosterior, args=(x, y, bounds, model))
	sampler.run_mcmc(starting_guesses, nsteps+nburn)
	emcee_chain = sampler.chain[:, nburn:, :]

	plt.figure()
	plt.plot(x,y, 'ok', alpha=0.3)
	plt.plot(x, model(np.mean(emcee_chain.reshape((-1, len(labels))), axis=0), x), '-r')
	fig = plt.figure()
	tarmac.walker_trace(fig, emcee_chain, labels=labels)
	fig = plt.figure()
	tarmac.corner_plot(fig, emcee_chain, labels=labels)

if __name__ == "__main__":
	base = 'C:/Users/pdmurray/Desktop/peyton/Projects/YBCO_Getters/Data/Combined Datasets/'
	fname = base+'XRD.csv'

	# main(fname)
	runFits(fname)
	plt.show()


