import matplotlib

# matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np
import pandas as pd
from scipy.misc import factorial
from scipy.stats import norm
from scipy.optimize import curve_fit
from pprint import pprint


def readData(fname):
	return pd.read_csv(fname, sep=',', skiprows=2, names=["T", "dT", "M", "dM", "T", "dT", "M", "dM", "T", "dT", "M", "dM", "T", "dT", "M", "dM"])

def maskData(data):

	T_AG, dT_AG, M_AG, dM_AG = data.iloc[:, 0].values, data.iloc[:, 1].values, data.iloc[:, 2].values, data.iloc[:, 3].values
	T_Gd3nm, dT_Gd3nm, M_Gd3nm, dM_Gd3nm = data.iloc[:, 4].values, data.iloc[:, 5].values, data.iloc[:, 6].values, data.iloc[:, 7].values
	T_Gd7nm, dT_Gd7nm, M_Gd7nm, dM_Gd7nm = data.iloc[:, 8].values, data.iloc[:, 9].values, data.iloc[:, 10].values, data.iloc[:, 11].values
	T_Gd20nm, dT_Gd20nm, M_Gd20nm, dM_Gd20nm = data.iloc[:, 12].values, data.iloc[:, 13].values, data.iloc[:, 14].values, data.iloc[:, 15].values

	T_Gd7nm, dT_Gd7nm, M_Gd7nm, dM_Gd7nm = np.delete(T_Gd7nm, 10), np.delete(dT_Gd7nm, 10), np.delete(M_Gd7nm, 10), np.delete(dM_Gd7nm, 10)	#Remove single spurious point in data, possibly shouldn't do this?

	return T_AG, dT_AG, M_AG, dM_AG, T_Gd3nm, dT_Gd3nm, M_Gd3nm, dM_Gd3nm, T_Gd7nm, dT_Gd7nm, M_Gd7nm, dM_Gd7nm, T_Gd20nm, dT_Gd20nm, M_Gd20nm, dM_Gd20nm

def momentToMagnetization(data):

	YBCOVolumes = {"AG":0.000003604876892955276361594551603,"3nm":0.000002469397697894593047952244882, "7nm":0.000002609017789566844203226052770, "20nm":9.102226931866367015984866525E-7}

	T_AG, dT_AG, M_AG, dM_AG, T_Gd3nm, dT_Gd3nm, M_Gd3nm, dM_Gd3nm, T_Gd7nm, dT_Gd7nm, M_Gd7nm, dM_Gd7nm, T_Gd20nm, dT_Gd20nm, M_Gd20nm, dM_Gd20nm = data
	ret = T_AG, dT_AG, M_AG/YBCOVolumes["AG"], dM_AG, T_Gd3nm, dT_Gd3nm, M_Gd3nm/YBCOVolumes["3nm"], dM_Gd3nm, T_Gd7nm, dT_Gd7nm, M_Gd7nm/YBCOVolumes["7nm"], dM_Gd7nm, T_Gd20nm, dT_Gd20nm, M_Gd20nm/YBCOVolumes["20nm"], dM_Gd20nm
	# ret = T_AG, dT_AG, M_AG, dM_AG, T_Gd3nm, dT_Gd3nm, M_Gd3nm, dM_Gd3nm, T_Gd7nm, dT_Gd7nm, M_Gd7nm, dM_Gd7nm, T_Gd20nm, dT_Gd20nm, M_Gd20nm, dM_Gd20nm
	return ret

def shieldingFraction(data, H=10):
	T_AG, dT_AG, M_AG, dM_AG, T_Gd3nm, dT_Gd3nm, M_Gd3nm, dM_Gd3nm, T_Gd7nm, dT_Gd7nm, M_Gd7nm, dM_Gd7nm, T_Gd20nm, dT_Gd20nm, M_Gd20nm, dM_Gd20nm = data
	ret = T_AG, dT_AG, M_AG/(4*np.pi*H), dM_AG, T_Gd3nm, dT_Gd3nm, M_Gd3nm/(4*np.pi*H), dM_Gd3nm, T_Gd7nm, dT_Gd7nm, M_Gd7nm/(4*np.pi*H), dM_Gd7nm, T_Gd20nm, dT_Gd20nm, M_Gd20nm/(4*np.pi*H), dM_Gd20nm
	return ret

def normalizeMoment(data):
	T_AG, dT_AG, M_AG, dM_AG, T_Gd3nm, dT_Gd3nm, M_Gd3nm, dM_Gd3nm, T_Gd7nm, dT_Gd7nm, M_Gd7nm, dM_Gd7nm, T_Gd20nm, dT_Gd20nm, M_Gd20nm, dM_Gd20nm = data
	return T_AG, dT_AG, M_AG/np.abs(M_AG[0]), dM_AG/np.abs(M_AG[0]), T_Gd3nm, dT_Gd3nm, M_Gd3nm/np.abs(M_Gd3nm[0]), dM_Gd3nm/np.abs(M_Gd3nm[0]), T_Gd7nm, dT_Gd7nm, M_Gd7nm/np.abs(M_Gd7nm[0]), dM_Gd7nm/np.abs(M_Gd7nm[0]), T_Gd20nm, dT_Gd20nm, M_Gd20nm/np.abs(M_Gd20nm[0]), dM_Gd20nm//np.abs(M_Gd20nm[0])

def normalizeMomentByMass(data, units="g"):
	T_AG, dT_AG, M_AG, dM_AG, T_Gd3nm, dT_Gd3nm, M_Gd3nm, dM_Gd3nm, T_Gd7nm, dT_Gd7nm, M_Gd7nm, dM_Gd7nm, T_Gd20nm, dT_Gd20nm, M_Gd20nm, dM_Gd20nm = data

	if units == "g":
		mass = {"AG": 86.72e-3, "Gd3nm": 59.41e-3, "Gd7nm": 62.77e-3, "Gd20nm": 21.90e-3}	#Masses in g
	elif units == "mg":
		mass = {"AG":86.72, "Gd3nm":59.41, "Gd7nm":62.77, "Gd20nm":21.90}	#Masses in mg
	else:
		raise ValueError("Invalid units.")

	return T_AG, dT_AG, M_AG/mass["AG"], dM_AG/mass["AG"], T_Gd3nm, dT_Gd3nm, M_Gd3nm/mass["Gd3nm"], dM_Gd3nm/mass["Gd3nm"], T_Gd7nm, dT_Gd7nm, M_Gd7nm/mass["Gd7nm"], dM_Gd7nm/mass["Gd7nm"], T_Gd20nm, dT_Gd20nm, M_Gd20nm/mass["Gd20nm"], dM_Gd20nm/mass["Gd20nm"]


def calculateDerivative(T, M, order=1):
	return

def toMicroEmu(data):
	T_AG, dT_AG, M_AG, dM_AG, T_Gd3nm, dT_Gd3nm, M_Gd3nm, dM_Gd3nm, T_Gd7nm, dT_Gd7nm, M_Gd7nm, dM_Gd7nm, T_Gd20nm, dT_Gd20nm, M_Gd20nm, dM_Gd20nm = data
	return T_AG, dT_AG, 1e6*M_AG, 1e6*dM_AG, T_Gd3nm, dT_Gd3nm, 1e6*M_Gd3nm, 1e6*dM_Gd3nm, T_Gd7nm, dT_Gd7nm, 1e6*M_Gd7nm, 1e6*dM_Gd7nm, T_Gd20nm, dT_Gd20nm, 1e6*M_Gd20nm, 1e6*dM_Gd20nm

def generateFDKernel(nthDerivative, nterms):
	return np.array([(((factorial(nthDerivative) ** 2) / (k * factorial(nthDerivative - k) * factorial(nthDerivative + k))) * (-1) ** (k + 1)) if k!=0 else 0 for k in range(-nterms, nterms+1)])

def bundle_data(T_AG, dT_AG, M_AG, dM_AG, T_Gd3nm, dT_Gd3nm, M_Gd3nm, dM_Gd3nm, T_Gd7nm, dT_Gd7nm, M_Gd7nm, dM_Gd7nm, T_Gd20nm, dT_Gd20nm, M_Gd20nm, dM_Gd20nm):
	T = {"AG":T_AG, "3nm":T_Gd3nm, "7nm":T_Gd7nm, "20nm":T_Gd20nm}
	dT = {"AG":dT_AG, "3nm":dT_Gd3nm, "7nm":dT_Gd7nm, "20nm":dT_Gd20nm}
	M = {"AG":M_AG, "3nm":M_Gd3nm, "7nm":M_Gd7nm, "20nm":M_Gd20nm}
	dM = {"AG":dM_AG, "3nm":dM_Gd3nm, "7nm":dM_Gd7nm, "20nm":dM_Gd20nm}
	return T, dT, M, dM

def extract_and_export():
	# base = 'C:/Users/pdmurray/Desktop/peyton/temp-ybco/Data/Combined Datasets/'
	base = 'C:/Users/pdmurray/Desktop/peyton/temp-ybco/Data/Combined Datasets/'
	fname = base+'MPMS.csv'
	data = readData(fname)
	data = maskData(data)
	# data = normalizeMoment(data)
	data = bundle_data(*data)
	return data

def fit_erf(T, M):

	dist = lambda x, A, loc1, scale1, loc2, scale2: (A*norm.cdf(x, loc=loc1, scale=scale1)+(1-A)*norm.cdf(x, loc=loc2, scale=scale2)) - 1
	popt, pcov = curve_fit(dist, T, M, p0=[0.5, 50, 10, 80, 3])

	fit_M = dist(T, *popt)

	return fit_M, popt

if __name__ == "__main__":
	base = 'C:/Users/pdmurray/Desktop/peyton/temp-ybco/Data/Combined Datasets/'
	fname = base+'MPMS.csv'
	data = readData(fname)
	data = maskData(data)
	# data = normalizeMomentByMass(data, units="g")
	# data = toMicroEmu(data)
	data = normalizeMoment(data)
	T, dT, M, dM = bundle_data(*data)

	cmap = get_cmap('inferno')
	labels = {"AG":"As Grown", "3nm":"Gd (3 nm)", "7nm":"Gd (7 nm)", "20nm":"Gd (20 nm)"}
	# colors = {"AG":cmap(0.0), "3nm":cmap(0.3), "7nm":cmap(0.5), "20nm":cmap(0.6)}
	colors = {"AG": 'black', "3nm": 'saddlebrown', "7nm": 'darkgoldenrod', "20nm": "olivedrab"}

	fig = plt.figure(figsize=(10,8))
	axMain = fig.add_subplot(111)

	for sample in T.keys():
		if sample == "20nm":
			continue
		axMain.errorbar(T[sample], M[sample], yerr=dM[sample], xerr=dT[sample], fmt='o', color=colors[sample], label=labels[sample])
		axMain.plot(T[sample], M[sample], linestyle='-', marker=None, color=colors[sample])

		# fit_M, popt = fit_erf(T[sample], M[sample])
		# axMain.plot(T[sample], fit_M, marker=None, linestyle='-', color=colors[sample])
		# print("______{}______".format(sample))
		# pprint(popt)


		# axMain.plot(T[sample], popt[0]*norm.cdf(T[sample], loc=popt[1], scale=popt[2])-1, marker=None, linestyle=':', color=colors[sample])
		# axMain.plot(T[sample], (1-popt[0])*norm.cdf(T[sample], loc=popt[3], scale=popt[4])-1, marker=None, linestyle=':', color=colors[sample])

	# axMain.errorbar(T_AG, M_AG, yerr=dM_AG, xerr=dT_AG, fmt='o', color=colors["AG"], label="As Grown")
	# axMain.errorbar(T_Gd3nm, M_Gd3nm, yerr=dM_Gd3nm, xerr=dT_Gd3nm, fmt='-o', color=colors["3nm"], label="Gd (3 nm)")
	# axMain.errorbar(T_Gd7nm, M_Gd7nm, yerr=dM_Gd7nm, xerr=dT_Gd7nm, fmt='-o', color=colors["7nm"], label="Gd (7 nm)")
	# axMain.errorbar(T_Gd20nm, M_Gd20nm, yerr=dM_Gd20nm, xerr=dT_Gd20nm, fmt='-o', color=colors["20nm"], label="Gd (20 nm)")

	textSizeMain = 'xx-large'
	textSizeInset = 'small'
	textSizeMainLabel = "xx-large"

	axMain.set_xlabel("Temperature (K)", size=textSizeMain)
	# axMain.set_ylabel("Normalized Moment ($\mathrm{\mu emu/g}$)", size=textSizeMain)
	axMain.set_ylabel(r"$M/|M(5\,\mathrm{K})|$", size=textSizeMain)
	axMain.legend(loc=(0.8, 0.1))
	fig.tight_layout()

	plt.savefig('Tc_normalized.svg', bbox_inches='tight')

	plt.show()