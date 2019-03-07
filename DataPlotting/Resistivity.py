import matplotlib

# matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import fsolve


def read_data(fname):
    names = ['T', 'H', 'I', 'R1', 'R2', 'dR1', 'dR2','T', 'H', 'I', 'R1', 'R2', 'dR1', 'dR2','T', 'H', 'I', 'R1', 'R2', 'dR1', 'dR2','T', 'H', 'I', 'R1', 'R2', 'dR1', 'dR2']
    df = pd.read_csv(fname, sep=',', skiprows=3, names=names)
    data = df.values
    data[data == '--'] = np.nan
    return data.astype(float)


def split_data(data):

    T1 = dict()
    T2 = dict()
    R1 = dict()
    R2 = dict()
    dR1 = dict()
    dR2 = dict()

    for i, name in zip([0,7,14,21], ["AG", "3nm", "7nm", "20nm"]):
        T1[name] = data[:,i][np.logical_not(np.isnan(data[:,i+3]))]
        T2[name] = data[:,i][np.logical_not(np.isnan(data[:,i+4]))]
        R1[name] = data[:,i+3][np.logical_not(np.isnan(data[:,i+3]))]
        R2[name] = data[:,i+4][np.logical_not(np.isnan(data[:,i+4]))]
        dR1[name] = data[:,i+5][np.logical_not(np.isnan(data[:,i+3]))]
        dR2[name] = data[:,i+6][np.logical_not(np.isnan(data[:,i+4]))]

    return T1, T2, R1, R2, dR1, dR2


def interpolate_data(T1, T2, R1, R2, dR1, dR2, step=1):

    interp_T = dict()
    interp_R1 = dict()
    interp_R2 = dict()
    interp_dR1 = dict()
    interp_dR2 = dict()

    for name in T1.keys():

        safe_temperature_min, safe_temperature_max = find_safe_interpolation_range(T1[name], T2[name], step)

        interp_T[name] = np.arange(safe_temperature_min, safe_temperature_max, step)
        interp_R1[name] = np.array(list(map(interp1d(T1[name], R1[name], kind='linear'), interp_T[name])))
        interp_R2[name] = np.array(list(map(interp1d(T2[name], R2[name], kind='linear'), interp_T[name])))
        interp_dR1[name] = np.array(list(map(interp1d(T1[name], dR1[name], kind='linear'), interp_T[name])))
        interp_dR2[name] = np.array(list(map(interp1d(T2[name], dR2[name], kind='linear'), interp_T[name])))

    return interp_T, interp_R1, interp_R2, interp_dR1, interp_dR2


def find_safe_interpolation_range(T1, T2, step):

    safe_temperature_range_min, safe_temperature_range_max = np.max([np.min(T1), np.min(T2)]), np.min([np.max(T1), np.max(T2)])
    rounded_min, rounded_max = (int(safe_temperature_range_min/step)+1)*step, (int(safe_temperature_range_max/step)-1)*step

    return rounded_min, rounded_max


def solve_vdp(T, R1, R2, dR1, dR2, thickness):

    vdp = dict()
    bad_vdp = dict()

    for name in T.keys():

        vdp[name] = np.zeros_like(R1[name])*np.nan
        bad_vdp[name] = np.zeros_like(R1[name])*np.nan

        for i in range(len(R1[name])):

            def f(Rs):
                return np.exp((-1*np.pi*R1[name][i])/Rs) + np.exp((-1*np.pi*R2[name][i])/Rs) - 1

            vdp[name][i] = fsolve(func=f, x0=R1[name][i]*np.pi*3/2)[0]


        vdp[name] *= thickness
        bad_vdp[name] *= thickness

    return vdp, bad_vdp


def take_cooling_curve(T1, T2, R1, R2, dR1, dR2):

    cooling_T1 = dict()
    cooling_T2 = dict()
    cooling_R1 = dict()
    cooling_R2 = dict()
    cooling_dR1 = dict()
    cooling_dR2 = dict()

    for name in T1.keys():
        cooling_T1[name] = T1[name][:np.nanargmin(T1[name])+1]
        cooling_T2[name] = T2[name][:np.nanargmin(T2[name])+1]
        cooling_R1[name] = R1[name][:np.nanargmin(T1[name])+1]
        cooling_R2[name] = R2[name][:np.nanargmin(T2[name])+1]
        cooling_dR1[name] = dR1[name][:np.nanargmin(T1[name])+1]
        cooling_dR2[name] = dR2[name][:np.nanargmin(T2[name])+1]

    return cooling_T1, cooling_T2, cooling_R1, cooling_R2, cooling_dR1, cooling_dR2


def extract_and_export():
    thickness = 100e-7 # [cm]

    base = 'C:/Users/pdmurray/Desktop/peyton/temp-ybco/Data/Combined Datasets/'
    fname = base+'Resistivity.csv'
    data = read_data(fname)
    data = split_data(data)
    data = take_cooling_curve(*data)
    data = interpolate_data(*data, step=1)
    vdp, bad_vdp = solve_vdp(*data, thickness)
    T, R1, R2, dR1, dR2 = data

    normalized_vdp = normalizeVdP(T, vdp, normalization_temperature=100)
    # T, normalized_vdp = stretchVdP(T, vdp, 100)

    return T, vdp, normalized_vdp


def normalizeVdP(T, input_vdp, normalization_temperature=None):

    indices = dict()
    if normalization_temperature is None:
        for name in T.keys():
            indices[name] = -1
    else:
        for name in T.keys():
            indices[name] = np.nanargmin(np.abs(T[name]-normalization_temperature))

    vdp = dict()
    for name in input_vdp.keys():
        vdp[name] = input_vdp[name]/input_vdp[name][indices[name]]
    return vdp


def stretchVdP(T, input_vdp, max_desired_temperature=100):

    for name in T.keys():
        input_vdp[name] = input_vdp[name][:np.nanargmin(np.abs(T[name]-max_desired_temperature))]
        T[name] = T[name][:np.nanargmin(np.abs(T[name]-max_desired_temperature))]

    vdp = dict()
    for name in input_vdp.keys():
        vdp[name] = (input_vdp[name] - np.min(input_vdp[name]))/(np.max(input_vdp[name])-np.min(input_vdp[name]))
    return T, vdp

if __name__ == "__main__":

    thickness = 100e-7 # [cm]

    base = 'C:/Users/pdmurray/Desktop/peyton/temp-ybco/Data/Combined Datasets/'
    fname = base+'Resistivity.csv'
    data = read_data(fname)
    data = split_data(data)
    data = take_cooling_curve(*data)
    data = interpolate_data(*data, step=1)
    vdp, bad_vdp = solve_vdp(*data, thickness)

    T, R1, R2, dR1, dR2 = data

    cmap = get_cmap("inferno")


    # fig = plt.figure(figsize=(12,8))
    # ax1 = fig.add_subplot(111)
    # yfmt = ticker.ScalarFormatter()
    # yfmt.set_powerlimits((0,0))
    # ax.errorbar(T, R1, yerr=dR1, fmt='or')
    # ax.errorbar(T, R2, yerr=dR2, fmt='ob')
    # ax1.plot(T["AG"], vdp["AG"], marker='o', color=cmap(0.0), label="As Grown")
    # ax1.plot(T["3nm"], vdp["3nm"], marker='o', color=cmap(0.3), label="Gd (3 nm)")
    # ax1.plot(T["7nm"], vdp["7nm"], marker='o', color=cmap(0.5), label="Gd (7 nm)")
    # ax1.plot(T["20nm"], vdp["20nm"], marker='o', color=cmap(0.6), label="Gd (20 nm)")
    # ax1.set_xlabel(r"Temperature (K)")
    # ax1.set_ylabel(r"Resistivity ($\Omega \cdot \mathrm{cm}$)")
    # ax1.get_yaxis().set_major_formatter(yfmt)
    # ax1.legend(loc=(0.8,0.02))
    # fig.tight_layout()

    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(111)
    yfmt = ticker.ScalarFormatter()
    yfmt.set_powerlimits((0,0))
    ax1.plot(T["AG"], vdp["AG"]/vdp["AG"][-1], marker='o', color=cmap(0.0), label="As Grown")
    ax1.plot(T["3nm"], vdp["3nm"]/vdp["3nm"][-1], marker='o', color=cmap(0.3), label="Gd (3 nm)")
    ax1.plot(T["7nm"], vdp["7nm"]/vdp["7nm"][-1], marker='o', color=cmap(0.5), label="Gd (7 nm)")
    ax1.plot(T["20nm"], vdp["20nm"]/vdp["20nm"][-1], marker='o', color=cmap(0.6), label="Gd (20 nm)")
    ax1.set_xlabel(r"Temperature (K)")
    ax1.set_ylabel(r"Normalized Resistivity $\rho/\rho(T=300 K)$")
    ax1.get_yaxis().set_major_formatter(yfmt)
    ax1.legend(loc=(0.8,0.02))
    ax1.set_xlim(0,300)
    fig.tight_layout()

    # plt.savefig("Resistivity_normalized.svg", bbox_inches='tight')

    index_of_100K = dict()
    for name in T.keys():
        index_of_100K[name] = np.nanargmin(np.abs(T[name]-100))

    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(111)
    yfmt = ticker.ScalarFormatter()
    yfmt.set_powerlimits((0,0))
    ax1.plot(T["AG"], vdp["AG"]/vdp["AG"][index_of_100K["AG"]], marker='o', color=cmap(0.0), label="As Grown")
    ax1.plot(T["3nm"], vdp["3nm"]/vdp["3nm"][index_of_100K["3nm"]], marker='o', color=cmap(0.3), label="Gd (3 nm)")
    ax1.plot(T["7nm"], vdp["7nm"]/vdp["7nm"][index_of_100K["7nm"]], marker='o', color=cmap(0.5), label="Gd (7 nm)")
    ax1.plot(T["20nm"], vdp["20nm"]/vdp["20nm"][index_of_100K["20nm"]], marker='o', color=cmap(0.6), label="Gd (20 nm)")
    ax1.set_xlabel(r"Temperature (K)")
    ax1.set_ylabel(r"Normalized Resistivity $\rho/\rho(T=100 K)$")
    ax1.get_yaxis().set_major_formatter(yfmt)
    ax1.legend(loc=(0.8,0.02))
    ax1.set_xlim(5,100)
    ax1.set_ylim(-0.05,1.05)
    fig.tight_layout()

    plt.savefig("Resistivity_normalized_100K.svg", bbox_inches='tight')

    # fig = plt.figure(figsize=(12,8))
    # ax1 = fig.add_subplot(111)
    # yfmt = ticker.ScalarFormatter()
    # yfmt.set_powerlimits((0,0))
    # ax1.plot(T["AG"], (vdp["AG"]-np.min(vdp["AG"]))/(np.max(vdp["AG"])-np.min(vdp["AG"])), marker='o', color=cmap(0.0), label="As Grown")
    # ax1.plot(T["3nm"], (vdp["3nm"]-np.min(vdp["3nm"]))/(np.max(vdp["3nm"])-np.min(vdp["3nm"])), marker='o', color=cmap(0.3), label="Gd (3 nm)")
    # ax1.plot(T["7nm"], (vdp["7nm"]-np.min(vdp["7nm"]))/(np.max(vdp["7nm"])-np.min(vdp["7nm"])), marker='o', color=cmap(0.5), label="Gd (7 nm)")
    # ax1.plot(T["20nm"], (vdp["20nm"]-np.min(vdp["20nm"]))/(np.max(vdp["20nm"])-np.min(vdp["20nm"])), marker='o', color=cmap(0.6), label="Gd (20 nm)")
    # ax1.set_xlabel(r"Temperature (K)")
    # ax1.set_ylabel(r"$(\rho-\rho_{min})/(\rho_{max} - \rho_{min}$")
    # ax1.get_yaxis().set_major_formatter(yfmt)
    # ax1.legend(loc=(0.8,0.02))
    # fig.tight_layout()



    #
    # plt.savefig('Tc_normalized .svg', dpi=300, bbox_inches='tight')
    #
    plt.show()