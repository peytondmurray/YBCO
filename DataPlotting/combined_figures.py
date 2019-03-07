import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib.ticker as ticker

import MPMS, Resistivity
import numpy as np


def plot_both():
    T_vdp, vdp, normalized_vdp = Resistivity.extract_and_export()
    T_m, dT_m, M, dM = MPMS.extract_and_export()
    labels = {"AG":"As grown", "3nm":"Gd (3 nm)", "7nm":"Gd (7 nm)", "20nm":"Gd (20 nm)"}

    fig = plt.figure(figsize=(6,6))
    axes = fig.subplots(2, 1, sharex=True)
    fig.subplots_adjust(top=0.98,
                        bottom=0.095,
                        left=0.085,
                        right=0.98,
                        hspace=0.03,
                        wspace=0.003)

    axes[1].set_xlim([0, 100])
    axes[1].set_ylim([-0.05, 1.05])
    axes[1].set_yticks([0.0, 0.5, 1.0])
    axes[1].set_ylabel(r"$\rho/\rho(T=100 \,K)$", size="xx-large")
    axes[1].set_xlabel(r"Temperature (K)", size="xx-large")
    axes[0].set_ylabel(r"$M/|M(T=5 \,K)|$", size="xx-large")
    axes[0].set_yticks([-1.0, -0.5, 0.0])

    cmap = get_cmap('inferno')
    # cmap_colors = [0.0, 0.3, 0.5, 0.6]
    colors = {"AG": 'black', "3nm": 'saddlebrown', "7nm": 'darkgoldenrod', "20nm": "olivedrab"}

    for name in ["AG", "3nm", "7nm", "20nm"]:
        # axes[1].errorbar(T_vdp[name], normalized_vdp[name], xerr= marker='o', color=colors[name], label=labels[name])
        axes[1].plot(T_vdp[name], normalized_vdp[name], marker='o', color=colors[name], label=labels[name])
        if name != "20nm":
            axes[0].errorbar(T_m[name], M[name], xerr=dT_m[name], yerr=dM[name], fmt='o', color=colors[name])
            axes[0].plot(T_m[name], M[name], linestyle='-', color=colors[name])

    axes[1].legend()
    fig.tight_layout()

    # plt.savefig("R and M vs T.svg", bbox_inches="tight")

    plt.show()
    return


def plot_only_magnetization():
    T_m, dT_m, M, dM = MPMS.extract_and_export()
    labels = {"AG":"As grown", "3nm":"Gd (3 nm)", "7nm":"Gd (7 nm)", "20nm":"Gd (20 nm)"}

    fig = plt.figure(figsize=(10,6))
    axes = fig.add_subplot(111)

    axes.set_ylabel(r"$M (\mu emu)$", size="large")
    # axes.set_ylabel(r"$M/|M(T=5 \,K)|$", size="large")
    axes.set_xlabel(r"Temperature (K)", size="large")
    # axes.set_yticks([-1.0, -0.5, 0.0])

    cmap = get_cmap('inferno')
    # cmap_colors = [0.0, 0.3, 0.5, 0.6]
    colors = {"AG": 'black', "3nm": 'saddlebrown', "7nm": 'darkgoldenrod', "20nm": "olivedrab"}

    for name in ["AG", "3nm", "7nm", "20nm"]:
        # if name != "20nm":
        if name == "20nm":
            axes.errorbar(T_m[name], M[name], xerr=dT_m[name], yerr=dM[name], fmt='o', color=colors[name], markersize=3)
            axes.plot(T_m[name], M[name], linestyle='-', color=colors[name])

    axes.set_xlim([0, 100])
    axes.set_ylim([-6e-7, 6e-7])

    fig.set_size_inches(4, 4)
    fig.tight_layout()

    # plt.savefig("R and M vs T.svg", bbox_inches="tight")


def plot_dm_dT():
    T_m, dT_m, M, dM = MPMS.extract_and_export()
    labels = {"AG":"As grown", "3nm":"Gd (3 nm)", "7nm":"Gd (7 nm)", "20nm":"Gd (20 nm)"}

    fig = plt.figure(figsize=(10, 6))
    axes = fig.add_subplot(111)

    axes.set_ylabel(r"$M/|M(T=5 \,K)|$", size="large")
    axes.set_xlabel(r"Temperature (K)", size="large")
    axes.set_yticks([-1.0, -0.5, 0.0])

    cmap = get_cmap('inferno')
    # cmap_colors = [0.0, 0.3, 0.5, 0.6]
    colors = {"AG": 'black', "3nm": 'saddlebrown', "7nm": 'darkgoldenrod', "20nm": "olivedrab"}

    for name in ["AG", "3nm", "7nm", "20nm"]:
        if name != "20nm":
            t_diff, m_diff = differentiate_M(T_m[name], M[name])
            axes.plot(t_diff, m_diff, linestyle='-', color=colors[name])
            axes.set_title(name)

    axes.set_xlim([5, 100])

    fig.set_size_inches(4, 4)
    fig.tight_layout()

    # plt.savefig("R and M vs T.svg", bbox_inches="tight")
    return


def differentiate_M(T, M):
    return T[1:-1], (M[2:]-M[:-2])/(T[2:]-T[:-2])


def print_Tc():
    T_m, dT_m, M, dM = MPMS.extract_and_export()
    for name in ["AG", "3nm", "7nm"]:

        # _T, _M = determine_Tc(T_m[name], M[name])

        Tc = interpolate_target(T_m[name], M[name], -0.5)

        print('{}: Tc = {}, M = {}'.format(name, Tc, 0.5))
    return


def interpolate_target(x, y, y0):

    if np.min(y) > y0 or np.max(y) < y0:
        print('y0 outside of input array. Bounds: ({}, {})'.format(np.min(y), np.max(y)))

    i_low = np.nonzero(y < y0)[0][-1]
    i_high = np.nonzero(y > y0)[0][0]

    return x[i_low] + (y0 - y[i_low])*(x[i_high] - x[i_low])/(y[i_high] - y[i_low])


def show_spread():
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(111)
    colors = {"AG": 'black', "3nm": 'saddlebrown', "7nm": 'darkgoldenrod', "20nm": "olivedrab"}

    _spreads = {}

    T, _, M, _ = MPMS.extract_and_export()
    for name, thickness in zip(["AG", "3nm", "7nm"], [0, 3, 7]):
        Tc = interpolate_target(T[name], M[name], -0.5)
        Tlow = interpolate_target(T[name], M[name], -0.9)
        Thigh = interpolate_target(T[name], M[name], -0.1)
        ax1.plot(T[name], M[name], linestyle='-', marker='o', color=colors[name])
        ax1.plot(Tc, -0.5, 'or')
        ax1.plot([Tlow, Tlow], [-1.1, 0.1], linestyle=':', color=colors[name])
        ax1.plot([Thigh, Thigh], [-1.1, 0.1], linestyle=':', color=colors[name])

        _spreads[thickness] = Thigh-Tlow

    spreads = [_spreads[thickness] for thickness in [0, 3, 7]]

    fig = plt.figure(figsize=(5, 6))
    ax2 = fig.add_subplot(111)
    ax2.plot([0, 3, 7], spreads, '-ok')
    ax2.set_ylim(0, 35)
    ax2.set_xlim(-1, 21)
    ax3 = ax2.twinx()
    ax3.plot([0, 3, 7, 20], [84, 62, 36, 0], '-or')
    ax3.set_ylim(-5, 100)

    return


def get_resistivity_tc():

    T, vdp, normalized_vdp = Resistivity.extract_and_export()
    for name, thickness in zip(['AG', '3nm'], [0, 3]):
        Tc = interpolate_target(T[name], normalized_vdp[name], 0.5)
        print('Tc ({}) = {} K'.format(name, Tc))

    print

if __name__ == "__main__":

    # plot_both()
    plot_only_magnetization()
    # plot_dm_dT()
    # print_Tc()
    # get_resistivity_tc()
    # show_spread()
    plt.show()

