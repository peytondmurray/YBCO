import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np
import pandas as pd
import matplotlib.patches as patches
import cmocean


class ReflData:
    """Given a PATH to reflectivity data, this class reads in PATH.datA and PATH.datB as pandas.DataFrame objects.
    Stores a bit of metadata as well."""

    def __init__(self, sample: str, fname_root: str):

        self.sample = sample
        self.fname_A = fname_root+'.datA'
        self.fname_D = fname_root+'.datD'

        self.pp = pd.read_csv(self.fname_A,
                              delim_whitespace=True,
                              names=['Q', 'dQ', 'R', 'dR', 'theory', 'fresnel'],
                              skiprows=3)
        self.mm = pd.read_csv(self.fname_D,
                              delim_whitespace=True,
                              names=['Q', 'dQ', 'R', 'dR', 'theory', 'fresnel'],
                              skiprows=3)

        return

def get_data():
    base = 'C:/Users/pdmurray/Desktop/peyton/Projects/YBCO_Getters/Data/'
    fnames = {'AG':base+'As Grown/YBCO_Gd0nm_PNR/YBCO_Gd0nm-1-steps.dat',
              '3nm':base+'Gd (3 nm)/YBCO_Gd3nm_PNR/YBCO_Gd3nm-1-steps.dat',
              '7nm':base+'Gd (7 nm)/YBCO_Gd7nm_PNR/YBCO_Gd7nm-1-steps.dat',
              '20nm':base+'Gd (20 nm)/YBCO_Gd20nm_PNR/YBCO_Gd20nm-1-steps.dat'}

    data = {'z':dict(), 'rho':dict(), 'irho':dict(), 'rhoM':dict(), 'theta':dict()}
    for key,fname in fnames.items():
        df = pd.read_csv(fname, delim_whitespace=True, names=['z','rho','irho','rhoM','theta'], skiprows=1)
        for column in df.columns:
            data[column][key] = df[column].values

    data['labels'] = {'AG':'As grown', '3nm':'Gd (3 nm)', '7nm':'Gd (7 nm)', '20nm':'Gd (20 nm)'}

    return data


def oxygen_content(a, b, c, rho):
    '''
    Inputs: lattice parameters in Å, and the measured SLD in 1e-6/Å^2
    Outputs number of oxygen atoms per unit cell

    Constants taken from neutron coherent scattering length table: https://ncnr.nist.gov/instruments/magik/Periodic.html
    '''
    return (a*b*c*rho*1e-6 - (7.750 + 2*5.070 + 3*7.718)*1e-5)/5.805e-5


def calculate_oxygen(data):

    c_axis = {'AG':11.68, '3nm':11.68, '7nm':11.68, '20nm':11.68}
    # c_axis = {'AG':11.678, '3nm':11.7163, '7nm':11.7558, '20nm':12.2271}
    oxygen = dict()
    for key in data['labels'].keys():
        oxygen[key] = oxygen_content(3.82, 3.89, c_axis[key], data['rho'][key])

    return oxygen


def plot_data(data, colors):

    fig = plt.figure(figsize=(12,10))
    ax = plt.subplot2grid((4,1), (0,0), rowspan=3)
    for key in ['AG', '3nm', '7nm', '20nm']:
        ax.plot(data['z'][key]/10, data['rho'][key], marker=None, linestyle='-', color=colors[key], label=data['labels'][key])
        # ax.plot(data['z'][key], data['irho'][key], marker=None, linestyle='--', color=colors[key])
        # ax.plot(data['z'][key], data['rhoM'][key], marker=None, linestyle='--', color=colors[key])
        # ax.fill_between(data['z'][key], 0, data['rhoM'][key], color='k', alpha=0.3)

    # ax.plot([-1000,3000],[4.182,4.182], linestyle=':', marker=None, color='k')
    # ax.plot([-1000,3000],[3.403,3.403], linestyle=':', marker=None, color='k')
    # ax.plot([-1000,3000],[4.662,4.662], linestyle=':', marker=None, color='r')

    ax.plot([-20,200],[4.7, 4.7], ':k')



    ax.set_xlim([-20,200])
    ax.set_ylim([-0.1, 5.2])
    ax.set_ylabel(ylabel=r'Re($\rho_N$) ($10^{-4}\,\mathrm{nm}^{-2}$)', size='xx-large')
    ax.add_patch(patches.Rectangle((-20, ax.get_ylim()[0]), 20, ax.get_ylim()[1]-ax.get_ylim()[0], fill=True, alpha=0.1))
    ax.legend(loc=(0.1,0.1))
    ax.set_xticklabels([])

    ax2 = plt.subplot2grid((4,1), (3,0), rowspan=1)
    for key in ['AG', '3nm', '7nm', '20nm']:
        # ax2.plot(data['z'][key]/10, data['irho'][key],marker=None, linestyle='-', color=colors[key])
        ax2.fill_between(data['z'][key]/10, 0, data['irho'][key], color=colors[key], alpha=0.4)

    ax2.add_patch(patches.Rectangle((-20, ax2.get_ylim()[0]), 20, ax2.get_ylim()[1]-ax2.get_ylim()[0], fill=True, alpha=0.1))
    ax2.set_xlabel(xlabel=r'$Z$ (nm)', size='xx-large')
    ax2.set_ylabel(ylabel=r'Im($\rho_N$) ($10^{-4}\,\mathrm{nm}^{-2}$)', size='xx-large')
    ax2.set_xlim([-20,200])


    fig.subplots_adjust(top=0.979,
                        bottom=0.07,
                        left=0.071,
                        right=0.977,
                        hspace=0.045,
                        wspace=0.2
                        )

    # plt.savefig('PNR.svg', bbox_inches='tight')

    return


def plot_data_separately(data, colors):
    fig = plt.figure(figsize=(10,8))
    ax = fig.subplots(4, 1, sharex='all')

    cutoff_sub = {'AG':0, '3nm':0, '7nm':0, '20nm':0}
    cutoff_ybco = {'AG':92, '3nm':101, '7nm':111, '20nm':120}
    cutoff_gd = {'AG':None, '3nm':109, '7nm':120, '20nm':149}
    cutoff_au = {'AG':None, '3nm':119, '7nm':137, '20nm':168}

    rhomin = 0
    rhomax = 5.2
    zmin = -10
    zmax = 210

    for i, key in enumerate(['AG', '3nm', '7nm', '20nm']):
        # Divide by 10 to change from Å to nm
        ax[i].plot(data['z'][key]/10, data['rho'][key], marker=None, linestyle='-', color='k', label=r'$\mathrm{Re}(\rho_N)$')


        z = (data['z'][key]/10)
        rho = data['rho'][key]

        i_cutoff_sub = get_nearest(z, cutoff_sub[key])
        # ax[i].plot([z[i_cutoff_sub], z[i_cutoff_sub]],  [0, rho[i_cutoff_sub]], linestyle=':', color='k')
        # ax[i].fill_between(x=z[0:i_cutoff_sub], y1=rho[0:i_cutoff_sub], y2=0, facecolor='k', label=data['labels'][key], alpha=0.3)
        # ax[i].fill_between(x=z[0:i_cutoff_sub], y1=rho[0:i_cutoff_sub], y2=0, facecolor='k', label=data['labels'][key], alpha=0.3)
        ax[i].add_patch(patches.Rectangle((zmin, rhomin), z[i_cutoff_sub]-zmin, rhomax-rhomin, fill=True, facecolor='k', alpha=0.3))
        ax[i].text(175, 4, data['labels'][key], fontsize='xx-large')

        i_cutoff_ybco = get_nearest(z, cutoff_ybco[key])
        # ax[i].plot([z[i_cutoff_ybco], z[i_cutoff_ybco]], [0, rho[i_cutoff_ybco]], linestyle=':', color='k')
        # ax[i].fill_between(x=z[i_cutoff_sub:i_cutoff_ybco], y1=rho[i_cutoff_sub:i_cutoff_ybco], y2=0, facecolor=colors[key], label=data['labels'][key], alpha=0.3)
        # ax[i].fill_between(x=z[i_cutoff_sub:i_cutoff_ybco], y1=rho[i_cutoff_sub:i_cutoff_ybco], y2=0, facecolor=colors[key], label=data['labels'][key], alpha=0.3)
        ax[i].add_patch(patches.Rectangle((z[i_cutoff_sub], rhomin), z[i_cutoff_ybco]-z[i_cutoff_sub], rhomax-rhomin, fill=True, facecolor='r', alpha=0.3))

        if cutoff_gd[key] is not None:

            i_cutoff_gd = get_nearest(z, cutoff_gd[key])
            # ax[i].plot([z[i_cutoff_gd], z[i_cutoff_gd]], [0, rho[i_cutoff_gd]], linestyle=':', color='k')
            # ax[i].fill_between(x=z[i_cutoff_ybco:i_cutoff_gd], y1=rho[i_cutoff_ybco:i_cutoff_gd], y2=0, facecolor=colors[key], label=data['labels'][key], alpha=0.3)
            # ax[i].fill_between(x=z[i_cutoff_ybco:i_cutoff_gd], y1=rho[i_cutoff_ybco:i_cutoff_gd], y2=0, facecolor=colors[key], label=data['labels'][key], alpha=0.3)
            ax[i].add_patch(patches.Rectangle((z[i_cutoff_ybco], rhomin), z[i_cutoff_gd]-z[i_cutoff_ybco], rhomax-rhomin, fill=True, facecolor='gold', alpha=0.3))

        if cutoff_au[key] is not None:
            i_cutoff_au = get_nearest(z, cutoff_au[key])
            # ax[i].plot([z[i_cutoff_au], z[i_cutoff_au]], [0, rho[i_cutoff_au]], linestyle=':', color='k')
            # ax[i].fill_between(x=z[i_cutoff_gd:i_cutoff_au], y1=rho[i_cutoff_gd:i_cutoff_au], y2=0, facecolor=colors[key], label=data['labels'][key], alpha=0.3)
            # ax[i].fill_between(x=z[i_cutoff_gd:i_cutoff_au], y1=rho[i_cutoff_gd:i_cutoff_au], y2=0, facecolor=colors[key], label=data['labels'][key], alpha=0.3)
            ax[i].add_patch(patches.Rectangle((z[i_cutoff_gd], rhomin), z[i_cutoff_au]-z[i_cutoff_gd], rhomax-rhomin, fill=True, facecolor='b', alpha=0.3))

        # ax[i].fill_between(data['z'][key]/10, 0, data['irho'][key], color=colors[key], alpha=0.4)
        ax[i].plot(data['z'][key]/10, data['irho'][key], linestyle=':', color='k', label=r'$\mathrm{Im}(\rho_N)$')
        ax[i].set_xlim([zmin,zmax])
        ax[i].set_ylim([rhomin,rhomax])
        ax[i].axhline(y=4.7, color='k')

    ax[3].set_xlabel(xlabel=r'$Z$ (nm)', size='xx-large')
    ax[1].set_ylabel(ylabel=r'$\rho_N$ ($10^{-4}\,\mathrm{nm}^{-2}$)', size='xx-large')

    ax[3].legend()

    fig.subplots_adjust(top=0.981,
                        bottom=0.07,
                        left=0.059,
                        right=0.977,
                        hspace=0.046,
                        wspace=0.245)

    return


def get_nearest(arr, pt):
    return np.argmin(np.abs(arr-pt))


def plot_oxygen(data, oxygen, colors):

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    for key in ['AG', '3nm', '7nm', '20nm']:
        ax.plot(data['z'][key], oxygen[key], marker=None, linestyle='-', color=colors[key], label=data['labels'][key])

    ax.set_xlim([-200, 2000])
    ax.set_xlabel(xlabel=r'$Z$ (Å)', size='xx-large')
    ax.set_ylabel(ylabel=r'YBa$_2$Cu$_3$O$_x$', size='xx-large')
    ax.add_patch(patches.Rectangle((-200, ax.get_ylim()[0]), 200, ax.get_ylim()[1] - ax.get_ylim()[0], fill=True, alpha=0.1))
    ax.legend(loc=(0.82, 0.85))
    fig.tight_layout()
    # plt.savefig('PNR_oxygen.svg', bbox_inches='tight')
    return


def get_reflectivity() -> dict:
    base = 'C:/Users/pdmurray/Desktop/peyton/Projects/YBCO_Getters/Data/'
    fnames = {'AG': base+'As Grown/YBCO_Gd0nm_PNR/YBCO_Gd0nm-1-refl',
              '3nm': base+'Gd (3 nm)/YBCO_Gd3nm_PNR/YBCO_Gd3nm-1-refl',
              '7nm': base+'Gd (7 nm)/YBCO_Gd7nm_PNR/YBCO_Gd7nm-1-refl',
              '20nm': base+'Gd (20 nm)/YBCO_Gd20nm_PNR/YBCO_Gd20nm-1-refl'}

    return {key: ReflData(key, fname) for key, fname in fnames.items()}


def plot_reflectivity():

    _, ax = plt.subplots(nrows=4, ncols=1, figsize=(12, 10))
    for i, (sample, data) in enumerate(get_reflectivity().items()):


        # -- Reflectivity
        ax[i].errorbar(data.mm['Q'],
                       data.mm['R'],
                       xerr=data.mm['dQ'],
                       yerr=data.mm['dR'],
                       color='r',
                       linestyle='',
                       marker='o',
                       markersize=4)

        ax[i].plot(data.mm['Q'],
                   data.mm['theory'],
                   linestyle='-',
                   marker='',
                   color='r')

        # ++ Reflectivity
        ax[i].errorbar(data.pp['Q'],
                       data.pp['R'],
                       xerr=data.pp['dQ'],
                       yerr=data.pp['dR'],
                       color='k',
                       linestyle='',
                       marker='o',
                       markersize=4)

        ax[i].plot(data.pp['Q'],
                   data.pp['theory'],
                   linestyle='-',
                   marker='',
                   color='k')

        ax[i].set_xlim([data.pp['Q'].min()-0.001, data.pp['Q'].max()+0.001])
        ax[i].legend()
        ax[i].set_yscale('log')
        ax[i].text(0.01, 0.1, sample)

    return


if __name__=='__main__':
    colors = {'AG': 'black', '3nm': 'saddlebrown', '7nm': 'darkgoldenrod', '20nm': 'olivedrab'}

    data = get_data()
    # oxygen = calculate_oxygen(data)
    plot_data(data, colors)
    # plot_oxygen(data, oxygen, colors)
    # plot_data_separately(data, colors)

    # plot_reflectivity()

    plt.show()