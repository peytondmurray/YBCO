import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_data(fname):
    return pd.read_csv(fname,
                       sep=',',
                       skiprows=2,
                       names=['2Th_AG',
                              'I_AG',
                              '2Th_3nm',
                              'I_3nm',
                              '2Th_7nm',
                              'I_7nm',
                              '2Th_20nm',
                              'I_20nm'],
                       usecols=[0, 1, 2, 3, 4, 5, 6, 7],
                       dtype=np.float)


def deg_to_rad(x):
    return x*np.pi/180


def rad_to_deg(x):
    return x*180/np.pi


def d_to_tth(d):
    L = 1.54056
    return 2*rad_to_deg(np.arcsin(L/(2*d)))


def c_axis_to_d_spacings(c, max_l=5):
    return c/np.arange(1, max_l, 1)


def add_stem_points(ax, c, color):
    max_l = 5
    ax.stem(d_to_tth(c_axis_to_d_spacings(c, max_l=max_l)), ax.get_ylim()[-1]*np.ones(max_l-1))
    return


def main():
    colors = {'AG': 'black', '3nm': 'saddlebrown', '7nm': 'darkgoldenrod', '20nm': 'olivedrab'}
    c = {'AG': 11.6755, '3nm': 11.6882, '7nm': 11.7555, '20nm': 12.4302}  # From (001)
    # c = {'AG': 11.6782, '3nm': 11.7163, '7nm': 11.7557, '20nm': 12.2271}  # From (002)

    data_xrd = read_data('C:/Users/pdmurray/Desktop/peyton/temp-ybco/Data/Combined Datasets/'+'XRD.csv')
    data_xrr = read_data('C:/Users/pdmurray/Desktop/peyton/temp-ybco/Data/Combined Datasets/'+'XRR.csv')

    fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(10, 8))

    for i, key in enumerate(colors.keys()):
        ax[i].plot(data_xrd['2Th_{}'.format(key)], data_xrd['I_{}'.format(key)], '-', color=colors[key])
        ax[i].plot(data_xrr['2Th_{}'.format(key)], data_xrr['I_{}'.format(key)], '-', color=colors[key])
        add_stem_points(ax[i], c[key], color=colors[key])
        ax[i].set_yscale('log', nonposy='clip')

    return

if __name__ == '__main__':
    main()
    plt.show()
