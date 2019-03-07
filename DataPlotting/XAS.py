import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.cm import get_cmap
# import numpy as np
import pandas as pd

def readData(fname):
	data = pd.read_csv(fname, sep=',', skiprows=3, names=["Photon Energy", "Absorption", "Photon Energy", "Absorption", "Photon Energy", "Absorption", "Photon Energy", "Absorption", "Photon Energy", "Absorption"])
	return data


if __name__ == "__main__":
	base = 'C:/Users/pdmurray/Desktop/peyton/Projects/YBCO_Getters/Data/Combined Datasets/'
	fname = base+'XAS.csv'
	data = readData(fname)

	cmap = get_cmap('inferno')

	fig = plt.figure(figsize=(10.015,8))
	fig.subplots_adjust(top=0.98,
						bottom=0.095,
						left=0.1,
						right=0.965,
						hspace=0.2,
						wspace=0.2)

	axMain = fig.add_subplot(111)
	left, bottom, width, height = [0.58, 0.55, 0.35, 0.4]
	axInset = fig.add_axes([left, bottom, width, height])

	colors = ['black', 'saddlebrown', 'darkgoldenrod', "olivedrab"]


	# Plot the data
	axMain.plot(data.iloc[:, 0].values, data.iloc[:, 1].values, linestyle='-', color=colors[0], label="As Grown")
	axMain.plot(data.iloc[:, 2].values, data.iloc[:, 3].values, linestyle='-', color=colors[1], label="Gd (3 nm)")
	axMain.plot(data.iloc[:, 4].values, data.iloc[:, 5].values, linestyle='-', color=colors[2], label="Gd (7 nm)")
	axMain.plot(data.iloc[:, 6].values, data.iloc[:, 7].values, linestyle='-', color=colors[3], label="Gd (20 nm)")
	# axMain.plot(data.iloc[:, 8].values, data.iloc[:, 9].values, linestyle='-', color='k', label="Ta (20 nm)")

	axInset.plot(data.iloc[:, 0].values, data.iloc[:, 1].values, linestyle='-', color=colors[0], label="As Grown")
	axInset.plot(data.iloc[:, 2].values, data.iloc[:, 3].values, linestyle='-', color=colors[1], label="Gd (3 nm)")
	axInset.plot(data.iloc[:, 4].values, data.iloc[:, 5].values, linestyle='-', color=colors[2], label="Gd (7 nm)")
	axInset.plot(data.iloc[:, 6].values, data.iloc[:, 7].values, linestyle='-', color=colors[3], label="Gd (20 nm)")
	# axInset.plot(data.iloc[:, 8].values, data.iloc[:, 9].values, linestyle='-', color='k', label="Ta (20 nm)")



	axMain.set_xlim([928, 940])
	axMain.set_ylim([0.0, 1.1])

	axInset.set_xlim([926, 960])
	axInset.set_ylim([0.0, 1.1])

	textSizeMain = 'xx-large'
	textSizeInset = 'small'
	textSizeMainLabel = "xx-large"

	axMain.set_xlabel("Photon Energy (eV)", size=textSizeMain)
	axMain.set_ylabel("Absorption (a.u.)", size=textSizeMain)
	axMain.text(x=934.5, y=0.36, s=r"$\mathrm{Cu}^{1+}$", fontsize='large')
	axMain.text(x=928.5, y=1.02, s="Cu $L_3$-Edge", fontsize=textSizeMainLabel)

	axInset.set_xlabel("Photon Energy (eV)", size=textSizeInset)
	axInset.set_ylabel("Absorption (a.u.)", size=textSizeInset)
	axInset.set_xticks([930,940,950,960])
	axInset.set_yticks([0.0, 0.5, 1.0])
	axInset.tick_params(axis='both', labelsize=textSizeInset)
	axInset.text(x=931, y=1.02, s=r"$\mathrm{L}_3$", fontsize=textSizeInset)
	axInset.text(x=951, y=0.75, s=r"$\mathrm{L}_2$", fontsize=textSizeInset)

	axInset.add_patch(patches.Rectangle((928,0.0), 12, 1.1, fill=True, alpha=0.1))

	axMain.legend(loc=(0.8, 0.02))

	# plt.savefig('XAS.svg', dpi=300, bbox_inches='tight')

	plt.show()