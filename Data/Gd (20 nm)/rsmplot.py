import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
import cmocean


def drawLegend(im):

	#Get current axes
	ax = plt.gca()

	#Make a divider for drawing a new set of axes
	divider = make_axes_locatable(ax)

	#Draw new set of axes to the right of the plot, 5% of the plot's width, and with 0.05 padding.
	cax = divider.append_axes("right", size="5%", pad=0.05)

	#Draw a legend at the new axes.
	cbar = plt.colorbar(im, cax=cax, ticks=[0,1,2,3,4,5,6])#, ticks=ticks)#, format = "%.3f")
	
	cbar.ax.set_yticklabels(["1E0", "1E1", "1E2", "1E3", "1E4", "1E5", "1E6"])

	#Set the max number of ticks to 5.
	#cbar.locator = ticker.MaxNLocator(nbins=6)

	#Force the colorbar to always display in scientific notation.
	#cbar.formatter.set_powerlimits((0, 0))

	#Update the ticks; otherwise the colorbar won't display in scientific notation.
	#cbar.update_ticks()

	#Set the label size of the colorbar markings
	# cbar.ax.tick_params(labelsize=35)

	#Set the current axes back to the original axes.
	plt.sca(ax)

	return

def interpolate(x, y, z):
	grid_x, grid_y = np.meshgrid(np.linspace(np.min(x), np.max(x), 1000), np.linspace(np.min(y), np.max(y), 1000))
	znew = griddata(list(zip(x, y)), z, (grid_x, grid_y), method='nearest')
	return grid_x, grid_y, znew


if __name__ == "__main__":
	filename = "(103)_2017-08-27_long-LOW INTENSITY.dat"
	print(filename)
	data = pd.read_csv(filename, sep=',', names=["x", "y", "z"])
	xtemp = data["x"].values
	ytemp = data["y"].values
	ztemp = data["z"].values
	
	x = ytemp
	y = xtemp
	z = ztemp
	
	interp_x, interp_y, interp_z = interpolate(x, y, z)
	interp_z[interp_z == 0] = np.nan
	zlog = np.log10(interp_z)
	zlog[np.isnan(zlog)] = 0
	
	extent = [np.min(x), np.max(x), np.min(y), np.max(y)]
	
	fig = plt.figure(figsize=(8,8))
	im = plt.imshow(zlog, extent=extent, interpolation='nearest', origin='lower', cmap=cmocean.cm.ice)
	drawLegend(im)
	plt.xlabel("$h$")
	plt.ylabel("$l$")
	plt.title("As Grown")
	plt.tight_layout()
	#plt.savefig(filename.split(".")[0]+".png", dpi=600, bbox_inches='tight')
	plt.savefig(filename.split(".")[0]+".svg", bbox_inches='tight')
	plt.show()