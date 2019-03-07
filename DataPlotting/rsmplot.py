import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
from matplotlib.cm import get_cmap
import matplotlib.gridspec as gridspec
from matplotlib.transforms import Bbox
import cmocean


def drawLegend(im, vmin, vmax):

	#Get current axes
	ax = plt.gca()

	#Make a divider for drawing a new set of axes
	divider = make_axes_locatable(ax)

	#Draw new set of axes to the right of the plot, 5% of the plot's width, and with 0.05 padding.
	cax = divider.append_axes("right", size="15%", pad=0.05)

	#Draw a legend at the new axes.
	# cbar = plt.colorbar(im, cax=cax, ticks=[0,1,2,3,4,5,6])#, ticks=ticks)#, format = "%.3f")
	cbar = plt.colorbar(im, cax=cax, ticks=[vmin,vmax])#, ticks=ticks)#, format = "%.3f")

	# cbar.ax.set_yticklabels(["1E0", "1E1", "1E2", "1E3", "1E4", "1E5", "1E6"])
	cbar.ax.set_yticklabels(["Min", "Max"])

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

def interpolate(x, y, z, stepx, stepy):
	grid_x, grid_y = np.meshgrid(np.arange(np.min(x), np.max(x), stepx), np.arange(np.min(y), np.max(y), stepy))
	znew = griddata(list(zip(x, y)), z, (grid_x, grid_y), method='nearest')
	return grid_x, grid_y, znew

def makePlot(fname, ax, vmin, scaling=1, legend=True, cmap=cmocean.cm.tempo_r):
	x, y, z, stepx, stepy = getData(fname)
	interp_x, interp_y, interp_z, zlog = interpAndMask(x, y, z, stepx, stepy)

	extent = [np.min(x), np.max(x), np.min(y), np.max(y)]
	vmax = np.nanmax(zlog)
	im = ax.imshow(scaling*zlog, extent=extent, interpolation='nearest', origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)

	if legend:
		drawLegend(im, vmin, vmax)

	return interp_x, interp_y, interp_z, zlog

def read_file(fname):
	data = pd.read_csv(fname, sep=',', names=["x", "y", "z"])
	x = data["x"].values
	y = data["y"].values
	z = data["z"].values
	# print("file: {}, nx: {}, ny: {}".format(fname, np.sum(y==y[0]), np.sum(x==x[0])))
	return x, y, z, getStep(x), getStep(y)

def getStep(a):
	for item in a:
		if item != a[0]:
			return np.abs(a[0]-item)

	raise ValueError("No step detected!")

def interpAndMask(x, y, z, stepx, stepy):
	interp_x, interp_y, interp_z = interpolate(x, y, z, stepx, stepy)
	interp_z[interp_z == 0] = np.nan
	zlog = np.log10(interp_z)
	zlog[np.isnan(zlog)] = 0
	return interp_x, interp_y, interp_z, zlog

def integrateNormalize(z, axis):
	integral = np.nansum(z, axis=axis)
	return integral/np.nanmax(integral)

def get_data():

	filenameBase = "C:/Users/pdmurray/Desktop/peyton/Projects/YBCO_Getters/Data/"
	filenames = {"AG":"As Grown/AsDep_103RSM_17hr.dat", "3nm":"Gd (3 nm)/Gd3nm_103RSM_20hr.dat", "7nm":"Gd (7 nm)/Gd7nm_103RSM_20hr.dat", "20nm":"Gd (20 nm)/Gd20nm_103RSM_17hr.dat"}

	# x, y, z, step_x, step_y = dict(), dict(), dict(), dict(), dict()
	data = dict()
	for key, value in filenames.items():
		data[key] = dict()
		x_tmp, y_tmp, z_tmp, step_x_tmp, step_y_tmp = read_file(filenameBase+value)
		data[key]['x'] = x_tmp
		data[key]['y'] = y_tmp
		data[key]['z'] = z_tmp
		data[key]['stepx'] = step_x_tmp
		data[key]['stepy'] = step_y_tmp

	return data

def old_main():
	filenameBase = "C:/Users/pdmurray/Desktop/peyton/Projects/YBCO_Getters/Data/"
	filenames = {"AG":"As Grown/AsDep_103RSM_17hr.dat", "3nm":"Gd (3 nm)/Gd3nm_103RSM_20hr.dat", "7nm":"Gd (7 nm)/Gd7nm_103RSM_20hr.dat", "20nm":"Gd (20 nm)/Gd20nm_103RSM_17hr.dat"}
	titles = {"AG":"As Grown", "3nm":"Gd (3 nm)", "7nm":"Gd (7 nm)", "20nm":"Gd (20 nm)"}
	vmins = {"AG":0.1, "3nm":0.6, "7nm":0.6, "20nm":0.3}
	colors = {"AG": 'black', "3nm": 'saddlebrown', "7nm": 'darkgoldenrod', "20nm": "olivedrab"}
	legends = {"AG":False, "3nm":False, "7nm":False, "20nm":False}
	cmap = cmocean.cm.deep_r

	offset = {"AG": 10, "3nm": 2, "7nm": 0.35, "20nm": 0.3}  # Use with normalization
	scaling = {"AG":2, "3nm":2, "7nm":2, "20nm":2}
	interp_x, interp_y, interp_z, zlog = dict(), dict(), dict(), dict()

	fig = plt.figure(figsize=(8, 8))
	axes = fig.subplots(nrows=2, ncols=5)

	ax = {"AG":axes[0,0],"3nm":axes[0,1],"7nm":axes[0,2],"20nm":axes[0,3]}
	ax2 = {"AG":axes[1,0],"3nm":axes[1,1],"7nm":axes[1,2],"20nm":axes[1,3]}

	ax3 = axes[0,4]
	ax4 = axes[1,4]

	for key in colors.keys():
		interp_x[key], interp_y[key], interp_z[key], zlog[key] = makePlot(filenameBase+filenames[key], ax[key], vmins[key], 1, legends[key], cmap=cmap)
		ax[key].set_title(titles[key])
		ax[key].set(xlim=(-1.0405, -0.97), ylim=(2.95, 3.05))
		ax[key].set_xticklabels([])


		if key != "AG":
			ax[key].set_yticklabels([])
			ax2[key].set_yticklabels([])

		makePlot(filenameBase+filenames[key], ax2[key], vmins[key], scaling[key], legends[key], cmap=cmap)
		ax2[key].set(xlim=(-1.0405, -0.97), ylim=(2.60, 2.75))

		ax[key].set_xticks([-1.04,-1.00])
		ax2[key].set_xticks([-1.04,-1.00])

		ax3.plot(integrateNormalize(interp_z[key], axis=1) * offset[key], interp_y[key][:, 0], linestyle='-', color=colors[key])
		# ax3.plot(integrateNormalize(zlog[key], axis=1) * offset[key], interp_y[key][:, 0], linestyle='-', color=colors[key])
		ax4.plot(integrateNormalize(interp_z[key], axis=1) * offset[key], interp_y[key][:, 0], linestyle='-', color=colors[key])
		# ax4.plot(integrateNormalize(zlog[key], axis=1) * offset[key], interp_y[key][:, 0], linestyle='-', color=colors[key])

	ax3.set(xscale="log", ylim=(2.95, 3.05))
	# ax3.set(ylim=(2.95, 3.05))
	ax4.set(xscale="log", ylim=(2.60, 2.75))
	# ax4.set(ylim=(2.60, 2.75))

	ax3.set_xticklabels([])
	ax3.set_yticklabels([])
	ax4.set_xticklabels([])
	ax4.set_yticklabels([])



	# 	if key != "AG":
	# 		ax[key].set_yticklabels([])
	# 	ax[key].set_xticks([-1.04, -1.00])
	#
	# 	axes[4].plot(integrateNormalize(interp_z[key], axis=1) * offset[key], interp_y[key][:, 0], linestyle='-', color=colors[key])
	#
	# axes[4].set(xscale="log", xlabel="Intensity", title="Int. Intensity", ylim=(2.6, 3.05))
	# axes[4].set_xticklabels([])
	# axes[4].set_yticklabels([])





	# offset = [20,2,0.5,0.2]		#Use without normalization
	# offset = [1,1,1,1]


	# offset = [20,2,0.5,0.2]		#Use without normalization
	# offset = [1,1,1,1]
	# offset = [3,2,1,0.7]				#Use with normalization
	#
	# fig2 = plt.figure(figsize=(8,4))
	# ax2 = fig2.add_subplot(111)
	# for i, data in enumerate(interpData):
	# 	x, y, z, zlog = data
	#
	# 	zcropped = z.copy()[np.where(y[:,0] < 2.8)[0], :]
	# 	ax2.plot(x[0,:], integrateNormalize(zcropped, axis=0)*offset[i], linestyle='-', color=colors[i])
	# 	# plt.plot(np.nansum(z, axis=1)*offset[i], y[:,0], linestyle='-', color=colors[i])
	# 	plt.xlim(-1.0405,-0.97)
	# 	ax2.set_yscale("log")
	# 	# ax2.set_yticklabels([])
	#
	# 	ax2.set_ylabel("Intensity")
	# 	ax2.set_xlabel(r"$h$")
	# 	ax2.set_title("YBCO Film Peak Integrated+Normalized Intensity")
	# 	fig2.tight_layout()
	# 	# fig2.savefig("Integrated_along_(00l).svg", bbox_inches='tight')

def reduce_data(data):
	for key in data.keys():
		data[key]['interp_x'], data[key]['interp_y'], data[key]['interp_z'], data[key]['zlog'] = interpAndMask(data[key]['x'], data[key]['y'], data[key]['z'], data[key]['stepx'], data[key]['stepy'])

	return data

def generate_plots(data, titles, colors, offsets, cmap=cmocean.cm.deep_r):
	fig = plt.figure(figsize=(8,8))

	axes = fig.subplots(nrows=2, ncols=5)

	ax = {"AG":axes[0,0],"3nm":axes[0,1],"7nm":axes[0,2],"20nm":axes[0,3]}
	ax2 = {"AG":axes[1,0],"3nm":axes[1,1],"7nm":axes[1,2],"20nm":axes[1,3]}
	factors = {"AG":2.5,"3nm":2.5,"7nm":2.5,"20nm":2.5}
	vmins = {"AG":0.2, "3nm":0.7, "7nm":0.7, "20nm":0.4}

	ax3 = axes[0,4]
	ax4 = axes[1,4]

	for key in data.keys():

		substrate_peak_height = np.nanmax(data[key]['zlog'])

		ax[key].imshow(data[key]['zlog'],
					   extent=[np.min(data[key]['x']), np.max(data[key]['x']), np.min(data[key]['y']), np.max(data[key]['y'])],
					   interpolation='nearest',
					   origin='lower',
					   vmin=vmins[key],
					   vmax=substrate_peak_height,
					   cmap=cmap)

		ax2[key].imshow(factors[key]*data[key]['zlog'],
					    extent=[np.min(data[key]['x']), np.max(data[key]['x']), np.min(data[key]['y']), np.max(data[key]['y'])],
					    interpolation='nearest',
					    origin='lower',
					    vmin=vmins[key]*factors[key],
					    vmax=substrate_peak_height,
					    cmap=cmap)

		offset_data = integrateNormalize(data[key]['interp_z'], axis=1)*offsets[key]
		ax3.plot(offset_data, data[key]['interp_y'][:, 0], color=colors[key], linestyle='-')
		ax4.plot(offset_data, data[key]['interp_y'][:, 0], color=colors[key], linestyle='-')

		ax[key].set_title(titles[key])
		ax[key].set(xlim=(-1.0405, -0.97), ylim=(2.95, 3.05))
		ax[key].set_xticks([-1.04,-1.00])
		ax[key].set_xticklabels([])

		ax2[key].set(xlim=(-1.0405, -0.97), ylim=(2.60, 2.75))
		ax2[key].set_xticks([-1.04,-1.00])
		if key != "AG":
			ax[key].set_yticklabels([])
			ax2[key].set_yticklabels([])

	ax3.set(xscale="log", ylim=(2.95, 3.05))
	ax4.set(xscale="log", ylim=(2.60, 2.75))
	ax3.set_xticklabels([])
	ax3.set_yticklabels([])
	ax4.set_xticklabels([])
	ax4.set_yticklabels([])

	return


def integrate_H_and_plot(data, colors):

	fig, ax = plt.subplots(nrows=1, ncols=1)
	for key in data.keys():

		i_film_upper = np.nonzero(data[key]['interp_y'][:, 0] <= 2.76)[0][-1]
		i_film_lower = np.nonzero(data[key]['interp_y'][:, 0] >= 2.60)[0][0]


		ax.plot(data[key]['interp_x'][0, :],
				integrateNormalize(data[key]['zlog'][i_film_lower:i_film_upper, :], axis=0),
				color=colors[key],
				marker='',
				linestyle='-',
				label=key)

	ax.legend()

	return




if __name__ == "__main__":

	titles = {"AG":"As Grown", "3nm":"Gd (3 nm)", "7nm":"Gd (7 nm)", "20nm":"Gd (20 nm)"}
	colors = {"AG": 'black', "3nm": 'saddlebrown', "7nm": 'darkgoldenrod', "20nm": "olivedrab"}
	offsets = {"AG": 10, "3nm": 2, "7nm": 0.35, "20nm": 0.3}
	data = get_data()
	data = reduce_data(data)
	# generate_plots(data, titles, colors, offsets)
	integrate_H_and_plot(data, colors)


	plt.show()