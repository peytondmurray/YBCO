#Calculate d spacings and strain of (002) bragg peaks

import numpy as np


tth = [15.161, 15.111, 15.063, 14.672, 14.477]
tthrad = [angle*np.pi/180 for angle in tth]
d = [1.54056/(2*np.sin(peak*0.5)) for peak in tthrad]
strain = [spacing/d[0] for spacing in d]

for peak, spacing, pctstrain in zip(tth, d, strain):
	print("{},{},{}\n".format(peak, spacing, pctstrain))