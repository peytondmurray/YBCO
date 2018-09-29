#As Grown
#-------------
#2theta = 15.1610±0.0006
#w = 0.0502±0.0008

#Gd 3 nm
#-------------
#2Theta = 15.1112±0.0005
#w = 0.0420±0.0006

#Gd 7 nm
#-------------
#2theta = 15.0602±0.0013
#w = 0.0818±0.0021

#Gd 20 nm
#-------------
#2theta = 14.4765±0.0036
#w = 0.3090±0.0051


import numpy as np

names = ["As Grown", "Gd 3 nm", "Gd 7 nm", "Gd 20 nm"]
tth = [15.1610,15.1112,15.0602,14.4765]
tth_uncertainty = [0.0006,0.0005,0.0013,0.0051]

wavelength = 1.54056 #Å

for name, angle_deg, uncertainty_deg in zip(names, tth, tth_uncertainty):
	angle = angle_deg*np.pi/180
	uncertainty = uncertainty_deg*np.pi/180
	print("{} d={}±{}".format(name, 2*wavelength/(2*np.sin(angle/2)), -1*2*uncertainty*(wavelength/2)*np.cos(angle/2)*(1/np.sin(angle/2))**2))