# Magic to import a module into the refl1d namespace
import os, imp;
from refl1d.names import *
from numpy import *
from scipy.special import erf
#from refl1d.flayer import FunctionalProfile as FP
#from refl1d.flayer import FunctionalMagnetism as FM
# FIT USING REFL1D 0.6.19

# === Instrument specific header ===
from refl1d.ncnrdata import MAGIK, PBR

# DEFINE BEAM PARAMETERS
FWHM = 0.044# FWHM of rocking curve in degrees
sampleWidth = 5	#mm
slit1 = 1.0		#slit width
slit2 = 1.0		#slit width
sampleBroadening = FWHM - degrees(0.5*(slit1+slit2)/(PBR.d_s1-PBR.d_s2))
instrument = PBR(slits_at_Tlo=(slit1, slit2), sample_width=sampleWidth, sample_broadening=sampleBroadening)

fileName = "YBCOGd(20nm)_20mT_baseTemp.reflA"

# LOAD DATA AND CREATE A PROBE
probe = instrument.load_magnetic(fileName, back_reflectivity=False)
# probe = instrument.load(fileName, back_reflectivity=False)

#SLDs (5 Å neutron radiation)
STO			= SLD(name="SrTiO3",	rho=3.325,		irho=0.000)
YBa2Cu3O6	= SLD(name="YBCO6",		rho=4.427,		irho=0.000)
YBa2Cu3O7	= SLD(name="YBCO7",		rho=4.652,		irho=0.000)
YBCOx1		= SLD(name="YBCOx",		rho=1.000,		irho=0.000)	#Fit rho and irho
YBCOx2		= SLD(name="YBCOx",		rho=1.000,		irho=0.000)	#Fit rho and irho
YBCOx3		= SLD(name="YBCOx",		rho=1.000,		irho=0.000)	#Fit rho and irho
Gd2O3 		= SLD(name="GdOx",		rho=4.483,		irho=-3.403)
Gd			= SLD(name="Gd",		rho=2.875,		irho=-4.182)
Au          = SLD(name="Au",        rho=4.662,      irho=-0.016)
shit		= SLD(name="Surface",	rho=5.000,		irho=0)

sample = (STO(0,0)|YBCOx1(300,50)|YBCOx2(300,50)|YBCOx3(300,50)|Gd2O3(200,10)|Gd(1,10,Magnetism(rhoM=0.5, thetaM=270))|Au(100,5)|shit(5,1)|air)

probe.pp.intensity.range(0.001,1)
probe.mm.intensity.range(0.001,1)
#probe.pp.theta_offset.pm(0.02)
#probe.mm.theta_offset.pm(0.02)
# probe.pp.background.range(1e-8, 1e-5)
# probe.mm.background.range(1e-8, 1e-5)

# probe.intensity.range(0.0001,0.1)
# probe.background.range(1e-8,1e-5)
# probe.theta_offset.pm(0.02)

YBCOx1.rho.range(1,6)
YBCOx2.rho.range(1,6)
YBCOx3.rho.range(1,6)
# YBCOx.irho.range(0,6)
shit.rho.range(1,9)

# sample[YBCOx1].thickness.range(0,1100)
# sample[YBCOx2].thickness.range(0,1100)
# sample[YBCOx3].thickness.range(0,1100)
sample[Gd2O3].thickness.range(0,210)
sample[Gd].thickness.range(0,210)
sample[Au].thickness.range(70,120)
sample[shit].thickness.range(1,30)

# sample[STO].interface.range(0,5)
sample[YBCOx1].interface.range(0,100)
sample[YBCOx2].interface.range(0,100)
sample[YBCOx3].interface.range(0,100)
sample[Gd2O3].interface.range(0,210)
sample[Gd].interface.range(0,210)
sample[Au].interface.range(0,10)
sample[shit].interface.range(1,15)

sample[Gd].magnetism.rhoM.range(-1,1)
# sample[YBCOx].magnetism.rhoM.range(-1,1)

#Constraints
# YBCO_total_thickness = 1000
# sample[YBCOx3].thickness = YBCO_total_thickness - (sample[YBCOx1].thickness + sample[YBCOx2].thickness)

# === Problem definition ===
zed = 0.1 # microslabbing bin size, in A
alpha = 0.0 # integration factor - leave this at zero
step = True
model = Experiment(probe=probe, sample=sample, name="Murray YBCO", dz=zed, dA=alpha, step_interfaces=step)
problem = FitProblem(model)
