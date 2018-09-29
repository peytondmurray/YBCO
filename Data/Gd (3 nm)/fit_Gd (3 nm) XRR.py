# Magic to import a module into the refl1d namespace
import os, imp;
from refl1d.names import *
from numpy import *
from scipy.special import erf
#from refl1d.flayer import FunctionalProfile as FP
#from refl1d.flayer import FunctionalMagnetism as FM
# FIT USING REFL1D 0.6.19

# === Instrument specific header ===
from refl1d.ncnrdata import XRay

# DEFINE BEAM PARAMETERS
instrument = XRay(slits_at_Tlo=0)


fileName = "Gd (3 nm) XRR.xy"

# LOAD DATA AND CREATE A PROBE
probe = instrument.load(filename, sample_broadening=1e-4)

#SLDs (5 Å neutron radiation)
#STO			= SLD(name="SrTiO3",	rho=3.325,		irho=0.000)
#YBa2Cu3O6	= SLD(name="YBCO6",		rho=4.427,		irho=0.000)
#YBa2Cu3O7	= SLD(name="YBCO7",		rho=4.652,		irho=0.000)
#YBCOx1		= SLD(name="YBCOx",		rho=1.000,		irho=0.000)	#Fit rho and irho
#YBCOx2		= SLD(name="YBCOx",		rho=1.000,		irho=0.000)	#Fit rho and irho
#YBCOx3		= SLD(name="YBCOx",		rho=1.000,		irho=0.000)	#Fit rho and irho
#Gd2O3 		= SLD(name="GdOx",		rho=4.483,		irho=-3.403)
#Gd			= SLD(name="Gd",		rho=2.875,		irho=-4.182)
#Au          = SLD(name="Au",        rho=4.662,      irho=-0.016)
#shit		= SLD(name="Surface",	rho=5.000,		irho=0)

#SLDs (Cu Ka X-rays)
STO			= SLD(name="SrTiO3",	rho=37.387,		irho=-1.699)
YBa2Cu3O6	= SLD(name="YBCO6",		rho=45.742,		irho=-3.653)
YBa2Cu3O7	= SLD(name="YBCO7",		rho=45.936,		irho=-3.571)
YBCOx1			= SLD(name="YBCOx1",		rho=45.8,			irho=-3.6)		#Fit rho and irho
YBCOx2			= SLD(name="YBCOx2",		rho=45.8,			irho=-3.6)		#Fit rho and irho
YBCOx3			= SLD(name="YBCOx3",		rho=45.8,			irho=-3.6)		#Fit rho and irho
Gd2O3 		= SLD(name="GdOx",		rho=46.233,		irho=-9.239)
Gd			= SLD(name="Gd",		rho=46.522,		irho=-11.311)
Au          = SLD(name="Au",        rho=124.694,      irho=-12.851)
#shit		= SLD(name="Surface",	rho=5.000,		irho=0)

sample = (STO(0,0)|YBCOx1(300,50)|Gd2O3(30,10)|Gd(1,5)|Au(100,5)|air)

probe.intensity.range(0.1,1.6)
probe.background.range(1e-8,1e-5)
# probe.theta_offset.pm(0.02)

YBCOx1.rho.range(44,46)
YBCOx1.irho.range(3.3,3.8)

# sample[YBCOx1].thickness.range(0,1100)
# sample[YBCOx2].thickness.range(0,1100)
# sample[YBCOx3].thickness.range(0,1100)
sample[Gd2O3].thickness.range(0,50)
sample[Gd].thickness.range(0,50)
sample[Au].thickness.range(70,120)

# sample[STO].interface.range(0,5)
sample[YBCOx1].interface.range(0,100)
sample[Gd2O3].interface.range(0,30)
sample[Gd].interface.range(0,30)
sample[Au].interface.range(0,30)

#Constraints
# YBCO_total_thickness = 1000
# sample[YBCOx3].thickness = YBCO_total_thickness - (sample[YBCOx1].thickness + sample[YBCOx2].thickness)

# === Problem definition ===
zed = 0.1 # microslabbing bin size, in A
alpha = 0.0 # integration factor - leave this at zero
step = True
model = Experiment(probe=probe, sample=sample, name="Murray YBCO", dz=zed, dA=alpha, step_interfaces=step)
problem = FitProblem(model)
