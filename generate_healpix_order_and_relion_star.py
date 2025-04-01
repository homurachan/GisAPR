#!/usr/bin/env python

import math, os, sys
try:
	from optparse import OptionParser
except:
	from optik import OptionParser
from healpix import healpix_py
import random, argparse
from eqps_from_fortran import eqps_with_input_step_return_relion_rot_tilt
def main():
	parser = argparse.ArgumentParser(description="This script generate orientation sampling points from healpix, then write to RELION starfile. Remember, the Rot range is [-180,180), \
	Tilt range is [0,180]")
	parser.add_argument("--o", type=str, required=True, help="Output filename")
	parser.add_argument("--randomPsi", action='store_true', help="Add random Psi to the starfile. default = False")
	parser.add_argument("--psiStep", type=int, default=8, help="The stepsize of the randomPsi, default = 8")
	parser.add_argument("--discardPositiveRot", action='store_true', help="For testing purpose, drop points that Rot > zero. default = False")
	parser.add_argument("--healpixOrder", type=int, default=3, help="The healpix order, default = 3, meaning 7.5 degree. Order=2, 15 degree. Order=4, 3.75 degree. Order=5, 1.875 degree.")
	parser.add_argument("--useEQPS", action='store_true', help="Use EQPS instead of healpx. default = False")
	parser.add_argument("--EQPSangleDegree", type=float, default=10., help="The angular distance in degree on EQPS, default = 10")
	parser.add_argument("--apix", type=float, default=1.42, help="The pixel size of this starfile, default = 1.42")
	args = parser.parse_args()
	output = args.o
	healpix_order = args.healpixOrder
	do_randomPsi=args.randomPsi
	psi_stepsize=args.psiStep
	NUM_PSI=360//psi_stepsize
	divisible=True
	do_discardPositiveRot=args.discardPositiveRot
	do_useEQPS=args.useEQPS
	EQPSangleDegree=args.EQPSangleDegree
	apix=args.apix
	if(360%psi_stepsize>0.5):
		divisible=False
		print("Your Psi stepsize cannot be divisible to 360.")
	r=open(output,"w")
	r.write("# relion 30001\n\ndata_optics\n\nloop_\n")
	r.write("_rlnOpticsGroup #1\n")
	r.write("_rlnOpticsGroupName #2\n")
	r.write("_rlnSphericalAberration #3\n")
	r.write("_rlnVoltage #4\n")
	r.write("_rlnImagePixelSize #5\n")
	r.write("\t1\topticsGroup1\t2.7\t300.0\t")
	r.write(str(apix)+"\n\n\n")
	r.write("# relion 30001\n\ndata_particles\n\nloop_\n")
	r.write("_rlnAngleRot #1\n")
	r.write("_rlnAngleTilt #2\n")
	r.write("_rlnAnglePsi #3\n")
	r.write("_rlnImageName #4\n")
	if(do_useEQPS):
		COUNT=0
		rot,tilt=eqps_with_input_step_return_relion_rot_tilt(EQPSangleDegree)
		MAX=len(rot)
		print("Total available points = ",MAX)
		for i in range(0,MAX):
			Psi = 0.0
			if(do_discardPositiveRot):
				if(math.fmod(rot[i]+360.,360.)>180.):
					continue
			## Note: Rot: FF[0] ~ [0,360) , Tilt: FF[1] ~ [0,180]
			r.write(str(math.fmod(rot[i]+360.,360.)-180.)+"\t"+str(math.fmod(tilt[i]+360.,360.))+"\t")
			if(do_randomPsi):
				TMP=0
				if(divisible):
					TMP=random.randint(0,NUM_PSI-1)
				else:
					TMP=random.randint(0,NUM_PSI)
				Psi=float(TMP*psi_stepsize)
			r.write(str(Psi)+"\t")
			r.write("1@1.mrcs\n")
			COUNT+=1
		r.close()
	else:
		AA=healpix_py(healpix_order,"NEST")
		MAX=AA.npix_
		print("Total available points = ",MAX)
		#print(MAX)
		COUNT=0
		for i in range(0,MAX):
			FF=AA.getDirectionFromHealPix(i)
			Psi = 0.0
			if(do_discardPositiveRot):
				if(math.fmod(FF[0]+360.,360.)>180.):
					continue
			## Note: Rot: FF[0] ~ [0,360) , Tilt: FF[1] ~ [0,180]
			r.write(str(math.fmod(FF[0]+360.,360.)-180.)+"\t"+str(math.fmod(FF[1]+360.,360.))+"\t")
			if(do_randomPsi):
				TMP=0
				if(divisible):
					TMP=random.randint(0,NUM_PSI-1)
				else:
					TMP=random.randint(0,NUM_PSI)
				Psi=float(TMP*psi_stepsize)
			r.write(str(Psi)+"\t")
			r.write("1@1.mrcs\n")
			COUNT+=1
		r.close()
	print("Total points written = ",COUNT)
if __name__== "__main__":
	main()
