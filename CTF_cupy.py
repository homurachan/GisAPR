import numpy as np
#import torch
#class CTF(nn.Module):
##########
##	Center of this FFT Image should be checked!!!!!!!!!!!!
##########
import mrcfile
class CTF:
	def __init__(self, defocus_U,defocus_V,defocus_A,CS=2.7,voltage=300.0,pixel_size=1.0,XSIZE=128,YSIZE=128,Amp_Const=0.1,Bfactor=0.0,Scale=1.0,Phase_Shift=0.0):
	#	super(CTF, self).__init__()
		self.defocus_U=defocus_U
		self.defocus_V=defocus_V
		self.defocus_A=defocus_A
		self.CS=CS
	#	self.beamtilt_X=beamtilt_X
	#	self.beamtilt_Y=beamtilt_Y
	##	Beamtilt cannot be calculated here. It must be applied directly to projection.
		self.voltage=voltage
		self.pixel_size=pixel_size
		self.XSIZE=XSIZE
		self.YSIZE=YSIZE
		self.Amp_Const=Amp_Const
		self.Bfactor=Bfactor
		self.Scale=Scale
		self.Phase_Shift=Phase_Shift
		
		self.local_Cs = self.CS * 1e7
		self.local_kV = voltage * 1e3
		self.rad_azimuth = np.radians(defocus_A)

		# Average focus and deviation
		self.defocus_average = -(self.defocus_U + self.defocus_V) * 0.5
		self.defocus_deviation = -(self.defocus_U - self.defocus_V) * 0.5

		# lambda=h/sqrt(2*m*e*kV)
		#    h: Planck constant
		#    m: electron mass
		#    e: electron charge
		# lambda=0.387832/sqrt(kV*(1.+0.000978466*kV)) # Hewz: Angstroms
		# lambda=h/sqrt(2*m*e*kV)
		self.lambda_value = 12.2643247 / np.sqrt(self.local_kV * (1. + self.local_kV * 0.978466e-6))  # See http://en.wikipedia.org/wiki/Electron_diffraction

		# Helpful constants
		# ICE: X(u)=-PI/2*deltaf(u)*lambda*u^2+PI/2*Cs*lambda^3*u^4
		#          = K1*deltaf(u)*u^2         +K2*u^4
		self.K1 = np.pi / 2 * 2 * self.lambda_value
		self.K2 = np.pi / 2 * self.local_Cs * self.lambda_value ** 3
		self.K3 = np.arctan(self.Amp_Const/np.sqrt(1-self.Amp_Const*self.Amp_Const))
		self.K4 = -self.Bfactor / 4.

		# Phase shift in radian
		self.K5 = np.radians(self.Phase_Shift)

		if self.Amp_Const < 0. or self.Amp_Const > 1.:
			raise ValueError("CTF::initialise ERROR: AmplitudeContrast cannot be smaller than zero or larger than one!")
		
		# express astigmatism as a bilinear form:
		sin_az = np.sin(self.rad_azimuth)
		cos_az = np.cos(self.rad_azimuth)
		'''
		Q=np.asarray([cos_az, sin_az, -sin_az, cos_az])
		Qt=np.asarray([cos_az, -sin_az, sin_az, cos_az])
		D=np.asarray([-self.defocus_U, 0.0, 0.0, -self.defocus_V])
		Q=Q.reshape(2,2)
		Qt=Qt.reshape(2,2)
		D=D.reshape(2,2)
		A = Qt * D * Q
		self.Axx = A[0,0]
		self.Axy = A[0,1]
		self.Ayy = A[1,1]
		# This is not correct. I manually conduct the result down here.
		'''
		self.Axx=-self.defocus_U*cos_az*cos_az-self.defocus_V*sin_az*sin_az
		self.Ayy=-self.defocus_U*sin_az*sin_az-self.defocus_V*cos_az*cos_az
		self.Axy=-self.defocus_U*sin_az*cos_az+self.defocus_V*sin_az*cos_az
#		print(self.defocus_U,self.defocus_V,self.defocus_A,sin_az,cos_az)
		#e.g. defocus_A=50, dfU=1000, dfv=1200, >>> A= [[-413.17591117, -0.],[-0., -495.8110934]]

	def getCTF(self,X,Y,do_abs=False,do_only_flip_phases=False,do_intact_until_first_peak=False,do_damping=False):
		gammaOffset=0.0
		u2 = X * X + Y * Y
		u4 = u2 * u2
		gamma = self.K1 * (self.Axx*X*X + 2.0*self.Axy*X*Y + self.Ayy*Y*Y) + self.K2 * u4 - self.K5 - self.K3 + gammaOffset
		
		np.ones_like(gamma)
		if do_intact_until_first_peak:
			mask = np.fabs(gamma) < np.pi / 2.
			retval[mask] = 1.0
			retval[~mask] = -np.sin(gamma[~mask])
		else:
			retval = -np.sin(gamma)

		if (do_damping):
			E = np.exp(self.K4 * u2) # B-factor decay (K4 = -Bfac/4)
			retval *= E

		if (do_abs):
			retval = np.abs(retval)
		elif (do_only_flip_phases):
			retval = np.where(retval < 0.0, -1.0, 1.0)
		retval *= self.Scale

		retval = np.where(np.fabs(retval) < 1e-8, np.sign(retval) * 1e-8, retval)
		return retval
		
	def getFftwImage(self,do_abs,do_only_flip_phases,do_intact_until_first_peak,do_damping):
		xs = self.XSIZE * self.pixel_size *1.0
		ys = self.YSIZE * self.pixel_size *1.0
		result = np.ones(self.YSIZE*self.XSIZE,dtype=np.complex_).reshape(self.YSIZE,self.XSIZE)
	#	C=0
		height, width = result.shape
		
		# Generate a grid of indices
		indices = np.arange(height * width)
		
		# Compute jp and ip values for all indices
		jp = np.mod(indices, self.XSIZE) - self.XSIZE / 2
		ip = np.floor(indices / self.XSIZE) - self.YSIZE / 2
		x = jp / xs
		y = ip / ys
		result.flat = self.getCTF(x, y, do_abs, do_only_flip_phases, do_intact_until_first_peak, do_damping)
		'''for i in np.nditer(result,order='C',op_flags = ['readwrite']):
			jp=float(np.mod(C,self.XSIZE)-self.XSIZE/2)
			ip=float(np.floor(C/self.XSIZE)-self.YSIZE/2)
			
			i[...] = self.getCTF(x, y, do_abs, do_only_flip_phases, do_intact_until_first_peak, do_damping)
		#	should not be i = ...
			C+=1
		'''
		return result
		
	def getFftwImage_with_isSPA_weight(self,do_abs,do_only_flip_phases,do_intact_until_first_peak,do_damping,kk=3.0):
		# gisspa fixed weight:
		a=-9.32
		b=2.65
		b2=0.01908
		bfactor=-78.7757
		bfactor2=-12.9121
		bfactor3=1.28732
		# reminder: These params should be fitted again for atmoic model!!
		xs = self.XSIZE * self.pixel_size *1.0
		ys = self.YSIZE * self.pixel_size *1.0
		result = np.ones(self.YSIZE*self.XSIZE,dtype=np.complex_).reshape(self.YSIZE,self.XSIZE)
		height, width = result.shape
		
		# Create a grid of indices
		C = np.arange(height * width)
		
		# Compute jp and ip values for all indices
		jp = np.mod(C, self.XSIZE) - self.XSIZE / 2
		ip = np.floor(C / self.XSIZE) - self.YSIZE / 2
		
		# Normalize x and y
		x = jp / xs
		y = ip / ys
		
		# Compute ss, signal, and Ncurve for all elements
		ss = np.sqrt(x**2 + y**2)
		signal = np.exp(bfactor * ss**2 + bfactor2 * ss + bfactor3) / (kk + 1)
		Ncurve = np.exp(a * ss**2 + b * ss + b2) / signal
		
		# Vectorized calculation of the CTF values
		v = self.getCTF(x, y, do_abs, do_only_flip_phases, do_intact_until_first_peak, do_damping)
		
		# Update the result array with the computed values
		result.flat = v * np.sqrt(1. / (Ncurve + kk * v**2))
		return result
# add GisSPA weighting. Right above.

# Usage: ctf1=CTF(dfu,dfv,dfa,cs,voltage,apix,xsize,ysize,ac,bf,scale,ps)
#ctf_image=ctf1.getFftwImage(False,False,False,False)

#result in np.complex_

#ctf1=CTF(10000.,5000.,30.)
#print(ctf1.Axx,ctf1.Ayy)
#ctf_data=ctf1.getFftwImage(False,False,False,False)
#print("debug, ctf_data.shape=",ctf_data.shape)
#print(ctf_data)
#with mrcfile.new("debug_ctf.mrc",overwrite=True) as mrc1:
#	mrc1.set_data((ctf_data.real).astype(np.float32))
#mrc1.close()

