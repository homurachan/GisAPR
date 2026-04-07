import numpy as np
#import mrcfile
#import time
class CTF:
	def __init__(
		self,
		defocus_U, defocus_V, defocus_A,
		CS=2.7, voltage=300.0, pixel_size=1.0,
		XSIZE=128, YSIZE=128,
		Amp_Const=0.1, Bfactor=0.0, Scale=1.0, Phase_Shift=0.0,
		dtype=np.float32
	):
		self.defocus_U = dtype(defocus_U)
		self.defocus_V = dtype(defocus_V)
		self.defocus_A = dtype(defocus_A)
		self.CS = dtype(CS)
		self.voltage = dtype(voltage)
		self.pixel_size = dtype(pixel_size)
		self.XSIZE = int(XSIZE)
		self.YSIZE = int(YSIZE)
		self.Amp_Const = dtype(Amp_Const)
		self.Bfactor = dtype(Bfactor)
		self.Scale = dtype(Scale)
		self.Phase_Shift = dtype(Phase_Shift)
		self.dtype = dtype

		if self.Amp_Const < 0.0 or self.Amp_Const > 1.0:
			raise ValueError("Amplitude contrast must be in [0, 1].")

		self._init_constants()
		self._build_frequency_grid()

	def _init_constants(self):
		local_Cs = self.CS * self.dtype(1e7)
		local_kV = self.voltage * self.dtype(1e3)
		rad_azimuth = np.radians(self.defocus_A).astype(self.dtype)

		self.lambda_value = self.dtype(12.2643247) / np.sqrt(
			local_kV * (self.dtype(1.0) + local_kV * self.dtype(0.978466e-6))
		)

		self.K1 = self.dtype(np.pi) * self.lambda_value
		self.K2 = self.dtype(np.pi / 2.0) * local_Cs * self.lambda_value ** 3
		self.K3 = np.arctan(
			self.Amp_Const / np.sqrt(self.dtype(1.0) - self.Amp_Const * self.Amp_Const)
		).astype(self.dtype)
		self.K4 = (-self.Bfactor / self.dtype(4.0)).astype(self.dtype)
		self.K5 = np.radians(self.Phase_Shift).astype(self.dtype)

		sin_az = np.sin(rad_azimuth).astype(self.dtype)
		cos_az = np.cos(rad_azimuth).astype(self.dtype)

		self.Axx = -self.defocus_U * cos_az * cos_az - self.defocus_V * sin_az * sin_az
		self.Ayy = -self.defocus_U * sin_az * sin_az - self.defocus_V * cos_az * cos_az
		self.Axy = -self.defocus_U * sin_az * cos_az + self.defocus_V * sin_az * cos_az

	def _build_frequency_grid(self):
		xs = self.XSIZE * self.pixel_size
		ys = self.YSIZE * self.pixel_size

		jp = np.arange(self.XSIZE, dtype=self.dtype) - self.dtype(self.XSIZE) / self.dtype(2.0)
		ip = np.arange(self.YSIZE, dtype=self.dtype) - self.dtype(self.YSIZE) / self.dtype(2.0)

		x = jp / xs
		y = ip / ys

		self.X, self.Y = np.meshgrid(x, y, indexing="xy")
		self.u2 = self.X * self.X + self.Y * self.Y
		self.u4 = self.u2 * self.u2
		self.ss = np.sqrt(self.u2)

	def getCTF(self, do_abs=False, do_only_flip_phases=False,
			   do_intact_until_first_peak=False, do_damping=False):
		gamma = (
			self.K1 * (self.Axx * self.X * self.X + 2.0 * self.Axy * self.X * self.Y + self.Ayy * self.Y * self.Y)
			+ self.K2 * self.u4
			- self.K5
			- self.K3
		)

		if do_intact_until_first_peak:
			retval = np.ones_like(gamma, dtype=self.dtype)
			mask = np.abs(gamma) < self.dtype(np.pi / 2.0)
			retval[mask] = self.dtype(1.0)
			retval[~mask] = -np.sin(gamma[~mask], dtype=self.dtype)
		else:
			retval = -np.sin(gamma, dtype=self.dtype)

		if do_damping:
			retval *= np.exp(self.K4 * self.u2, dtype=self.dtype)

		if do_abs:
			retval = np.abs(retval)
		elif do_only_flip_phases:
			retval = np.where(retval < 0.0, self.dtype(-1.0), self.dtype(1.0))

		retval *= self.Scale
		retval = np.where(np.abs(retval) < self.dtype(1e-8),
						  np.sign(retval) * self.dtype(1e-8), retval)
		return retval
	#	return retval.astype(np.complex64, copy=False)
	def getFftwImage(self, do_abs=False, do_only_flip_phases=False,
					 do_intact_until_first_peak=False, do_damping=False):
		v = self.getCTF(do_abs, do_only_flip_phases,
						   do_intact_until_first_peak, do_damping)
		return v.astype(np.complex64, copy=False)
	def getFftwImage_with_isSPA_weight(self, do_abs=False, do_only_flip_phases=False,
									   do_intact_until_first_peak=False, do_damping=False,
									   kk=3.0):
		a = self.dtype(-9.32)
		b = self.dtype(2.65)
		b2 = self.dtype(0.01908)
		bfactor = self.dtype(-78.7757)
		bfactor2 = self.dtype(-12.9121)
		bfactor3 = self.dtype(1.28732)
		kk = self.dtype(kk)

		signal = np.exp(bfactor * self.ss * self.ss + bfactor2 * self.ss + bfactor3, dtype=self.dtype) / (kk + self.dtype(1.0))
		Ncurve = np.exp(a * self.ss * self.ss + b * self.ss + b2, dtype=self.dtype) / signal

		v = self.getCTF(do_abs, do_only_flip_phases, do_intact_until_first_peak, do_damping)
		v1 = v * np.sqrt(self.dtype(1.0) / (Ncurve + kk * v * v), dtype=self.dtype)
		return v1.astype(np.complex64, copy=False)

# add GisSPA weighting. Right above.

# changelog 20260326
# A much faster version of CTF_cupy. It's 3x faster than the previous version. Thanks to ChatGPT.

# Usage: ctf1=CTF(dfu,dfv,dfa,cs,voltage,apix,xsize,ysize,ac,bf,scale,ps)
#ctf_image=ctf1.getFftwImage(False,False,False,False)

#result in np.complex64

#ctf1=CTF(10000.,5000.,30.)
#print(ctf1.Axx,ctf1.Ayy)
#ctf_data=ctf1.getFftwImage_with_isSPA_weight(False,False,False,False,3.0)
#print("debug, ctf_data.shape=",ctf_data.shape)
#print(ctf_data)
#with mrcfile.new("debug_ctf.mrc",overwrite=True) as mrc1:
#	mrc1.set_data((ctf_data.real).astype(np.float32))
#mrc1.close()
