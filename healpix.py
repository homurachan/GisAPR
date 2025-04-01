import numpy as np

class healpix_py:
	# This is a direct copy of healpix2.15a from RELION. RELION uses Z-Y-Z euler system. 
	# Usage: initialize: A=healpix_py(order), 0<order<=13
	# for i in range(0,A.npix_):
	#	FF=AA.getDirectionFromHealPix(i)
	# FF=(ROTi,TILTi)
	# The third in-plane rotation should be generated from other class.
	def __init__(self,order=4,scheme="NEST"):
		self.order_max=13
		self.jrll=[2,2,2,2,3,3,3,3,4,4,4,4]
		self.jpll=[1,3,5,7,0,2,4,6,1,3,5,7]
		self.ctab = [0] * 256
		self.utab = [0] * 256
		for m in range(256):
			self.ctab[m] = (m & 0x1) | ((m & 0x2) << 7) | ((m & 0x4) >> 1) | ((m & 0x8) << 6) | ((m & 0x10) >> 2) | ((m & 0x20) << 5) | ((m & 0x40) >> 3) | ((m & 0x80) << 4)
			self.utab[m] = (m & 0x1) | ((m & 0x2) << 1) | ((m & 0x4) << 2) | ((m & 0x8) << 3) | ((m & 0x10) << 4) | ((m & 0x20) << 5) | ((m & 0x40) << 6) | ((m & 0x80) << 7)
		assert order >= 0 and order <= self.order_max, "bad order"
		self.order_ = order
		self.nside_ = 1 << order
		self.npface_ = self.nside_ << self.order_
		self.ncap_ = (self.npface_ - self.nside_) << 1
		self.npix_ = 12 * self.npface_
		self.fact2_ = 4.0 / self.npix_
		self.fact1_ = (self.nside_ << 1) * self.fact2_
		self.scheme_ = scheme

	def nest2xyf(self,pix):
		face_num = pix >> (2*self.order_)
		pix &= (self.npface_-1)
		raw = (pix & 0x5555) | ((pix & 0x55550000) >> 15)
		ix = self.ctab[raw & 0xff] | (self.ctab[raw >> 8] << 4)
		pix >>= 1
		raw = (pix & 0x5555) | ((pix & 0x55550000) >> 15)
		iy = self.ctab[raw & 0xff] | (self.ctab[raw >> 8] << 4)
		return (face_num, ix, iy)
		
	def isqrt(self,arg):
		if arg.bit_length() <= 32:
			return int(np.sqrt(arg + 0.5))
		else:
			arg2 = arg
			return int(np.sqrt(arg2 + 0.5))


	def pix2ang_z_phi(self,pix):
		halfpi=np.pi/2.0
		pi=np.pi
		if self.scheme_ == "RING":
			if pix < self.ncap_: # North Polar cap
				iring = int(0.5 * (1 + isqrt(1 + 2 * pix))) #counted from North pole
				iphi = pix + 1 - 2 * iring * (iring - 1)
				z = 1.0 - (iring * iring) * fact2_
				phi = (iphi - 0.5) * halfpi / iring
			elif pix < (self.npix_ - self.ncap_): # Equatorial region
				ip = pix - self.ncap_
				iring = ip // (4 * self.nside_) + self.nside_ # counted from North pole
				iphi = ip % (4 * self.nside_) + 1
				# 1 if iring+nside is odd, 1/2 otherwise
				fodd = 1 if ((iring + self.nside_) & 1) else 0.5

				nl2 = 2 * self.nside_
				z = (nl2 - iring) * self.fact1_
				phi = (iphi - fodd) * pi / nl2
			else: # South Polar cap
				ip = self.npix_ - pix
				iring = int(0.5 * (1 + self.isqrt(2 * ip - 1))) #counted from South pole
				iphi = 4 * iring + 1 - (ip - 2 * iring * (iring - 1))

				z = -1.0 + (iring * iring) * self.fact2_
				phi = (iphi - 0.5) * halfpi / iring
		else:
			nl4 = self.nside_ * 4
			face_num, ix, iy = self.nest2xyf(pix)
			jr = (self.jrll[face_num] << self.order_) - ix - iy - 1
			if jr < self.nside_:
				nr = jr
				z = 1 - nr * nr * self.fact2_
				kshift = 0
			elif jr > 3 * self.nside_:
				nr = nl4 - jr
				z = nr * nr * self.fact2_ - 1
				kshift = 0
			else:
				nr = self.nside_
				z = (2 * self.nside_ - jr) * self.fact1_
				kshift = (jr - self.nside_) & 1

			jp = (self.jpll[face_num] * nr + ix - iy + 1 + kshift) // 2
			if (jp > nl4):
				jp -= nl4
			if (jp < 1):
				jp += nl4
			phi = (jp - (kshift + 1) * 0.5) * (halfpi / nr)
		return (z,phi)

	def getDirectionFromHealPix(self,ipix):
		(zz,phi) = self.pix2ang_z_phi(ipix)
		rot = np.rad2deg(phi)
		tilt = np.rad2deg(np.arccos(zz))
		return (rot,tilt)
'''
#usage:
order=3 #3 means 7.5 degree.
AA=healpix_py(order,"NEST")
MAX=AA.npix_
#print(MAX)
for i in range(0,MAX):
	FF=AA.getDirectionFromHealPix(i)
#	print(FF)
'''
