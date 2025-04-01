import numpy as np
import math
def gcd(m, n):
	while n != 0:
		m, n = n, m % n
	return m

def eq_sphere(nMarker):
	PI = np.pi
	Point = np.zeros((nMarker, 3), dtype=np.float64)

	if nMarker < 1:
		raise ValueError(f"Error in eq_sphere, nMarker wrong: {nMarker}")
	if nMarker < 3:
		Point[0, :] = [0.0, 0.0, 1.0]
		if nMarker == 1:
			return Point
		Point[1, :] = [1.0, 0.0, 0.0]
		if nMarker < 4:
			return Point
		Point[-1, :] = [0.0, 0.0, -1.0]

	area_cap = 4.0 * PI / nMarker
	c_polar = 2.0 * np.arcsin(np.sqrt(area_cap / (4.0 * PI)))
	n_collars = max(1, int((PI - 2.0 * c_polar) / np.sqrt(area_cap)))
	n_regions = np.zeros(n_collars + 2, dtype=np.int32)
	n_regions[0] = 1
	n_regions[-1] = 1
	discrepancy = 0.0
	area_top = area_cap
	a_fitting = (PI - 2.0 * c_polar) / n_collars

	for k in range(n_collars):
		area_bot = c_polar + (k + 1) * a_fitting
		area_bot = np.sin(0.5 * area_bot)
		area_bot = 4 * PI * area_bot * area_bot
		r_regions = (area_bot - area_top) / area_cap
		area_top = area_bot
		n_regions[k + 1] = int(r_regions + discrepancy)
		discrepancy += r_regions - n_regions[k + 1]

	nCount = 2
	offset = 0.0
	a_top = c_polar
	area_tot = area_cap

	for k in range(n_collars):
		n_top = n_regions[k + 1]
		n_bot = n_regions[k + 2]
		area_tot += n_top * area_cap
		a_bot = 2.0 * np.arcsin(np.sqrt(area_tot / (4.0 * PI)))
		Psi = 0.5 * (a_top + a_bot)
		a_top = a_bot

		for m in range(1, n_top + 1):
			aTemp = (2 * m - 1) * PI / n_top + 2.0 * PI * offset
			Phi = aTemp - 2.0 * PI * np.floor(aTemp / (2.0 * PI))
			rx = np.sin(Psi) * np.cos(Phi)
			ry = np.sin(Psi) * np.sin(Phi)
			rz = np.cos(Psi)
			rnorm = np.sqrt(rx * rx + ry * ry + rz * rz)
			Point[nCount, :] = [rx, ry, rz] / rnorm
			nCount += 1

		offset += (n_top - n_bot + gcd(n_top, n_bot)) / (2 * n_top * n_bot)
		offset -= np.floor(offset)

#	if nMarker != nCount:
#		print(f"Error in eq_sphere: nMarker != nCount: {nMarker} != {nCount}")
		#raise ValueError(f"Error in eq_sphere: nMarker != nCount: {nMarker} != {nCount}")

	return Point

def SQR(x):
	y=float(x)
	return(y*y)
def Euler_angles2direction(alpha, beta):

	alpha = DEG2RAD(alpha)
	beta = DEG2RAD(beta)
	v=[]
	for i in range(0,3):
		v.append([])
	ca = math.cos(alpha)
	cb = math.cos(beta)
	sa = math.sin(alpha)
	sb = math.sin(beta)
	sc = sb * ca
	ss = sb * sa

	v[0]= sc
	v[1] = ss
	v[2] = cb
	return v

def DEG2RAD(x):
	return(x/180.0*3.14159265359)
def Euler_angles2matrix(alpha, beta, gamma):
	alpha = DEG2RAD(alpha)
	beta  = DEG2RAD(beta)
	gamma = DEG2RAD(gamma)
	ca =  math.cos(alpha)
	cb =  math.cos(beta)
	cg =  math.cos(gamma)
	sa =  math.sin(alpha)
	sb =  math.sin(beta)
	sg =  math.sin(gamma)
	cc =  cb * ca
	cs =  cb * sa
	sc =  sb * ca
	ss =  sb * sa
	A=[]
	for i in range(0,3):
		A.append([])
		for j in range(0,3):
			A[i].append([])
	A[0][0] =  cg * cc - sg * sa
	A[0][1] =  cg * cs + sg * ca
	A[0][2] = -cg * sb
	A[1][0] = -sg * cc - cg * sa
	A[1][1] = -sg * cs + cg * ca
	A[1][2] = sg * sb
	A[2][0] =  sc
	A[2][1] =  ss
	A[2][2] = cb
	return A

def calculateAngularDistance(rot1, tilt1,psi1,rot2, tilt2,psi2):

#	direction1=Euler_angles2direction(alpha=rot1, beta=tilt1)
#	direction2=Euler_angles2direction(alpha=rot2, beta=tilt2)
	min_axes_dist = 3600.0

	E1=Euler_angles2matrix(alpha=rot1, beta=tilt1, gamma=psi1)
	E2=Euler_angles2matrix(alpha=rot2, beta=tilt2, gamma=psi2)
	v1=[]
	v2=[]
	axes_dist = 0;
	for i in range(0,3):
		v1=E1[i]
		v2=E2[i]
		axes_dist += math.acos(CLIP(a=dotProduct(v1, v2),b=-1., c=1.))*180.0/3.14159265359
	axes_dist=axes_dist/3.0
	if (axes_dist < min_axes_dist):
		min_axes_dist = axes_dist
	return min_axes_dist
def selfNormalize(v):
	tmp=0.0
	for i in range(0,len(v)):
		tmp+=SQR(v[i])
	tmp=math.sqrt(tmp)
	if (tmp>1E-6):
		for i in range(0,len(v)):
			v[i]/=tmp
	else:
		for i in range(0,len(v)):
			v[i]=0.0
	return v
def Euler_direction2angles(v0):

#	v[0]=sb * ca,v[1]=sb * sa,v[2]=math.cos(beta)
	v=selfNormalize(v0)
#	print "debug ",v
	alpha = math.degrees(math.atan2(v[1], v[0]))
	beta =  math.degrees(math.acos(v[2]))

	if ((math.fabs(beta) < 0.001) or (math.fabs(beta - 180.) < 0.001)):
		alpha = 0.;

	return (alpha,beta)

def CLIP(a,b,c):
	if(float(a)<float(b)):
		return float(b)
	else:
		if(float(a)>float(c)):
			return float(c)
		return float(a)
def dotProduct(v1,v2):
	if(len(v1)!=len(v2)):
		return -9999999.0
	sum=0.0
	for i in range(0,len(v1)):
		sum+=float(v1[i])*float(v2[i])
	return sum
def VectorProduct(v1,v2):
	v=[]
	for i in range(0,3):
		v.append([])
		v[i]=v1[i]*v2[i]
	return v
def eqps_with_input_step_return_relion_rot_tilt(step0):
#	step0= float(input('step in deg ='))
	step = step0* math.pi/180.
	N = int(4*math.pi/(step*step))
#	print(N)
#	nMarker = int(input('nMarker='))
	Point = eq_sphere(N)
	rot=[]
	tilt=[]
#	print('    nid,      x,      y,      z')
	for k in range(1,N-1):
		V=[Point[k, 0],Point[k, 1],Point[k, 2]]
		ROT,TILT=Euler_direction2angles(V)
		rot.append(ROT)
		tilt.append(TILT)
	#	print(f'{k + 1}, {Point[k, 0]:.6f}, {Point[k, 1]:.6f}, {Point[k, 2]:.6f}')
#		print(k,rot,tilt)
	return rot,tilt
#if __name__ == "__main__":
#	main()
'''
ap=open("anglelist_fortest.txt","w")
rot,tilt=eqps_with_input_step_return_relion_rot_tilt(5.0)
for i in range(len(rot)):
	ap.write(str(rot[i])+"\t"+str(tilt[i])+"\n")
ap.close()
'''