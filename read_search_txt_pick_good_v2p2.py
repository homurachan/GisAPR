import numpy as np
import matplotlib.pyplot as plt
import random,re,math,os
import copy, pickle,argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
# changelog v2p2
# Experimental multi-threading.
def decode_transform_string(s):
	def convert(val):
		val = val.replace('p', '.')
		return -float(val[1:]) if val.startswith('N') else float(val)

	# Use re.search to find the main transform pattern inside a longer string
	match = re.search(
		r'rot(?P<rot>[Np\d]+)_tilt(?P<tilt>[Np\d]+)_psi(?P<psi>[Np\d]+)deg_trans(?P<x>[Np\d]+)_(?P<y>[Np\d]+)_(?P<z>[Np\d]+)ANG', 
		s
	)
	if not match:
	#	raise ValueError("No transform pattern found in string.")
		rot =0.0
		tilt=0.0
		psi=0.0
		xshift=0.0
		yshift=0.0
		zshift=0.0
		return rot, tilt, psi, xshift, yshift, zshift 
	rot    = convert(match.group('rot'))
	tilt   = convert(match.group('tilt'))
	psi    = convert(match.group('psi'))
	xshift = convert(match.group('x'))
	yshift = convert(match.group('y'))
	zshift = convert(match.group('z'))

	return rot, tilt, psi, xshift, yshift, zshift
def read_textfile_convert_to_numpy(file,dist):
	FL=open(file,"r")
	aa=FL.readlines()
	arr_SN=np.zeros(len(aa)//3,dtype=int)
	arr_CC=np.zeros(len(aa)//3)
	arr_ZS=np.zeros(len(aa)//3)
	arr_DS=np.zeros(len(aa)//3)
	COUNT=0
	for i in range(0,len(aa),3):
		CC_or_PR=str(aa[i].split(',')[0])
		LN=str(aa[i].split(',')[1])
		arr_SN[COUNT]=-1
		arr_CC[COUNT]=0.0
		arr_ZS[COUNT]=0.0
		arr_DS[COUNT]=999
	#	print(CC_or_PR,LN)
		if(CC_or_PR=="['cc: '"):
			Serial_Number = int(aa[i].split(',')[1])
			ZSCORE=float(aa[i].split(',')[2])
			rawCC=float(aa[i+2].split(',')[-1].split(']')[0])/1024.0
			arr_SN[COUNT]=Serial_Number
			arr_CC[COUNT]=rawCC
			arr_ZS[COUNT]=ZSCORE
			arr_DS[COUNT]=dist
		COUNT+=1	
	return arr_SN,arr_CC,arr_ZS,arr_DS
def calc_weight_between_distance_rotation(diameter):
	# diameter in Angstrom
	# this weight multiplies to degrees.
	w0=360.0/np.pi/diameter
	return 1.0/w0
def compute_weighted_overall_distance(rotation_in_degree, translation_in_angstrom,weight):
	final =np.sqrt((rotation_in_degree*weight)**2.0+translation_in_angstrom**2.0)
	return final
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
def calculateShiftDistance(x,y,z,x1,y1,z1):
	A1=(x1-x)**2.0
	A2=(y1-y)**2.0
	A3=(z1-z)**2.0
	s=math.sqrt(A1+A2+A3)
	return s
def judge_relion30_or_relion31(inline):
	trys=3
	RELION30=1
	for i in range (0,trys):
		
		if(inline[i].split()):
			if(len(inline[i].split())>=2):
				if(inline[i].split()[0]=="#"):
					if(len(inline[i].split())>=4):
						if(str(inline[i].split()[3])=="3.0.8"):
							break
				if(int(str(inline[i].split()[2]))>30000):
					RELION30=0
					break
	return RELION30
def judge_mline0(inline):
	trys=60
	intarget=-1
	for i in range (0,trys):
		if(inline[i].split()):
			if(inline[i].split()[0][0]!="_"):
				if(intarget==1):
					return i
					break
				else:
					continue
			if(inline[i].split()[0][0]=="_"):
				intarget=1
def judge_mline1(inline,start):
	trys=70
	intarget=-1
	for i in range (start,trys):
		if(inline[i].split()):
			if(inline[i].split()[0][0]!="_"):
				if(intarget==1):
					return i
					break
				else:
					continue
			if(inline[i].split()[0][0]=="_"):
				intarget=1				
def read_from_selection(data):
	list_obj=[]
	for i in range(len(data)):
		list_obj.append(int(data[i].split()[0]))
	return list_obj
def read_list_sel_search_particles(set_good, data_AA):
	return_str = []
	for i in range(0, len(data_AA), 3):
		SN = int(data_AA[i].split(',')[1])
		if SN in set_good:
			return_str.extend([data_AA[i], data_AA[i+1], data_AA[i+2]])
	return ''.join(return_str)
def process_searchfile(entry,set_good,set_bad,args):
	search_filename = entry.split()[0]
	try:
		with open(search_filename, "r") as AA:
			data_AA = AA.readlines()

		towrite_good = read_list_sel_search_particles(set_good, data_AA)
		towrite_bad = read_list_sel_search_particles(set_bad, data_AA)

		with open(search_filename + args.searchfile_suffix + "_good.txt", 'w') as new_good:
			new_good.write(towrite_good)
		with open(search_filename + args.searchfile_suffix + "_bad.txt", 'w') as new_bad:
			new_bad.write(towrite_bad)
		new_good.close()
		new_bad.close()
	except Exception as e:
		print(f"Error processing {search_filename}: {e}")
# Parallel processing
def create_pick_parser():
	parser = argparse.ArgumentParser(description="Read search files and pick goods and bads.")
	parser.add_argument("--i", type=str, required=True, help="Input file index. Generate it with `ls search*.txt > index.txt`")
	parser.add_argument("--best_string", type=str, required=True, help="The marker string of the best fit. It looks like this: rotN1p0_tilt1p2_psiN3p2deg_trans2p5_N1p2_N2p4ANG . This is needed to compute the relative distance")
	parser.add_argument("--o", type=str, required=True, help="The output prefix.")
#	parser.add_argument("--apix", type=float,default = 1.04, help="Pixel size for the dataset, in Angstrom.")
	parser.add_argument("--diameter", type=float,default = 70, help="Estimated diameter of the subunit, in Angstrom. This is for computing the relative distance. No need to be very accurate.")
	parser.add_argument("--top_N", type=int,default = 5, help="Number of the nearest positions to be account for.")
	parser.add_argument("--top_N_sum_threshold", type=float,default = 1.0, help="Sum threshold for these nearest positions.")
	parser.add_argument("--regenerate_pickel", action='store_true', help="Regenerate the pickel files. default = False")
	parser.add_argument("--do_generate_searchfile", action='store_true', help="Generated the picked search files. default = False")
	parser.add_argument("--searchfile_suffix", type=str,default = "_picked", help="The generated search file prefix.")
	parser.add_argument("--do_pick_star", action='store_true', help="Pick good and bad from given starfile. default = False")
	parser.add_argument("--starfile", type=str,default = None, help="The starfile to be picked. default = None")
	parser.add_argument("--max_cpu", type=int, default=6, help="Maximum number of threads to use")

	return parser
###### 
######
if __name__== "__main__":
	parser = create_pick_parser()
	args = parser.parse_args()
	pickel_filename_To_compute_ZS_serial_score_per_file = args.o+"_To_compute_ZS_serial_score_per_file.pkl"
	pickel_filename_serial_score_per_file = args.o+"_serial_score_per_file.pkl"
	do_regenerate_pkl = args.regenerate_pickel
	if os.path.exists(pickel_filename_To_compute_ZS_serial_score_per_file) and (not do_regenerate_pkl):
		with open(pickel_filename_To_compute_ZS_serial_score_per_file, "rb") as f:
			To_compute_ZS_serial_score_per_file = pickle.load(f)
		with open(pickel_filename_serial_score_per_file, "rb") as f:
			serial_score_per_file = pickle.load(f)
		total_particle_number = len(serial_score_per_file[0])
		
	#	p1=random.randint(50, total_particle_number)
	#	print("Particle number: ",p1)
	#	for i in range(len(serial_score_per_file)):
	#		print(serial_score_per_file[i][p1],To_compute_ZS_serial_score_per_file[i][p1])
	else:
		BEST_FIT_STRING = args.best_string
		pivot_rot, pivot_tilt, pivot_psi, pivot_xshift, pivot_yshift, pivot_zshift = decode_transform_string(BEST_FIT_STRING)
		diameter = args.diameter
		weight = calc_weight_between_distance_rotation(diameter)

		FL=open(args.i,"r")
		data_FL=FL.readlines()
		file_list=[]
		for i in range(len(data_FL)):
			file_list.append(data_FL[i].split()[0])
			
		file_dist_pairs = []
		for file in file_list:
			rot,tilt,psi,xshift,yshift,zshift = decode_transform_string(file)
			rotation_dist = calculateAngularDistance(pivot_rot, pivot_tilt,pivot_psi,rot, tilt,psi)
			trans_dist = calculateShiftDistance(pivot_xshift,pivot_yshift,pivot_zshift,xshift,yshift,zshift)
			overall_dist = compute_weighted_overall_distance(rotation_dist, trans_dist,weight)
			file_dist_pairs.append((file, overall_dist))

		file_dist_pairs.sort(key=lambda x: x[1])

		serial_score_per_file = []
		all_serials = set()
		for pair in file_dist_pairs:
			file = pair[0]
			dist = pair[1]
			arr_SN,arr_CC,arr_ZS,arr_DS = read_textfile_convert_to_numpy(file,dist)

			serial_score_dict = dict(zip(arr_SN, zip(arr_CC, arr_DS)))
			serial_score_per_file.append(serial_score_dict)
			all_serials.update(arr_SN)

		To_compute_ZS_serial_score_per_file = copy.deepcopy(serial_score_per_file)
		total_particle_number = len(serial_score_per_file[0])
		for j in range(total_particle_number):
			aver = 0.0
			SQR_aver = 0.0
			count = 0.0
			for i in range(len(serial_score_per_file)):
				try:
					aver += serial_score_per_file[i][j][0]
					SQR_aver += (serial_score_per_file[i][j][0])**2
					count += 1.0
				except:
					continue
			if(count>0):
				aver /=count
				SQR_aver /=count
				sigma=math.sqrt(SQR_aver-aver**2)
				for i in range(len(serial_score_per_file)):
					ZS = (serial_score_per_file[i][j][0]-aver)/sigma
					dist = serial_score_per_file[i][j][1]
					temp = list(serial_score_per_file[i][j])
					temp[0] = ZS
					temp[1] = dist
					To_compute_ZS_serial_score_per_file[i][j]=tuple(temp)
		with open(pickel_filename_To_compute_ZS_serial_score_per_file, "wb") as f:
			pickle.dump(To_compute_ZS_serial_score_per_file, f)
		with open(pickel_filename_serial_score_per_file, "wb") as f:
			pickle.dump(serial_score_per_file, f)
		FL.close()
	##
	## Start selection.
	print("Total number of particles: ",total_particle_number)
	sel_good_filename = args.o+"_sel_good.txt"
	sel_bad_filename = args.o+"_sel_bad.txt"
	sel_good=open(sel_good_filename,"w")
	sel_bad=open(sel_bad_filename,"w")
	MAX=args.top_N
	COUNT_good = 0.0
	COUNT_bad = 0.0
	if(MAX > len(serial_score_per_file)):
		print("Warning! --top_N is more than the actual number of search files.")
		MAX = len(serial_score_per_file)
	for N in range(total_particle_number):
		SUM_N_Particle = 0.0
		try:
			for i in range(0,MAX):
				SUM_N_Particle += To_compute_ZS_serial_score_per_file[i][N][0]
			if(SUM_N_Particle >= args.top_N_sum_threshold):
				sel_good.write(str(N)+"\n")
				COUNT_good += 1.0
			else:
				sel_bad.write(str(N)+"\n")
				COUNT_bad += 1.0
		except:
			continue
	sel_good.close()
	sel_bad.close()
	print(f"Good = {COUNT_good}, bad = {COUNT_bad}")
	if(args.do_pick_star):
		starfile=open(args.starfile,'r')
		sel_good=open(sel_good_filename,"r")
		sel_bad=open(sel_bad_filename,"r")
		data_good=sel_good.readlines()
		data_bad=sel_bad.readlines()
		instar_line=starfile.readlines()
		relion30=1
		relion30=judge_relion30_or_relion31(inline=instar_line)
		print ("Is target star relion3.0? = "+str(relion30))
		mline=-1
		if(relion30):
			mline=judge_mline0(inline=instar_line)
		else:
			#regard as relion3.1
			MLINE=judge_mline0(inline=instar_line)
			mline=judge_mline1(inline=instar_line,start=MLINE)
		if(mline<0):
			print ("Particle starfile error or this script cannot handle it. EXIT.")
			quit()
		print ("star mline = "+str(mline))
		out_good_pick_filename=args.o+"_goodpick.star"
		out_bad_pick_filename=args.o+"_badpick.star"
		out_good_pick=open(out_good_pick_filename,'w')
		out_bad_pick=open(out_bad_pick_filename,'w')
		for i in range(0,mline):
			out_good_pick.write(instar_line[i])
			out_bad_pick.write(instar_line[i])
		for i in range(0,len(data_good)):
			NUM=int(data_good[i].split()[0])
			out_good_pick.write(instar_line[NUM])
		for i in range(0,len(data_bad)):
			NUM=int(data_bad[i].split()[0])
			out_bad_pick.write(instar_line[NUM])
		out_good_pick.close()
		out_bad_pick.close()
		starfile.close()
		sel_good.close()
		sel_bad.close()
	
	if args.do_generate_searchfile:
		with open(args.i, "r") as FL:
			data_FL = FL.readlines()
		with open(sel_good_filename, "r") as sel_good:
			list_good = read_from_selection(sel_good.readlines())
		with open(sel_bad_filename, "r") as sel_bad:
			list_bad = read_from_selection(sel_bad.readlines())
		
		set_good = set(list_good)
		set_bad = set(list_bad)
	#	process_searchfile(data_FL,set_good,set_bad,args)
		wrapped_func = partial(process_searchfile, set_good=set_good, set_bad=set_bad, args=args)
		with ThreadPoolExecutor(max_workers=args.max_cpu) as executor:
			executor.map(wrapped_func, data_FL)