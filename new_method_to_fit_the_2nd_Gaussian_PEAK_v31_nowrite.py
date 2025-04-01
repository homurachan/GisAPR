import numpy as np
#import mrcfile
import os,sys,math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from scipy.stats import norm
# changelog ver30
# In this version, I adapt the MLE into the scores.
# Basic equation is: ln(likelihood) = 1/SQR(sigma)*SUM(CC)
# But how can we get the sigma? I think sigma is the width of noise peak, obtained by fitting the noise peak.
# SUM(CC) can be calculated by splitting the signal peak into two parts.
# The first part contains the values those are larger than the peak position. Just some simple sum.
# However, the left part is way more complicated. It turns out to be number_of_particles_to_the_left * (peak_positions - 0.7978).
# We regard number_of_particles_to_the_left being equal the number of particles on the right side of the signal peak.

# changelog ver31
# Move the total_estimated_sum_cc to the first column in order to be read by the following scrits.
try:
	from optparse import OptionParser
except:
	from optik import OptionParser
def gaussian(x, amp, mean, stddev):
	return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))
def double_gaussian(x, amp1, mean1, stddev1, amp2, mean2, stddev2):
	return gaussian(x, amp1, mean1, stddev1) + gaussian(x, amp2, mean2, stddev2)
	
def main():
	(results_file, starfile,output_root,CC_THRES,output_filename) = parse_command_line()
	bb=open(output_filename,'w')
	a=open(results_file,'r')
	aa=a.readlines()
	print("debug,",len(results_file))
	for i in range(0,len(aa)):
		filename=aa[i].split()[0]
		if(i%1==0):
			bb.write(filename+"\n")
		thres,CC_value,ZSCORE_value,total_estimated_sum_cc=run_this(filename, starfile,output_root,CC_THRES)
		bb.write(str(total_estimated_sum_cc)+" "+str(ZSCORE_value)+" "+str(CC_value)+"\n")
		
	bb.close()
	a.close()
def run_this(results_file, starfile,output_root,CC_THRES):
	print(results_file)
	Replace_bad_rot_tilt=False
	do_write_revised=False
#	CC_THRES=6.0
	DISTANCE_THRES=11.0
	a=open(results_file,'r')
	aa=a.readlines()
	star=open(starfile,'r')
	star_line=star.readlines()
	p_relion30=1
	p_relion30=judge_relion30_or_relion31(inline=star_line)
	print ("Is target star relion3.0? = "+str(p_relion30))
	p_mline=-1
	if(p_relion30):
		p_mline=judge_mline0(inline=star_line)
	else:
		#regard as relion3.1
		p_MLINE=judge_mline0(inline=star_line)
		p_mline=judge_mline1(inline=star_line,start=p_MLINE)
	if(p_mline<0):
		print ("Particle starfile error or this script cannot handle it. EXIT.")
		quit()
	for i in range(0,p_mline):
		if (star_line[i].split()):
			if (str(star_line[i].split()[0])=="_rlnAngleRot"):
				ROT_index=int(star_line[i].split('#')[1])-1
			if (str(star_line[i].split()[0])=="_rlnAngleTilt"):
				TILT_index=int(star_line[i].split('#')[1])-1
			if (str(star_line[i].split()[0])=="_rlnAnglePsi"):
				PSI_index=int(star_line[i].split('#')[1])-1
	print ("star mline = "+str(p_mline))
#	Name_Method1=output_root+"_cc_good_pick.star"
#	output_Method1=open(Name_Method1,'w')
#	Name_bad=output_root+"_cc_bad_pick.star"
#	output_bad=open(Name_bad,'w')
	###############
	# during the picking process, we only use the correct orientations. (For testing purpose only.)
#	for i in range(0,p_mline):
#		output_Method1.write(star_line[i])
#		output_bad.write(star_line[i])
	CASE1_CORR=0.
	CASE1_TOTAL=0.

	CASE2_CORR=0.
	CASE2_TOTAL=0.

	DIS_TOTAL=0.
	DIS_CORR_TOTAL=0.
	DIS_CORR_NUM=0.
	DIS_NUM=0.
	
	PSI_DISTANCE_TOTAL=0.
	PSI_DISTANCE_TOTAL_SQR=0.
	PSI_DISTANCE_NUM=0.
	PSI_DIS_CORR_TOTAL=0.
	PSI_DIS_CORR_NUM=0.
	PSI_DIS_CORR_TOTAL_SQR=0.
	PSI_DIS_BAD_TOTAL=0.
	PSI_DIS_BAD_NUM=0.
	PSI_DIS_BAD_TOTAL_SQR=0.
	Case=0
	CC_ARRAY=[]
	raw_CC_array=[]
	# Note: The raw CC value was multiplied by 1024. So we should divide by it.
	for i in range(0,len(aa)):
		CC_or_PR=str(aa[i].split(',')[0])
		if(CC_or_PR=="['cc: '"):
			Serial_Number = int(aa[i].split(',')[1])
			CC_=float(aa[i].split(',')[2])
			CC_L3=int(aa[i].split(',')[3])
			CC_ACCU=int(aa[i].split(',')[4])
			new_rot=float(aa[i].split(',')[5])
			new_tilt=float(aa[i].split(',')[6])
			DIS_CC=float(aa[i].split(',')[7])
			X=float(aa[i].split(',')[8])
			Y=float(aa[i].split(',')[9])
			CX=float(aa[i].split(',')[10])
			CY=float(aa[i].split(',')[-1].split(']')[0])
			DisTance=distance(X,Y,CX,CY)
			DIS_TOTAL+=DisTance
			DIS_NUM+=1.0
			if(CC_<9999):
				CC_ARRAY.append(CC_)
			if(CC_>CC_THRES):
				Case=1
				DIS_CORR_TOTAL+=DisTance
				DIS_CORR_NUM+=1.0
			else:
				Case=2
			
			if(Case==1):
				if(DIS_CC<DISTANCE_THRES):
					CASE1_CORR+=1
					
				CASE1_TOTAL+=1.
			#	if(Serial_Number>=p_mline):
				#	output_Method1.write(star_line[Serial_Number])
			if(Case==2):
				if(DIS_CC<DISTANCE_THRES):
					CASE2_CORR+=1.
				CASE2_TOTAL+=1.
				if(Serial_Number>=p_mline):
					if(Replace_bad_rot_tilt):
						rot=str(star_line[Serial_Number].split()[ROT_index])
						tilt=str(star_line[Serial_Number].split()[TILT_index])
						towrite=""
						for j in range(0,len(star_line[Serial_Number].split())):
							if(j!=ROT_index and j!=TILT_index):
								towrite+=str(star_line[Serial_Number].split()[j])
							if(j==ROT_index):
								towrite+=str(new_rot)
							if(j==TILT_index):
								towrite+=str(new_tilt)
							if(j<len(star_line[Serial_Number].split())-1):
								towrite+="\t"
							if(j>=len(star_line[Serial_Number].split())-1):
								towrite+="\n"
					#	output_bad.write(towrite)
				#	else:
					#	output_bad.write(star_line[Serial_Number])
		
		if(CC_or_PR=="['psi: '"):
			Serial_Number = int(aa[i].split(',')[1])
			PSI=float(aa[i].split(',')[2])
			PSI_ORI=float(aa[i].split(',')[-1].split(']')[0])
			PSI_ORI=(PSI_ORI+360.)%360.
			PSI_DISTANCE=(math.fabs(PSI-PSI_ORI)+360.)%360.
		#	print(PSI_DISTANCE)
			PSI_DISTANCE_TOTAL+=PSI_DISTANCE
			PSI_DISTANCE_TOTAL_SQR+=PSI_DISTANCE*PSI_DISTANCE
			PSI_DISTANCE_NUM+=1.
			if(Case==1):
				PSI_DIS_CORR_TOTAL+=PSI_DISTANCE
				PSI_DIS_CORR_NUM+=1.
				PSI_DIS_CORR_TOTAL_SQR+=PSI_DISTANCE*PSI_DISTANCE
			if(Case==2):
				PSI_DIS_BAD_TOTAL+=PSI_DISTANCE
				PSI_DIS_BAD_NUM+=1.
				PSI_DIS_BAD_TOTAL_SQR+=PSI_DISTANCE*PSI_DISTANCE
		if(CC_or_PR=="['rawCC: '"):
			Serial_Number = int(aa[i].split(',')[1])
			rawCC=float(aa[i].split(',')[-1].split(']')[0])/1024.0
			raw_CC_array.append(rawCC)
			
#	output_Method1.close()
#	output_bad.close()

	ave_PSI_DISTANCE=PSI_DISTANCE_TOTAL/PSI_DISTANCE_NUM
	ave_PSI_DIS_CORR=PSI_DIS_CORR_TOTAL/PSI_DIS_CORR_NUM
	ave_PSI_DIS_BAD=PSI_DIS_BAD_TOTAL/PSI_DIS_BAD_NUM
	a.close()
#	print("PSI: average error=",ave_PSI_DISTANCE,", s.d.=",math.sqrt(PSI_DISTANCE_TOTAL_SQR/PSI_DISTANCE_NUM-ave_PSI_DISTANCE*ave_PSI_DISTANCE))
	print("PSI: significant average error=",ave_PSI_DIS_CORR,", s.d.=",math.sqrt(PSI_DIS_CORR_TOTAL_SQR/PSI_DIS_CORR_NUM-ave_PSI_DIS_CORR*ave_PSI_DIS_CORR))
	print("PSI: bad average error=",ave_PSI_DIS_BAD,", s.d.=",math.sqrt(PSI_DIS_BAD_TOTAL_SQR/PSI_DIS_BAD_NUM-ave_PSI_DIS_BAD*ave_PSI_DIS_BAD))
	if(CASE1_TOTAL>0):
		print("CC significant Total=", CASE1_TOTAL,", corr=",CASE1_CORR/CASE1_TOTAL,", good number=",CASE1_CORR)
		print("Translation significant aver=",DIS_CORR_TOTAL/DIS_CORR_NUM,", Full aver=",DIS_TOTAL/DIS_NUM)
	if(CASE2_TOTAL>0):
		print("CC bad Total=", CASE2_TOTAL,", corr=",CASE2_CORR/CASE2_TOTAL)
	data=np.asarray(CC_ARRAY)
	# data is the z-score array.
	CC_data=np.asarray(raw_CC_array)
	First_Peak_POSI=-99999.
	First_Peak_SIGMA=-99999.
	Second_Peak_POSI=-99999.
	Second_Peak_SIGMA=-99999.
	REMOVE_POSITION=0.140
	REMOVE_SIGMA=0.07
	### REMOVE_SIGMA is a guess. Should be manually adjusted.
	if(len(CC_data>0)):
		hist_CC, bin_edges_CC = np.histogram(CC_data, bins=1000,range=(0,1.0), density=True)
		smoothed_hist_CC = gaussian_filter(hist_CC, sigma=2)
		bin_centers_CC = (bin_edges_CC[:-1] + bin_edges_CC[1:]) / 2
		GUESS_PEAK_CC=bin_centers_CC[np.unravel_index(np.argmax(smoothed_hist_CC),smoothed_hist_CC.shape)]
		initial_guess_CC = [max(smoothed_hist_CC), GUESS_PEAK_CC, 0.02, max(smoothed_hist_CC)/2.0,GUESS_PEAK_CC+0.05, 0.1]
		params_CC, covariance_CC = curve_fit(double_gaussian, bin_centers_CC, smoothed_hist_CC, p0=initial_guess_CC)
		print("params_CC: ",params_CC)
		x_values_CC = np.linspace(bin_edges_CC[0], bin_edges_CC[-1], 1000)
		fitted_curve_CC = double_gaussian(x_values_CC, *params_CC)
		####### newly add
		First_Peak_POSI=np.fabs(params_CC[1])
		First_Peak_SIGMA=np.fabs(params_CC[2])
		Second_Peak_POSI=np.fabs(params_CC[4])
		Second_Peak_SIGMA=np.fabs(params_CC[5])
		'''
		plt.hist(CC_data, bins=1000,range=(0,1.0), density=True, alpha=0.6, color='blue', edgecolor='black')
		plt.xlim(0,0.3)
		plt.ylim(0,15)
		# Plot the fitted Gaussian curves
		plt.plot(x_values_CC, gaussian(x_values_CC, *params_CC[:3]), 'r--', label='CC Gaussian 1')
		plt.plot(x_values_CC, gaussian(x_values_CC, *params_CC[3:]), 'b--', label='CC Gaussian 2')
		plt.plot(x_values_CC, smoothed_hist_CC, 'r-', label='Smoothed histogram 2')
		plt.plot(x_values_CC, double_gaussian(x_values_CC, *params_CC), 'g-', label='CC Double Gaussian Fit')

		# Add titles and labels
		plt.title(str(output_root)+', CC Histogram and Gaussian Fit')
		plt.xlabel('CC value')
		plt.ylabel('Density')
		plt.legend()

		# Show the plot
		plt.show()
		'''
		derivative_values_CC = np.gradient(fitted_curve_CC, x_values_CC)
		# Find initial guesses for zero-points where the sign of the derivative changes
		sign_changes_CC = np.where(np.diff(np.sign(derivative_values_CC)))[0]
		initial_guesses_CC = x_values_CC[sign_changes_CC]

		# Define a function for the derivative
		def derivative_func_CC(x):
			return np.interp(x, x_values_CC, derivative_values_CC)

		# Refine the zero-points of the derivative
		zero_points_CC = []
		for guess in initial_guesses_CC:
			zero_point_CC = fsolve(derivative_func_CC, guess)
			zero_points_CC.append(zero_point_CC[0])

		# Remove duplicate and out-of-bound solutions
		zero_points_CC = np.unique(zero_points_CC)
		zero_points_CC = zero_points_CC[(zero_points_CC >= x_values_CC[0]) & (zero_points_CC <= x_values_CC[-1])]
		print("The zero points in rawCC are: "+str(zero_points_CC))
		# It looks like the third Zero-point is a stable way to distinguish the structures.
		
		# Adjust the parameters as needed (prominence helps filter out small fluctuations)
		peaks, properties = find_peaks(smoothed_hist_CC, prominence=0.01, distance=20,width=5)

		# Get the peak positions and heights
		peak_positions = bin_centers_CC[peaks]
		peak_heights = smoothed_hist_CC[peaks]

		print("Detected peak positions:", peak_positions)
		print("Peak heights:", peak_heights)
		'''
		# Plot the peaks on the histogram
		plt.figure(figsize=(10, 6))
		plt.plot(bin_centers_CC, smoothed_hist_CC, label='Smoothed Histogram')
		plt.plot(peak_positions, peak_heights, 'rx', label='Detected Peaks')
		plt.xlabel('Bin Centers')
		plt.ylabel('Smoothed Histogram Values')
		plt.title('Detected Peaks in Smoothed Histogram')
		plt.legend()
		plt.show()
		'''
		refined_x = params_CC[4]
		number_larger_than_peak = np.sum(data > refined_x)
		left_estimated_cc= number_larger_than_peak*(refined_x - 0.7978)
		right_estimated_cc = np.sum(data[data > refined_x])
		total_estimated_sum_cc = left_estimated_cc+right_estimated_cc
		print("total_estimated_sum_cc = ",total_estimated_sum_cc)
		return 0.0,refined_x,0.0,total_estimated_sum_cc
		# Step 3: Refine the peak positions using quadratic fitting
		def refine_peak_position(x, y, peak_index, window_size=5):
			# Extract local data around the peak
			start = max(peak_index - window_size, 0)
			end = min(peak_index + window_size + 1, len(x))
			x_local = x[start:end]
			y_local = y[start:end]

			# Fit a quadratic curve
			coefficients = np.polyfit(x_local, y_local, 2)
			a, b, c = coefficients

			# The vertex of the parabola gives the peak position
			peak_x = -b / (2 * a)
			peak_y = a * peak_x ** 2 + b * peak_x + c

			return peak_x, peak_y

		# Check if at least two peaks are detected
		# width=5 in find_peaks elimates the weird last peak.
		if len(peaks) >= 2:
			second_peak_index = peaks[-1]
		#	print("debug, second_peak_index,zero_points_CC[-1] = ",bin_centers_CC[second_peak_index],zero_points_CC[-1])
			if(np.fabs(bin_centers_CC[second_peak_index]-zero_points_CC[2])>0.003):
				refined_x = zero_points_CC[2]
				refined_y = -1.
				print("Distance between find_peaks and fitting is too large. Use the gaussian fitting result.")
			else:
				refined_x, refined_y = refine_peak_position(bin_centers_CC, smoothed_hist_CC, second_peak_index)
			print("Refined second peak position:", refined_x)
			print("Refined second peak height:", refined_y)
		else:
			print("Less than two peaks detected.")
		'''
		# Plot the refined peak on the histogram
		plt.figure(figsize=(10, 6))
		plt.plot(bin_centers_CC, smoothed_hist_CC, label='Smoothed Histogram')
		plt.plot(peak_positions, peak_heights, 'rx', label='Detected Peaks')
		plt.plot(refined_x, refined_y, 'go', label='Refined Second Peak')
		plt.xlabel('Bin Centers')
		plt.ylabel('Smoothed Histogram Values')
		plt.title('Refined Second Peak Position')
		plt.legend()
		plt.show()
		'''
	#### ver 30 mod starts
	number_larger_than_peak = np.sum(data > refined_x)
	left_estimated_cc= number_larger_than_peak*(refined_x - 0.7978)
	right_estimated_cc = np.sum(data[data > refined_x])
	total_estimated_sum_cc = left_estimated_cc+right_estimated_cc
	print("total_estimated_sum_cc = ",total_estimated_sum_cc)
	# Don't apply divide by noise_sigma^2 here. 
	#### ver 30 mod ends
	
	hist, bin_edges = np.histogram(data, bins=300, density=True)
	
	######################## wavelet
	smoothed_hist = gaussian_filter(hist, sigma=2)
#	print(smoothed_hist)
	bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#	print(bin_centers)
	guess_peak=bin_centers[np.unravel_index(np.argmax(smoothed_hist),smoothed_hist.shape)]
	initial_guess = [max(smoothed_hist),guess_peak , 0.3, max(smoothed_hist)/4.0, guess_peak+2, 4]
#	print(initial_guess)
	# initial_guess = [amp1, mean1, sd1, amp2, mean2, sd2]
	params, covariance = curve_fit(double_gaussian, bin_centers, smoothed_hist, p0=initial_guess)
#	print("parameters: ",params)
	x_values = np.linspace(bin_edges[0], bin_edges[-1], 1000)
	fitted_curve = double_gaussian(x_values, *params)
	derivative_values = np.gradient(fitted_curve, x_values)
	# Find initial guesses for zero-points where the sign of the derivative changes
	sign_changes = np.where(np.diff(np.sign(derivative_values)))[0]
	initial_guesses = x_values[sign_changes]

	# Define a function for the derivative
	def derivative_func(x):
		return np.interp(x, x_values, derivative_values)

	# Refine the zero-points of the derivative
	zero_points = []
	for guess in initial_guesses:
		zero_point = fsolve(derivative_func, guess)
		zero_points.append(zero_point[0])

	# Remove duplicate and out-of-bound solutions
	zero_points = np.unique(zero_points)
	zero_points = zero_points[(zero_points >= x_values[0]) & (zero_points <= x_values[-1])]
#	print("The zero points are: "+str(zero_points))
#	if(len(zero_points)>2):
#		print("Suggested Z-score threshold = "+str(zero_points[1]))
#	else:
#		print("There is only one or two zero-points of your Z-score curve, check the figure for best threshold.")
#		print("Or you can adjust the initial_guess in this script.")
	Second_Peak_POSI=refined_x
	if(do_write_revised):
		print("do_write_revised, params: ",First_Peak_POSI,First_Peak_SIGMA,Second_Peak_POSI,Second_Peak_SIGMA,REMOVE_POSITION,REMOVE_SIGMA)
		SUM_WEIGHT=0.0
		SUM_NUM=0.0
		NEW_FILE_NAME=results_file+"_newRevised.txt"
		XX=open(NEW_FILE_NAME,"w")
		for i in range(0,len(aa),3):
			CC_or_PR=str(aa[i].split(',')[0])
			LN=str(aa[i].split(',')[1])
			
			if(CC_or_PR=="['cc: '"):
				ZSCORE=float(aa[i].split(',')[2])
				rawCC=float(aa[i+2].split(',')[-1].split(']')[0])/1024.0
				MULTI=calculateProbability(rawCC,ZSCORE,First_Peak_POSI,First_Peak_SIGMA,Second_Peak_POSI,Second_Peak_SIGMA,REMOVE_POSITION,REMOVE_SIGMA)
				SUM_WEIGHT+=MULTI
				SUM_NUM+=1.0
			if(LN==" 'cc: '"):
				ZSCORE=float(aa[i].split(',')[3])
				Serial_Number = int(aa[i+2].split(',')[2])
				rawCC=float(aa[i+2].split(',')[-1].split(']')[0])/1024.0
				MULTI=calculateProbability(rawCC,ZSCORE,First_Peak_POSI,First_Peak_SIGMA,Second_Peak_POSI,Second_Peak_SIGMA,REMOVE_POSITION,REMOVE_SIGMA)
				SUM_WEIGHT+=MULTI
				SUM_NUM+=1.0
		print("debug, SUM_WEIGHT,SUM_NUM=",SUM_WEIGHT,SUM_NUM)
		for i in range(0,len(aa),3):
			CC_or_PR=str(aa[i].split(',')[0])
			LN=str(aa[i].split(',')[1])
		#	print(CC_or_PR,LN)
			if(CC_or_PR=="['cc: '"):
				Serial_Number = int(aa[i].split(',')[1])
				ZSCORE=float(aa[i].split(',')[2])

				rawCC=float(aa[i+2].split(',')[-1].split(']')[0])/1024.0
				
				MULTI=calculateProbability(rawCC,ZSCORE,First_Peak_POSI,First_Peak_SIGMA,Second_Peak_POSI,Second_Peak_SIGMA,REMOVE_POSITION,REMOVE_SIGMA)
			#	print("debug, MULTI/(SUM_WEIGHT/SUM_NUM)= ",MULTI/(SUM_WEIGHT/SUM_NUM))
				rawCC_weighted=rawCC*MULTI/(SUM_WEIGHT/SUM_NUM)*1024.0
				XX.write(aa[i])
				XX.write(aa[i+1])
			#	F1.append(["rawCC: ",Serial_Number,rawCC_weighted])
				XX.write(str("['rawCC: ', ")+str(Serial_Number)+", "+str(rawCC_weighted)+"]\n")
			if(LN==" 'cc: '"):
				Serial_Number = int(aa[i].split(',')[2])
				ZSCORE=float(aa[i].split(',')[3])

				Serial_Number = int(aa[i+2].split(',')[2])
				rawCC=float(aa[i+2].split(',')[-1].split(']')[0])/1024.0
				MULTI=calculateProbability(rawCC,ZSCORE,First_Peak_POSI,First_Peak_SIGMA,Second_Peak_POSI,Second_Peak_SIGMA,REMOVE_POSITION,REMOVE_SIGMA)
				rawCC_weighted=rawCC*MULTI/(SUM_WEIGHT/SUM_NUM)*1024.0
				XX.write(aa[i])
				XX.write(aa[i+1])
			#	F1.append(["rawCC: ",Serial_Number,rawCC_weighted])
				XX.write(aa[i+2].split(',')[0]+", 'rawCC: ', "+str(Serial_Number)+", "+str(rawCC_weighted)+"]\n")
		XX.close()

	
	a.close()
	
	if(len(peaks) >= 2):
		return zero_points[1],refined_x,params[4],total_estimated_sum_cc
	else:
		if(len(zero_points)>=2):
			return zero_points[1],params_CC[4],params[4],total_estimated_sum_cc
		else:
			return zero_points[-1],params_CC[4],params[4],total_estimated_sum_cc

def calculateProbability(rawCC,ZSCORE,First_Peak_POSI,First_Peak_SIGMA,Second_Peak_POSI,Second_Peak_SIGMA,REMOVE_POSITION,REMOVE_SIGMA):
	probability=0.0
	# Math: 1st peak marks the prob of trash, Realprob = 1-prob.
	# 2nd peak marks the prob of truth. Realprob = prob.
	# The prob of structures is the position of Second_Peak_POSI related to REMOVE_POSITION. Regard it having a Gaussian distribution.
	### TO CONSIDER: Should I normalize all the three probs seperately?
	FP_prob_Z = (rawCC-First_Peak_POSI)/First_Peak_SIGMA
	FP_prob = norm.cdf(FP_prob_Z)
	# It should be (1- (1-norm.cdf(FP_prob_Z))) = norm.cdf(FP_prob_Z). We just regard the position as a lowest boundary. So the prob = P(X > FP_prob_Z).
	SP_prob_Z = (rawCC-Second_Peak_POSI)/Second_Peak_SIGMA
	SP_prob = norm.cdf(SP_prob_Z)
	# Same here.
	RM_prob_Z = (Second_Peak_POSI-REMOVE_POSITION)/REMOVE_SIGMA
	RM_prob = norm.cdf(RM_prob_Z)
	
	probability = FP_prob * SP_prob * RM_prob
	return probability

def manual_integration(arr,upper_limit,d):
	SUM=0.
	for i in range(len(arr)):
		if(arr[i]<=upper_limit):
			SUM+=np.fabs(arr[i])
	return SUM*d
def parse_command_line():
	usage="%prog <Results file> <starfile> <output rootname> <CC threshold> <output filename>"
	parser = OptionParser(usage=usage, version="%1")
	
	if len(sys.argv)<6: 
		print ("<Results file> <starfile> <output rootname> <CC threshold> <output filename>")
		sys.exit(-1)
	
	(options, args)=parser.parse_args()
	results_file=str(args[0])
	starfile=str(args[1])
	output_root=str(args[2])
	CC_THRES=float(args[3])
	output_filename=str(args[4])
	return (results_file, starfile,output_root,CC_THRES,output_filename)
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
def distance(X,Y,CX,CY):
	dis_X=X-CX
	dis_Y=Y-CY
	return(math.sqrt(dis_X*dis_X+dis_Y*dis_Y))			
if __name__== "__main__":
	main()

