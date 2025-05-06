import numpy as np
import os,sys,math
from scipy.optimize import curve_fit, fsolve
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
# changelog ver30
# In this version, I adapt the MLE into the scores.
# Basic equation is: ln(likelihood) = 1/SQR(sigma)*SUM(CC)
# But how can we get the sigma? I think sigma is the width of noise peak, obtained by fitting the noise peak.
# SUM(CC) can be calculated by splitting the signal peak into two parts.
# The first part contains the values those are larger than the peak position. Just some simple sum.
# However, the left part is way more complicated. It turns out to be number_of_particles_to_the_left * peak_positions.
# We regard number_of_particles_to_the_left being equal the number of particles on the right side of the signal peak.

# changelog ver31
# Move the total_estimated_sum_cc to the first column in order to be read by the following scrits.
# Fix when the first position of Gaussian peak is larger than the second.
# TODO: When only one peak detected, refined_x should be the peak of all.

# changelog ver4
# completely rewrite the codes. Now it doesn't need to read the starfile.
# add switching between CC and Zscore, integration and simple sum.
# fix a terrible bug, where total_estimated_sum_cc is completely wrong.
# fix when curve_fit failed.
try:
	from optparse import OptionParser
except:
	from optik import OptionParser
def gaussian(x, amp, mean, stddev):
	return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))
def double_gaussian(x, amp1, mean1, stddev1, amp2, mean2, stddev2):
	return gaussian(x, amp1, mean1, stddev1) + gaussian(x, amp2, mean2, stddev2)
	
def main():
	(results_file,do_run_CC,do_simple_sum,output_filename) = parse_command_line()
	bb=open(output_filename,'w')
	a=open(results_file,'r')
	aa=a.readlines()
	print("debug,",len(results_file))
	for i in range(0,len(aa)):
		filename=aa[i].split()[0]
		if(i%1==0):
			bb.write(filename+"\n")
		thres,CC_value,ZSCORE_value,total_estimated_sum_cc=run_this(filename,do_run_CC,do_simple_sum)
		bb.write(str(total_estimated_sum_cc)+" "+str(ZSCORE_value)+" "+str(CC_value)+"\n")
		
	bb.close()
	a.close()
def run_this(results_file,do_run_CC,do_simple_sum):
	# if do_run_CC = False, fit with the Zscore.
	print(results_file)
	a=open(results_file,'r')
	aa=a.readlines()
	CC_ARRAY=[]
	raw_CC_array=[]
	# Note: The raw CC value was multiplied by 1024. So we should divide by it.
	for i in range(0,len(aa)):
		CC_or_PR=str(aa[i].split(',')[0])
		if(CC_or_PR=="['cc: '"):
			Serial_Number = int(aa[i].split(',')[1])
			CC_=float(aa[i].split(',')[2])
		# NOTE: CC_ is the Z-score
		#	CC_L3=int(aa[i].split(',')[3])
		#	CC_ACCU=int(aa[i].split(',')[4])
			new_rot=float(aa[i].split(',')[5])
			new_tilt=float(aa[i].split(',')[6])
			DIS_CC=float(aa[i].split(',')[7])
			X=float(aa[i].split(',')[8])
			Y=float(aa[i].split(',')[9])
			CX=float(aa[i].split(',')[10])
			CY=float(aa[i].split(',')[-1].split(']')[0])
		#	DisTance=distance(X,Y,CX,CY)
		#	DIS_TOTAL+=DisTance
		#	DIS_NUM+=1.0
			if(CC_<9999):
				CC_ARRAY.append(CC_)

		if(CC_or_PR=="['psi: '"):
			Serial_Number = int(aa[i].split(',')[1])
			PSI=float(aa[i].split(',')[2])
			PSI_ORI=float(aa[i].split(',')[-1].split(']')[0])
			PSI_ORI=(PSI_ORI+360.)%360.
			PSI_DISTANCE=(math.fabs(PSI-PSI_ORI)+360.)%360.

		if(CC_or_PR=="['rawCC: '"):
			Serial_Number = int(aa[i].split(',')[1])
			rawCC=float(aa[i].split(',')[-1].split(']')[0])/1024.0
			raw_CC_array.append(rawCC)

	a.close()

	data=np.asarray(CC_ARRAY)
	# data is the z-score array.
	CC_data=np.asarray(raw_CC_array)

	if (len(CC_data)<1):
		return 0.0,0.0,0.0,0.0
		# NO data in the file, return all zeros.
	if (do_run_CC > 0 and len(CC_data>0)):
		hist_CC, bin_edges_CC = np.histogram(CC_data, bins=1000,range=(0,1.0), density=True)
		smoothed_hist_CC = gaussian_filter(hist_CC, sigma=2)
		bin_centers_CC = (bin_edges_CC[:-1] + bin_edges_CC[1:]) / 2
		GUESS_PEAK_CC=bin_centers_CC[np.unravel_index(np.argmax(smoothed_hist_CC),smoothed_hist_CC.shape)]
		initial_guess_CC = [max(smoothed_hist_CC), GUESS_PEAK_CC, 0.02, max(smoothed_hist_CC)/2.0,GUESS_PEAK_CC+0.05, 0.1]
		try:
			params_CC, covariance_CC = curve_fit(double_gaussian, bin_centers_CC, smoothed_hist_CC, p0=initial_guess_CC)
		except:
			total_estimated_sum_cc = np.sum(CC_data)
			print(f"Fitting the double Gaussian failed. Return simple SUM = {total_estimated_sum_cc}")
			return 0.0,0.0,0.0,total_estimated_sum_cc
		print("params_CC: ",params_CC)
		# First approach, refine_x = second fitted peak.
		refined_x = params_CC[4]
		if(params_CC[1]>params_CC[4]):
			refined_x = params_CC[1]
	#	do_simple_sum = False
		try:
			peaks, properties = find_peaks(smoothed_hist_CC, prominence=0.01, distance=20,width=5)
			peak_positions = bin_centers_CC[peaks]
			peak_heights = smoothed_hist_CC[peaks]

			print("Detected peak positions:", peak_positions)
			print("Peak heights:", peak_heights)
			
		except:
			do_nothing = True
		number_larger_than_peak = np.sum(CC_data > refined_x)
		
		left_estimated_cc= number_larger_than_peak*(refined_x)
		right_estimated_cc = np.sum(CC_data[CC_data > refined_x])
		total_estimated_sum_cc = left_estimated_cc+right_estimated_cc
	#	print(number_larger_than_peak,left_estimated_cc,right_estimated_cc,total_estimated_sum_cc)
		if(do_simple_sum>0):
			total_estimated_sum_cc = np.sum(CC_data)
		print("total_estimated_sum_cc = ",total_estimated_sum_cc)
		return 0.0,refined_x,0.0,total_estimated_sum_cc
	if (do_run_CC < 1 and len(data>0)):
		hist_CC, bin_edges_CC = np.histogram(data, bins=1000, density=True)
		# Zscore don't have a concrete range
		smoothed_hist_CC = gaussian_filter(hist_CC, sigma=2)
		bin_centers_CC = (bin_edges_CC[:-1] + bin_edges_CC[1:]) / 2
		GUESS_PEAK_CC=bin_centers_CC[np.unravel_index(np.argmax(smoothed_hist_CC),smoothed_hist_CC.shape)]
		initial_guess_CC = [max(smoothed_hist_CC), GUESS_PEAK_CC, 0.3, max(smoothed_hist_CC)/4.0,GUESS_PEAK_CC+2, 4]
		try:
			params_CC, covariance_CC = curve_fit(double_gaussian, bin_centers_CC, smoothed_hist_CC, p0=initial_guess_CC)
		except:
			total_estimated_sum_cc = np.sum(data)
			print(f"Fitting the double Gaussian failed. Return simple SUM = {total_estimated_sum_cc}")
			return 0.0,0.0,0.0,total_estimated_sum_cc
		print("params_Zscore: ",params_CC)
		# First approach, refine_x = second fitted peak.
		refined_x = params_CC[4]
		if(params_CC[1]>params_CC[4]):
			refined_x = params_CC[1]
		try:
			peaks, properties = find_peaks(smoothed_hist_CC, prominence=0.01, distance=20,width=5)
			peak_positions = bin_centers_CC[peaks]
			peak_heights = smoothed_hist_CC[peaks]

			print("Detected peak positions:", peak_positions)
			print("Peak heights:", peak_heights)
			
		except:
			do_nothing = True
		number_larger_than_peak = np.sum(data > refined_x)
		left_estimated_cc= number_larger_than_peak*(refined_x)
		right_estimated_cc = np.sum(data[data > refined_x])
		total_estimated_sum_cc = left_estimated_cc+right_estimated_cc
		if(do_simple_sum>0):
			total_estimated_sum_cc = np.sum(data)
		print("total_estimated_sum_zs = ",total_estimated_sum_cc)
		return 0.0,refined_x,0.0,total_estimated_sum_cc
def parse_command_line():
	usage="%prog <Results file> <do_run_CC = 1, zscore = 0> <do_simple_sum = 1> <output filename>"
	parser = OptionParser(usage=usage, version="%1")
	
	if len(sys.argv)<5: 
		print ("<Results file> <do_run_CC = 1, zscore = 0> <do_simple_sum = 1> <output filename>")
		sys.exit(-1)
	
	(options, args)=parser.parse_args()
	results_file=str(args[0])
	do_run_CC = float(args[1])
	do_simple_sum = float(args[2])
	output_filename=str(args[3])
	return (results_file,do_run_CC,do_simple_sum,output_filename)
	
if __name__== "__main__":
	main()

