import numpy as np
import os,sys,argparse,math
from scipy.spatial.transform import Rotation as R
# changelog 20250329
# update the read_pdb_index_generate_sh_3DEG_local_v33.py to read_pdb_index_generate_sh_3DEG_local_v34.py
# update func_check_boundary_for_testing_v6.py to func_check_boundary_for_testing_v7.py
# update sh usage so it can be run under windows.
# changelog 20250330
# update func_check_boundary_for_testing_v7.py to v8.
def func(x, args):
	PDB_NAME = args.PDB_NAME
	STAR_NAME = args.STAR_NAME
	rotate_chain = args.rotate_chain
	output_name_root = args.output_name_root
	gpuid = args.gpuid
	ang = args.ang
	boxsize = args.boxsize
	apix = args.apix
	newboxsize = args.newboxsize
	search_script = args.search_script
	fsc_file = args.fsc_file
	transRange = args.transRange
	voltage = args.voltage
	cs = args.cs
	psiStep = args.psiStep
	kk = args.kk
	do_local_search = args.do_local_search
	local_stepsize = args.local_stepsize
	do_ignoreFSC = args.do_ignoreFSC
	maskRadius = args.maskRadius
	Geometric_restrain_Scaling_Factor=args.Geometric_restrain_Scaling_Factor
	chain_MASS_in_residues = args.chain_MASS_in_residues
	MAXIUM_ALLOWED_overlapped_pixels = args.MAXIUM_ALLOWED_overlapped_pixels
	MAX_MinDistance_Allowed = args.MAX_MinDistance_Allowed
	yflip=args.yflip
	rot=x[0]
	tilt=x[1]
	psi=x[2]
	xshift=x[3]
	yshift=x[4]
	zshift=x[5]
	score_in_this_conformation = 9999.
	def convert_number_to_filename(number):
		integer_part = int(np.fabs(number))
		sign = (number <0)
		decimal_part = abs(number - integer_part)
		dp=f"{decimal_part:.1f}".split('.')[1]
		tmp=""
		if(sign):
			tmp="N"
		output=tmp+str(integer_part)+"p"+str(dp)
		return output
	str_Rot=convert_number_to_filename(rot)
	str_Tilt=convert_number_to_filename(tilt)
	str_Psi=convert_number_to_filename(psi)
	str_XSHIFT=convert_number_to_filename(xshift)
	str_YSHIFT=convert_number_to_filename(yshift)
	str_ZSHIFT=convert_number_to_filename(zshift)

	# Step: 
	# 0. Compute geometric restrain. If overlapped pixels are too many, skip all the rest computation.
	Step0_Python_Name="python func_check_boundary_for_testing_v8.py "
	Geometric_restrain_Result_filename = output_name_root+rotate_chain+"_rot"+str_Rot+"_tilt"+str_Tilt+"_psi"+str_Psi+"deg_trans"+str_XSHIFT+"_"+str_YSHIFT+"_"+str_ZSHIFT+"_GeometricRestrain_Result.txt"
	To_Run_Command_Step0 = Step0_Python_Name+"--i "+PDB_NAME+" --chainID "+rotate_chain+" --rot "+str(rot)+" --tilt "+str(tilt)+" --psi "+str(psi)\
	+" --centerX "+str(xshift)+" --centerY "+str(yshift)+" --centerZ "+str(zshift)+" --outputRoot "+output_name_root+" --outputFile "+Geometric_restrain_Result_filename+"\n"
	os.system(To_Run_Command_Step0)
	Geometric_restrain_Result_FILE = open(Geometric_restrain_Result_filename,"r")
	Geometric_restrain_Result_FILE_lines=Geometric_restrain_Result_FILE.readlines()
	Geometric_restrain_Overlapped_Pixels=float(Geometric_restrain_Result_FILE_lines[0].split()[7])
	Min_Distance = float(Geometric_restrain_Result_FILE_lines[0].split()[8])
	print("debug, Overlapped pixels, Min_Distance = ",Geometric_restrain_Overlapped_Pixels,Min_Distance)
	def Compute_OP_related_bias(Geometric_restrain_Overlapped_Pixels,chain_MASS_in_residues):
		TEST=Geometric_restrain_Overlapped_Pixels+3
		if(TEST<1.):
			TEST=1.0
		Upper = np.log(TEST)
		Lower = 0.0001*(0.225*chain_MASS_in_residues+87.0)
		# This Lower is obtained by linear fitting to (the linear fitting results of Likelikhood vs ln(overlapped_pixels)) vs chain_mass_in_residues.
		return(Upper/Lower)
	Geometric_restrain_Bias = 0.0
	Geometric_restrain_Bias = Geometric_restrain_Scaling_Factor*Compute_OP_related_bias(Geometric_restrain_Overlapped_Pixels,chain_MASS_in_residues)
	print("debug, Geometric_restrain_Bias = ",Geometric_restrain_Bias)
	if(Geometric_restrain_Overlapped_Pixels>=MAXIUM_ALLOWED_overlapped_pixels or Min_Distance>=MAX_MinDistance_Allowed):
		print("Search on this conformation is skipped. A preset likelihood value will be set.")
	#	return Preset_likelihood_value
		return Geometric_restrain_Overlapped_Pixels + MAX_MinDistance_Allowed
	# Geometric_restrain_Bias is a positive number, add to the -1*Likelikhood
	# 1. Rotate PDB by the chain. The command is `python rotate_subunit_of_PDB_v2.py 7k00_refined_fit_EMPIAR11241_J028.pdb A $rot $tilt 7k00_refined_chain_A_rotN5_tiltN5deg.pdb`
	Step1_Python_Name="python rotate_translate_subunit_of_PDB_v2.py "
	Rotated_PDB_Name=output_name_root+rotate_chain+"_rot"+str_Rot+"_tilt"+str_Tilt+"_psi"+str_Psi+"deg_trans"+str_XSHIFT+"_"+str_YSHIFT+"_"+str_ZSHIFT+"ANG.pdb"

	To_Run_Command_Step1=Step1_Python_Name+PDB_NAME+" "+rotate_chain+" "+str(rot)+" "+str(tilt)+" "+str(psi)+" "+str(xshift)+" "+str(yshift)+" "+str(zshift)+" "+Rotated_PDB_Name
	os.system(To_Run_Command_Step1)
	# no need to run in parallel here.
	
	# 2. PDB2MRC, Generate angle tables and Project the mrc with whitening.
	# require angle step size. For testing purpose, simple read_pdb_index_generate_sh_3DEG.py is used.
	
	Step2_Python_Name="python read_pdb_index_generate_sh_3DEG_local_v34.py"
	Step2_Root_Name="RUN01_"+str(Rotated_PDB_Name)
	To_Run_Command_Step2=Step2_Python_Name+" --i "+Rotated_PDB_Name+" --o "+Step2_Root_Name+" --searchScript "+search_script\
	+" --ang "+ang+" --p "+STAR_NAME+" --apix "+str(apix)+" --fsc "+fsc_file+" --kk "+str(kk)+" --oriboxsize "+str(boxsize)+" --newboxsize "+str(newboxsize)\
	+" --transRange "+str(transRange)+" --voltage "+str(voltage)+" --cs "+str(cs)+" --psiStep "+str(psiStep)+" --gpuid "+gpuid
	if(yflip):
		To_Run_Command_Step2+=" --yflip"
	####
	# gpuid here is problematic.
	####
	if(do_local_search):
		To_Run_Command_Step2+=" --doLocalSearch "+" --localRange "+str(local_stepsize)
	if(do_ignoreFSC):
		To_Run_Command_Step2+=" --ignoreFSC "
	To_Run_Command_Step2+=" --maskRadius "+str(maskRadius)

	print(To_Run_Command_Step2)
	os.system(To_Run_Command_Step2)
	# no need to run in parallel here.
			
	# 2.1. The generated sh file 
	Search_SUFFIX="_search_script.sh"
	Gen_PDB_SH = Step2_Root_Name+"_generate_models_from_pdb.sh"
	Search_SH = Step2_Root_Name+Search_SUFFIX
	if(os.name == 'nt'):
		TMP_FILE=open(Gen_PDB_SH,'r')
		TMP_LINE=TMP_FILE.readlines()
		RUN_Command_PDB2MRC = TMP_LINE[0]
		os.system(RUN_Command_PDB2MRC)
		RUN_Command_PDB2MRC = TMP_LINE[1]
		os.system(RUN_Command_PDB2MRC)
		TMP_FILE.close()
		# Current version, the _generate_models_from_pdb.sh file has 2 lines, _search_script.sh only has 1 line.
	else:
		RUN_Command_PDB2MRC = "sh "+Gen_PDB_SH
		print(RUN_Command_PDB2MRC)
		os.system(RUN_Command_PDB2MRC)
	Runid=Rotated_PDB_Name
	# 3. The search command.
	# Read the Output Filename first.
	Step3p1_Index_Name="index_for_result_"+str(Runid)+".txt"
	File_Result_index=open(Step3p1_Index_Name,'w')
	FF=open(Search_SH,"r")
	FF_LINE=FF.readlines()
	for i in range(len(FF_LINE)):
		FF_LINE_i_OutputName=FF_LINE[i].split()[-1]
		# remove the "\n".
		Search_Output_Name=str(FF_LINE_i_OutputName.split()[0])
		File_Result_index.write(Search_Output_Name+"\n")
	FF.close()
	File_Result_index.close()
	if(os.name == 'nt'):
		TMP_FILE=open(Search_SH,'r')
		TMP_LINE=TMP_FILE.readlines()
		RUN_Command_Search = TMP_LINE[0]
		TMP_FILE.close()
	else:
		RUN_Command_Search = "sh "+Search_SH
	print(RUN_Command_Search)
	os.system(RUN_Command_Search)

	# 4. Search for the Signal/Noise Peak.

	Step4_Python_Name = "python new_method_to_fit_the_2nd_Gaussian_PEAK_v31_nowrite.py"
	Result_File="ReSuLt_"+str(Runid)+".txt"
	To_Run_Command_Step4=Step4_Python_Name+" "+Step3p1_Index_Name+" "+STAR_NAME+" nouse 8.5"+" "+Result_File
	print(To_Run_Command_Step4)
	os.system(To_Run_Command_Step4)
	Aa=open(Result_File,'r')
	Result_Line=Aa.readlines()
	Value=float(Result_Line[1].split()[0])
#	score_in_this_conformation = -1.0*Value+Geometric_restrain_Bias
	score_in_this_conformation = -1.0*Value
	return score_in_this_conformation