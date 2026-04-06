import math,os,sys,argparse
try:
	from optparse import OptionParser
except:
	from optik import OptionParser
# changelog
# v3: v3 only read a single filename, not a file index.
# v31: replace pdb2mrc to pdb2mrc_path. Use a standalone pdb2mrc binary.
# v32: if running SIRM-relion 1.31, no need to run relion_image_handler anymore. Add yflip feature to pdb2mrc. Should be used together with the pdb2mrc_remove_verbose.exe
# v33: the name of the search script will be fixed. currently the local value is changing, leading to some error.
# v34: apply the pdb2mrc_gpu_ver_fp32_v3.py and project3d_and_whiten.py in our programs.
# v35: add apix_PDB (20250415)
# v36: add doEnableGpuProj to enable GPU projection. In this version --gpuid is a single number, not like "0:0:1:1" anymore.
# v37: now use wrap_to_search.py to run the actual search. add --doSplitDiffGpu

# v38: now use wrap_to_search_v2.py to run the actual search. This version does the projection in device does not generate large intermediate files.
# v38 should be used together with wrap_to_search_v2.py, project3d_and_whiten_cuda_nowrite.py, and test1_with_isspa_weight_varingKK_search_translation_also_v605.py.
# v38 don't use the search script v60422 and before.
pdb2mrc_path = "python pdb2mrc_gpu_ver_fp32_v3.py"
project3d_and_whiten_path = "python project3d_and_whiten_cuda.py"
wrap_to_search_path = "python wrap_to_search_v2.py"
#SplitParticles = 2
def create_pdb2projection_parser_and_read():
	parser = argparse.ArgumentParser(description="GRID_SEARCH_CC_PR_WITH_TRANSLATION.")
	parser.add_argument("--searchScript", type=str, required=True, help="Input search script file")
	parser.add_argument("--i", type=str, required=True, help="Input pdb index file")
	parser.add_argument("--ang", type=str, required=True, help="Input angle starfile")
	parser.add_argument("--p", type=str, required=True, help="Input particle file")
	parser.add_argument("--o", type=str, required=True, help="Output Root name")
	parser.add_argument("--fsc", type=str, required=True, help="Input fsc file")
	parser.add_argument("--kk", type=float, default=3., help="The kk value, default = 0")
	parser.add_argument("--gpuid", type=str, default="0", help="The specified GPU ID, default = 0. To use multiple gpus, type 0:1:2:3... or 0:0:0")
	parser.add_argument("--oriboxsize", type=int, default=256, help="The original boxsize, default = 256 (pixel)")
	parser.add_argument("--newboxsize", type=int, default=160, help="The new boxsize for search, default = 160 (pixel)")
	parser.add_argument("--apix", type=float, default=1.42, help="The ORIGINAL pixel size of the particles, default = 1.42")
	parser.add_argument("--apix_PDB", type=float, default=1.42, help="The pixel size of the PDB, default = 1.42")
	parser.add_argument("--transRange", type=int, default=0, help="Translation search range in pixel, default = 30. When set to 0, no translation would be searched.")
	parser.add_argument("--voltage", type=float, default=300, help="The voltage in kV, default = 300")
	parser.add_argument("--cs", type=float, default=2.7, help="The cc in mm, default = 2.7")
	parser.add_argument("--maskRadius", type=int, default=110, help="The softmask radius in pixel, corresponding to original boxsize, default = 110")
	parser.add_argument("--maskEdge", type=int, default=6, help="The softmask edge width in pixel, default = 6")
	parser.add_argument("--ignoreFSC", action='store_true', help="For testing purpose, ignoring the FSC weight. default = False")
	parser.add_argument("--discardMask", action='store_true', help="For testing purpose, apply NO soft mask. default = False")
	parser.add_argument("--psiStep", type=int, default=3, help="The psi angle search step in deg, default = 15")
	parser.add_argument("--doLocalSearch", action='store_true', help="Only search for the local orientations. Should combine with --localRange. default = False")
	parser.add_argument("--localRange", type=float, default=30., help="The local search range in +- this degree. Also applies to psi search.")
	parser.add_argument("--yflip", action='store_true', help="If the reconstruction has the inversed handedness. default = False")
	parser.add_argument("--doEnableGpuProj", action='store_true', help="Enable GPU projection. Will consume large amount of device memory. default = False")
	parser.add_argument("--doSplitDiffGpu", action='store_true', help="If enabled, wrap_to_search will use different gpuid. The inital gpuid is provided by --gpuid. default = False")
	parser.add_argument("--SplitParticles", type=int, default=1, help="Split the starfile into these sections. Default = 1")
	return parser
def main():
	## reading parameters
	parser = create_pdb2projection_parser_and_read()
	args = parser.parse_args()
	create_pdb2projection(args)

def convert_number_to_filename(num):
	if num < 0:
		# Convert negative number
		formatted_num = f"N{abs(num):.1f}".replace('.', 'p')
	else:
		# Convert positive number
		formatted_num = f"{num:.1f}".replace('.', 'p')

	# Remove 'p0' if it exists at the end
	if formatted_num.endswith('p0'):
		return formatted_num[:-2]
	return formatted_num
def convert_number_to_filename_2(num):
	if num < 0:
		# Convert negative number
		formatted_num = f"N{abs(num):.2f}".replace('.', 'p')
	else:
		# Convert positive number
		formatted_num = f"{num:.2f}".replace('.', 'p')

	# Remove 'p0' if it exists at the end
	if formatted_num.endswith('p0'):
		return formatted_num[:-2]
	return formatted_num
def read_gpuid(toread_gpuid):
	table=toread_gpuid.split(':')
	return len(table),table
		
def create_pdb2projection(args):
	NN=1
	# NN = run the same search how many times.
	PDB_filename=args.i
	ang = args.ang
	boxsize=args.oriboxsize
	apix=args.apix
	apix_PDB=args.apix_PDB
	APIX=convert_number_to_filename_2(apix_PDB)
	res=2.0*apix_PDB
	search_script = args.searchScript
	ver="v603"
	particle_file=args.p
	newboxsize=args.newboxsize
	do_local_search=args.doLocalSearch
	local_stepsize=str(int(args.localRange))
	transRange=str(args.transRange)
	voltage=str(args.voltage)
	cs=str(args.cs)
	psiStep=str(int(args.psiStep))
	kk=str(int(args.kk))
	fsc_file=args.fsc
	maskRadius=args.maskRadius
	maskEdge=args.maskEdge
	output_root=args.o
	doEnableGpuProj = args.doEnableGpuProj
	doSplitDiffGpu = args.doSplitDiffGpu
	SplitParticles = args.SplitParticles
	toread_gpuid = args.gpuid
	ngpus,gpu_table=read_gpuid(toread_gpuid)

	sh_name1_suffix="_generate_models_from_pdb.sh"
	sh_name2_suffix="_search_script.sh"
	
	b=open(output_root+sh_name1_suffix,'w')
	c=open(output_root+sh_name2_suffix,'w')
	sh1_command_line=""
	sh2_command_line=""
	pdb2mrc_command_line=""
	project_command_line=""
	whitening_command_line=""
	COUNT_FOR_GPU=0

	PDB_filename_prefix=PDB_filename.split(".pdb")[0]
	mrc_filename=PDB_filename_prefix+"_box"+str(boxsize)+"_apix"+APIX+".mrc"
	pdb2mrc_command_line+=(pdb2mrc_path+" --i "+PDB_filename+" --o "+mrc_filename+" --box "+str(boxsize)+" --apix "+str(apix_PDB)+" --res "+str(res))
	if(args.yflip):
		pdb2mrc_command_line+=(" --yflip\n")
	else:
		pdb2mrc_command_line+=("\n")
	star_filename_prefix="test_"+PDB_filename_prefix+"_noctf_nonoise_whitened"
	project_command_line+=(project3d_and_whiten_path+" --i "+mrc_filename+" --o "+star_filename_prefix+" --ang "+ang+" --normal_background_powerspectrum")
	if(doEnableGpuProj):
		project_command_line+=(" --gpuid "+str(gpu_table[COUNT_FOR_GPU])+ "\n")
	else:
		project_command_line+=("\n")
#	whitening_command_line+=("relion_image_handler --i "+star_filename_prefix+".star --o whitened --bg_radius 0.0 --boxsize "+str(boxsize)+" --normal_background_powerspectrum\n")
#	sh1_command_line+=("rm *noctf_nonoise.mrcs *noctf_nonoise.star\n")
#	b.write(sh1_command_line)
	
	for j in range(NN):
		sh2_command_line+=wrap_to_search_path+" --script "+search_script+" --i "+star_filename_prefix+".star --p "+particle_file+" --FSC "+ fsc_file+" --oriboxsize "+str(boxsize)+" --newboxsize "+str(newboxsize)+" --apix "+str(apix)+" --transRange "+transRange+" --voltage "+voltage+" --cs "+cs \
		+" --maskRadius "+str(maskRadius)+" --maskEdge "+str(maskEdge)+" --psiStep "+psiStep+" --kk "+kk+" --mrc "+str(mrc_filename)+" --ang "+str(ang)
		if(args.ignoreFSC):
			sh2_command_line+=" --ignoreFSC "
		sh2_command_line+="--gpuid "
		sh2_command_line+=str(gpu_table[COUNT_FOR_GPU])
		COUNT_FOR_GPU+=1
		if(COUNT_FOR_GPU>=ngpus):
			COUNT_FOR_GPU=0
		if(do_local_search):
			sh2_command_line+=" --doLocalSearch --localRange "+local_stepsize +" --SplitParticles "+str(SplitParticles)
		if(doSplitDiffGpu):
			sh2_command_line+=" --doSplitDiffGpu"
		sh2_command_line+=" --o search_"+ver+"_"+PDB_filename_prefix+"_orient3degree_trans"+transRange+"pixels_psi"+psiStep+"_kk"+kk+"_Correct_run"+str(j+1)+".txt\n"

	b.write(pdb2mrc_command_line)
#	b.write(project_command_line)
#	b.write(whitening_command_line)
	
	c.write(sh2_command_line)
	b.close()
	c.close()

if __name__== "__main__":
	main()
