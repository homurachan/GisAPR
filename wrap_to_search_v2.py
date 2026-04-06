import os, sys
import argparse
import concurrent.futures
import subprocess
# set default transRange to 0, set psiStep type to float.
def create_SEARCH_parser():
	parser = argparse.ArgumentParser(description="Read search params.")
	parser.add_argument("--script", type=str, required=True, help="The script file")
	parser.add_argument("--i", type=str, required=True, help="Input model file")
	parser.add_argument("--ang", type=str, required=True, help="Input angle star file")
	parser.add_argument("--mrc", type=str, required=True, help="Input 3D mrc file")
	parser.add_argument("--p", type=str, required=True, help="Input particle file")
	parser.add_argument("--FSC", type=str, default = None, help="Input FSC file")
	parser.add_argument("--o", type=str, required=True, help="Output file")
	parser.add_argument("--kk", type=float, default=0., help="The kk value, default = 0")
	parser.add_argument("--gpuid", type=int, default=0, help="The specified GPU ID, default = 0")
	parser.add_argument("--oriboxsize", type=int, default=256, help="The original boxsize, default = 256 (pixel)")
	parser.add_argument("--newboxsize", type=int, default=256, help="The new boxsize, default = 256 (pixel)")
	parser.add_argument("--apix", type=float, default=1.42, help="The ORIGINAL pixel size, default = 1.42")
	parser.add_argument("--transRange", type=int, default=0, help="Translation search range in pixel, default = 30. When set to 0, no translation would be searched.")
	parser.add_argument("--voltage", type=float, default=300, help="The voltage in kV, default = 300")
	parser.add_argument("--cs", type=float, default=2.7, help="The cc in mm, default = 2.7")
	parser.add_argument("--maskRadius", type=int, default=110, help="The softmask radius in pixel, corresponding to original boxsize, default = 110")
	parser.add_argument("--maskEdge", type=int, default=6, help="The softmask edge width in pixel, default = 6")
	parser.add_argument("--ignoreFSC", action='store_true', help="For testing purpose, ignoring the FSC weight. default = False")
	parser.add_argument("--discardMask", action='store_true', help="For testing purpose, apply NO soft mask. default = False")
	parser.add_argument("--psiStep", type=float, default=15, help="The psi angle search step in deg, default = 15")
	parser.add_argument("--doLocalSearch", action='store_true', help="Only search for the local orientations. Should combine with --localRange. default = False")
	parser.add_argument("--localRange", type=float, default=20., help="The local search range in +- this degree. Also applies to psi search.")
	parser.add_argument("--SplitParticles", type=int, default=1, help="Split the starfile into these sections. Default = 1")
	parser.add_argument("--doSplitDiffGpu", action='store_true', help="If enabled, wrap_to_search will use different gpuid. The inital gpuid is provided by --gpuid. default = False")
	return parser
	

def test_length(starfile_name):
	a=open(starfile_name,'r')
	data=a.readlines()
	length=len(data)
	a.close()
	return length
def generate_command(args,start,end,serial_number):
	sh2_command_line ="python "+args.script+" --i "+args.i+" --p "+args.p+" --FSC "+ args.FSC+" --oriboxsize "+str(args.oriboxsize)+" --newboxsize "+str(args.newboxsize)+\
	" --apix "+str(args.apix)+" --transRange "+str(args.transRange)+" --voltage "+str(args.voltage)+" --cs "+str(args.cs) \
	+" --maskRadius "+str(args.maskRadius)+" --maskEdge "+str(args.maskEdge)+" --psiStep "+str(args.psiStep)+" --kk "+str(args.kk)\
	+" --mrc "+str(args.mrc)+" --ang "+str(args.ang)
	if(args.ignoreFSC):
		sh2_command_line+=" --ignoreFSC "
	if(args.discardMask):
		sh2_command_line+=" --discardMask "
	if(args.doSplitDiffGpu):
		sh2_command_line+="--gpuid "+str(args.gpuid+serial_number)
	else:
		sh2_command_line+="--gpuid "+str(args.gpuid)
	if(args.doLocalSearch):
		sh2_command_line+=" --doLocalSearch --localRange "+str(args.localRange)
	sh2_command_line+=" --start "+str(start)+" --end "+str(end)
	new_filename = args.o+"_tmp"+str(serial_number)
	sh2_command_line+=" --o "+new_filename
#	print(sh2_command_line)
	return sh2_command_line,new_filename
	
def run_command(command):
	return subprocess.call(command, shell=True)	
	
if __name__ == "__main__":
	parser = create_SEARCH_parser()
	args = parser.parse_args()
	length = test_length(args.p)
	Split_num = length // args.SplitParticles + 1
	commands = []
	new_file = []
	for i in range(0,args.SplitParticles):
		start = i*Split_num
		end = (i+1)*Split_num
		command,new_filename = generate_command(args,start,end,i)
		commands.append(command)
		new_file.append(new_filename)
#	print(commands)
	with concurrent.futures.ProcessPoolExecutor(max_workers=args.SplitParticles) as executor:
		results = list(executor.map(run_command, commands))
	
	with open(args.o, 'w') as outfile:
		for temp_filename in new_file:
			with open(temp_filename, 'r') as infile:
				outfile.write(infile.read())
			os.remove(temp_filename)