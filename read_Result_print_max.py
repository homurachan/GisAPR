import argparse
def create_pick_parser():
	parser = argparse.ArgumentParser(description="Read search files and print best one.")
	parser.add_argument("--i", type=str, required=True, help="The output of new_..._v4.py")
	return parser
if __name__== "__main__":
	parser = create_pick_parser()
	args = parser.parse_args()
	input_filename=args.i
	a=open(input_filename,'r')
	data=a.readlines()
	MAX_score=-999999999
	REM_NAME = ""
	for i in range(0,len(data),2):
		score=float(data[i+1].split()[0])
		if(score > MAX_score):
			MAX_score = score
			REM_NAME = data[i].split()[0]
	
	print(f"Greatest score: {MAX_score}")
	print(f"Filename: {REM_NAME}")
	a.close()