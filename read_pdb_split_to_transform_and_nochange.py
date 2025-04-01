import numpy as np
import os,sys,math
from scipy.spatial.transform import Rotation as R
try:
	from optparse import OptionParser
except:
	from optik import OptionParser
from Bio import PDB

from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.vectors import Vector
import xpdb

def main():
	(pdb_filename, subunit_serial,output) = parse_command_line()
	output_chain_file = output+"_selected_chain_"+str(subunit_serial)+".pdb"
	output_rest_file = output+"_the_Main_Body.pdb"
	split_pdb_by_chain(pdb_filename, subunit_serial, output_chain_file, output_rest_file)

def split_pdb_by_chain(input_pdb, chain_id, output_chain_file, output_rest_file):
	# Initialize parser and structure
	parser = PDBParser(PERMISSIVE=True, structure_builder=xpdb.SloppyStructureBuilder())
	structure = parser.get_structure('test', input_pdb)

	# Initialize writers
	io = xpdb.SloppyPDBIO()

	# Extract and write the specified chain
	chain_structure = structure[0]  # assuming first model
	io.set_structure(chain_structure)

	# Select atoms for the specified chain
	class ChainSelect(PDB.Select):
		def __init__(self, chain_id, include=True):
			self.chain_id = chain_id
			self.include = include
		
		def accept_chain(self, chain):
			return chain.id == self.chain_id if self.include else chain.id != self.chain_id

	# Save the specified chain
	io.save(output_chain_file, ChainSelect(chain_id, include=True))

	# Save the rest of the chains
	io.save(output_rest_file, ChainSelect(chain_id, include=False))	
def parse_command_line():
	usage="%prog <pdb filename> <subunit serial> <output pdb rootname>"
	parser = OptionParser(usage=usage, version="%1")
	parser.disable_interspersed_args()
	if len(sys.argv)<4: 
		print ("<pdb filename> <subunit serial> <output pdb rootname>")
		sys.exit(-1)
	
	(options, args)=parser.parse_args()
	pdb_filename=str(args[0])
	subunit_serial=str(args[1])
	output=str(args[2])
	return (pdb_filename, subunit_serial,output)

if __name__== "__main__":
	main()
