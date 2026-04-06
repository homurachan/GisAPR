import numpy as np
import os,sys,math
from scipy.spatial.transform import Rotation as R
try:
	from optparse import OptionParser
except:
	from optik import OptionParser
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.vectors import Vector
import xpdb
# changelog: Change to rotation vector.
def main():
	(pdb_filename, subunit_serial,rot,tilt,psi,xshift,yshift,zshift,output) = parse_command_line()
	parser = PDBParser(PERMISSIVE=True, structure_builder=xpdb.SloppyStructureBuilder())
	# xpdb must be used to read large (>100,000 atoms in a subunit) PDBs
	structure = parser.get_structure('test', pdb_filename)

	chain_A = None
	for model in structure:
		for chain in model:
			if chain.id == subunit_serial:
				chain_A = chain
				break
		if chain_A is not None:
			break

	coordinates = []
	atoms = []
	if chain_A is not None:
		for residue in chain_A:
			for atom in residue:
				coordinates.append(atom.get_coord())
				atoms.append(atom)

	coordinates = np.array(coordinates)
	shift_coordinates = np.array([xshift,yshift,zshift])
	geometric_center = np.mean(coordinates, axis=0)
#	print("geometric_center of selected chain=",geometric_center)
	translated_coords = coordinates - geometric_center
#	rotation_matrix=np.asarray(Euler_angles2matrix(rot,tilt, psi))
	######## Switch to rotvec
	omega_deg = np.array([rot,tilt,psi])
	omega_rad = np.deg2rad(omega_deg)
	rotation_vector = R.from_rotvec(omega_rad)
	rotation_matrix = rotation_vector.as_matrix()
	
	
	## convert matrix into quaternion below
	r = R.from_matrix(rotation_matrix)
	quaternion = r.as_quat()
#	print("Quaternion:", quaternion)
	angle = r.magnitude()
	axis = r.as_rotvec() / angle
#	print("Angle (deg):", np.degrees(angle))
#	print("Axis (direction):", axis)
	## end
	## Using quaternion is more convinient to do differential
	
	rotated_coords = np.dot(translated_coords, rotation_matrix)
	new_coords = rotated_coords + geometric_center + shift_coordinates
	Set=0
	if chain_A is not None:
		for residue in chain_A:
			for atom in residue:
				new_coord_vector=Vector(new_coords[Set][0],new_coords[Set][1],new_coords[Set][2])
				atom.set_coord(new_coord_vector)
				Set+=1
	for model in structure:
		for chain in model:
			if chain.id == subunit_serial:
				chain = chain_A
	io = xpdb.SloppyPDBIO()
	io.set_structure(structure)
	io.save(output)

	
def parse_command_line():
	usage="%prog <pdb filename> <subunit serial> <rot> <tilt> <psi> <xshift in Angstrom> <yshift> <zshift> <output pdb name>"
	parser = OptionParser(usage=usage, version="%1")
	parser.disable_interspersed_args()
	if len(sys.argv)<10: 
		print ("<pdb filename> <subunit serial> <rot> <tilt> <psi> <xshiftin Angstrom> <yshift> <zshift> <output pdb name>")
		sys.exit(-1)
	
	(options, args)=parser.parse_args()
	pdb_filename=str(args[0])
	subunit_serial=str(args[1])
	rot=float(args[2])
	tilt=float(args[3])
	psi=float(args[4])
	xshift=float(args[5])
	yshift=float(args[6])
	zshift=float(args[7])
	output=str(args[8])
	return (pdb_filename, subunit_serial,rot,tilt,psi,xshift,yshift,zshift,output)
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
if __name__== "__main__":
	main()
