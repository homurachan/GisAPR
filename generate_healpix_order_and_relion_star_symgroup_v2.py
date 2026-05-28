#!/usr/bin/env python

import math
import random
import argparse
import numpy as np
from eqps_from_fortran import eqps_with_input_step_return_relion_rot_tilt


def normalize_vector(v):
	v = np.asarray(v, dtype=np.float64)
	n = float(np.linalg.norm(v))
	if n <= 1.0e-12:
		return None
	return v / n


def direction_from_rot_tilt(rot_deg, tilt_deg):
	rot = math.radians(float(rot_deg))
	tilt = math.radians(float(tilt_deg))
	return np.array([
		math.sin(tilt) * math.cos(rot),
		math.sin(tilt) * math.sin(rot),
		math.cos(tilt),
	], dtype=np.float64)


def rotation_matrix_z(angle_rad):
	c = math.cos(angle_rad)
	s = math.sin(angle_rad)
	return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def rotation_matrix_x(angle_rad):
	c = math.cos(angle_rad)
	s = math.sin(angle_rad)
	return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def unique_matrices(mats, tol=1.0e-10):
	seen = set()
	out = []
	for m in mats:
		key = tuple(np.round(np.asarray(m).reshape(-1) / tol).astype(np.int64))
		if key not in seen:
			seen.add(key)
			out.append(np.asarray(m, dtype=np.float64))
	return out


def proper_signed_permutation_matrices():
	"""All 24 proper rotations of the cube/octahedral group."""
	mats = []
	for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
		for sx in (-1.0, 1.0):
			for sy in (-1.0, 1.0):
				for sz in (-1.0, 1.0):
					m = np.zeros((3, 3), dtype=np.float64)
					m[0, perm[0]] = sx
					m[1, perm[1]] = sy
					m[2, perm[2]] = sz
					if np.linalg.det(m) > 0.5:
						mats.append(m)
	return mats


def tetrahedral_matrices():
	"""The 12 proper rotations that preserve one tetrahedron inside a cube."""
	verts = np.array([
		[1.0, 1.0, 1.0],
		[1.0, -1.0, -1.0],
		[-1.0, 1.0, -1.0],
		[-1.0, -1.0, 1.0],
	], dtype=np.float64)
	vset = {tuple(v) for v in verts}
	mats = []
	for m in proper_signed_permutation_matrices():
		mapped = {tuple(np.round(m @ v).astype(int)) for v in verts}
		if mapped == vset:
			mats.append(m)
	return mats


def rotation_matrix_axis_angle(axis, angle_rad):
	"""Right-handed active rotation matrix around an arbitrary axis."""
	a = normalize_vector(axis)
	if a is None:
		raise ValueError("Rotation axis has zero length")
	x, y, z = a
	c = math.cos(angle_rad)
	s = math.sin(angle_rad)
	C = 1.0 - c
	return np.array([
		[c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
		[y*x*C + z*s, c + y*y*C,     y*z*C - x*s],
		[z*x*C - y*s, z*y*C + x*s, c + z*z*C],
	], dtype=np.float64)


def matrix_key(m, tol=1.0e-10):
	return tuple(np.round(np.asarray(m, dtype=np.float64).reshape(-1) / tol).astype(np.int64))


def close_rotation_group(generators, tol=1.0e-10, max_order=1000):
	"""Generate the finite rotation group from a list of generator matrices."""
	identity = np.eye(3, dtype=np.float64)
	group = [identity]
	seen = {matrix_key(identity, tol)}
	queue = [identity]
	gens = [np.asarray(g, dtype=np.float64) for g in generators]

	while queue:
		a = queue.pop(0)
		for g in gens:
			for m in (g @ a, a @ g):
				m[np.abs(m) < tol] = 0.0
				k = matrix_key(m, tol)
				if k not in seen:
					seen.add(k)
					group.append(m)
					queue.append(m)
					if len(group) > max_order:
						raise RuntimeError("Generated symmetry group is too large; check generators")
	return group


def relion_icosahedral_matrices(sym):
	"""
	RELION-style icosahedral proper rotations.

	In RELION's symmetries.cpp, I and I2 use the same generator axes.
	I3 is a different embedding with a 5-fold axis along Z.
	Only proper rotations are generated here; I*H mirror variants are intentionally omitted.
	"""
	sym = sym.lower()
	if sym in ("i", "icos", "ico", "i2"):
		generator_specs = [
			(2, [0.0, 0.0, 1.0]),
			(5, [0.525731114, 0.0, 0.850650807]),
			(3, [0.0, 0.356822076, 0.934172364]),
		]
	elif sym == "i3":
		generator_specs = [
			(2, [-0.5257311143, 0.0, 0.8506508070]),
			(5, [0.0, 0.0, 1.0]),
			(3, [-0.4911234778630044, 0.3568220764705179, 0.7946544753759428]),
		]
	else:
		raise ValueError("Unsupported icosahedral symmetry: %s. Use I2 or I3." % sym)

	gens = [rotation_matrix_axis_angle(axis, 2.0 * math.pi / fold) for fold, axis in generator_specs]
	mats = close_rotation_group(gens, tol=1.0e-5, max_order=120)
	if len(mats) != 60:
		raise RuntimeError("Expected 60 matrices for %s, but generated %d" % (sym, len(mats)))
	return mats


def symmetry_matrices(sym):
	"""
	Return proper rotation matrices for point-group symmetry.

	C1 means no symmetry reduction.  Cn keeps roughly 1/n of projection
	directions; Dn keeps roughly 1/(2n).  For T/O/I the finite proper rotation
	groups have orders 12/24/60, respectively.  I/icos are treated as RELION I2.
	"""
	sym = (sym or "c1").strip().lower()
	if sym in ("", "c1"):
		return [np.eye(3, dtype=np.float64)]

	if sym[0] == "c":
		n = int(sym[1:])
		if n < 1:
			raise ValueError("Cyclic symmetry order must be >= 1: %s" % sym)
		return [rotation_matrix_z(2.0 * math.pi * k / n) for k in range(n)]

	if sym[0] == "d":
		n = int(sym[1:])
		if n < 1:
			raise ValueError("Dihedral symmetry order must be >= 1: %s" % sym)
		mats = []
		rx180 = rotation_matrix_x(math.pi)
		for k in range(n):
			rz = rotation_matrix_z(2.0 * math.pi * k / n)
			mats.append(rz)
			mats.append(rz @ rx180)
		return unique_matrices(mats)

	if sym == "tet":
		return tetrahedral_matrices()
	if sym == "oct":
		return proper_signed_permutation_matrices()
	if sym in ("icos", "ico", "i", "i2", "i3"):
		return relion_icosahedral_matrices(sym)

	raise ValueError("Unsupported symmetry: %s. Use cN, dN, tet, oct, I2, or I3." % sym)


def canonical_direction_key(v, sym_mats, key_tol=1.0e-10):
	"""A stable representative key for one symmetry orbit of a direction."""
	best = None
	for m in sym_mats:
		q = normalize_vector(m @ v)
		if q is None:
			continue
		# Lexicographic ordering gives one deterministic representative per orbit.
		# The exact boundary is arbitrary, but the retained fraction follows the
		# symmetry-group order for uniformly distributed directions.
		key = tuple(np.round(q / key_tol).astype(np.int64).tolist())
		if best is None or key > best:
			best = key
	return best


def keep_direction_by_symmetry(rot_deg, tilt_deg, sym_mats, key_tol=1.0e-10):
	if len(sym_mats) <= 1:
		return True
	v = normalize_vector(direction_from_rot_tilt(rot_deg, tilt_deg))
	if v is None:
		return False
	this_key = tuple(np.round(v / key_tol).astype(np.int64).tolist())
	return this_key == canonical_direction_key(v, sym_mats, key_tol=key_tol)


def normalized_relion_rot(rot):
	# Preserve the original script behavior: input [0, 360) is shifted to [-180, 180).
	return math.fmod(float(rot) + 360.0, 360.0) - 180.0


def normalized_relion_tilt(tilt):
	return math.fmod(float(tilt) + 360.0, 360.0)


def random_psi(do_randomPsi, divisible, num_psi, psi_stepsize):
	if not do_randomPsi:
		return 0.0
	if divisible:
		tmp = random.randint(0, num_psi - 1)
	else:
		tmp = random.randint(0, num_psi)
	return float(tmp * psi_stepsize)


def write_relion_header(r, apix):
	r.write("# relion 30001\n\ndata_optics\n\nloop_\n")
	r.write("_rlnOpticsGroup #1\n")
	r.write("_rlnOpticsGroupName #2\n")
	r.write("_rlnSphericalAberration #3\n")
	r.write("_rlnVoltage #4\n")
	r.write("_rlnImagePixelSize #5\n")
	r.write("\t1\topticsGroup1\t2.7\t300.0\t")
	r.write(str(apix) + "\n\n\n")
	r.write("# relion 30001\n\ndata_particles\n\nloop_\n")
	r.write("_rlnAngleRot #1\n")
	r.write("_rlnAngleTilt #2\n")
	r.write("_rlnAnglePsi #3\n")
	r.write("_rlnImageName #4\n")


def iter_eqps_rot_tilt(eqps_angle_degree):
	rot, tilt = eqps_with_input_step_return_relion_rot_tilt(eqps_angle_degree)
	for r, t in zip(rot, tilt):
		yield float(r), float(t)


def iter_healpix_rot_tilt(healpix_order):
	from healpix import healpix_py
	aa = healpix_py(healpix_order, "NEST")
	for i in range(aa.npix_):
		ff = aa.getDirectionFromHealPix(i)
		yield float(ff[0]), float(ff[1])


def main():
	parser = argparse.ArgumentParser(
		description=(
			"Generate orientation sampling points from HEALPix or EQPS, remove symmetry-equivalent "
			"projection directions if requested, and write a RELION starfile. Rot range is [-180,180), "
			"Tilt range is [0,180]."
		)
	)
	parser.add_argument("--o", type=str, required=True, help="Output filename")
	parser.add_argument("--randomPsi", action="store_true", help="Add random Psi to the starfile. default = False")
	parser.add_argument("--psiStep", type=int, default=8, help="The stepsize of randomPsi, default = 8")
	parser.add_argument("--discardPositiveRot", action="store_true", help="For testing purpose, drop points that Rot > zero. default = False")
	parser.add_argument("--healpixOrder", type=int, default=3, help="HEALPix order, default = 3: 7.5 degree. Order=2: 15 degree. Order=4: 3.75 degree. Order=5: 1.875 degree.")
	parser.add_argument("--useEQPS", action="store_true", help="Use EQPS instead of HEALPix. default = False")
	parser.add_argument("--EQPSangleDegree", type=float, default=10.0, help="Angular distance in degree on EQPS, default = 10")
	parser.add_argument("--apix", type=float, default=1.42, help="Pixel size in the output starfile, default = 1.42")
	parser.add_argument("--sym", type=str, default="C1", help="Point-group symmetry for removing equivalent directions: C1, Cn, Dn, tet, oct, I2, or I3. Default = C1, meaning no reduction. Aliases I/icos use RELION I2.")
	parser.add_argument("--symKeyTol", type=float, default=1.0e-10, help="Tolerance for choosing one representative from each symmetry orbit. Default = 1e-10")
	args = parser.parse_args()

	if args.psiStep <= 0:
		raise ValueError("--psiStep must be positive")

	num_psi = 360 // args.psiStep
	divisible = True
	if 360 % args.psiStep > 0.5:
		divisible = False
		print("Your Psi stepsize cannot be divisible to 360.")

	sym = args.sym.strip().lower()
	sym_mats = symmetry_matrices(sym)
	use_symmetry_reduction = len(sym_mats) > 1
	if use_symmetry_reduction:
		print("Use symmetry reduction: sym = %s, group order = %d" % (args.sym, len(sym_mats)))
	else:
		print("Use symmetry reduction: none (C1/full sphere)")

	rot_tilt_iter = iter_eqps_rot_tilt(args.EQPSangleDegree) if args.useEQPS else iter_healpix_rot_tilt(args.healpixOrder)

	all_count = 0
	count = 0
	with open(args.o, "w",newline="\n") as out:
		write_relion_header(out, args.apix)
		for rot, tilt in rot_tilt_iter:
			all_count += 1
			if args.discardPositiveRot and math.fmod(rot + 360.0, 360.0) > 180.0:
				continue
			if use_symmetry_reduction and not keep_direction_by_symmetry(rot, tilt, sym_mats, key_tol=args.symKeyTol):
				continue
			psi = random_psi(args.randomPsi, divisible, num_psi, args.psiStep)
			out.write(str(normalized_relion_rot(rot)) + "\t" + str(normalized_relion_tilt(tilt)) + "\t")
			out.write(str(psi) + "\t")
			out.write("1@1.mrcs\n")
			count += 1

	print("Total available points = ", all_count)
	print("Total points written = ", count)


if __name__ == "__main__":
	main()
