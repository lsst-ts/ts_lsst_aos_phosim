#!/usr/bin/env python
##
##
# @authors: Bo Xin
# @       Large Synoptic Survey Telescope

# CRS: coordinate system

def M1CRS2ZCRS(x, y, z):
	"""
	
	Do the coordinate transformation from M1 to Zemax.
	
	Arguments:
		x {[float]} -- x coordinate.
		y {[float]} -- y coordinate.
		z {[float]} -- z coordinate.

	Returns:
		[float] -- x, y, z in Zemax coordinate.
	"""

	return -x, y, -z

def ZCRS2M1CRS(x, y, z):
	"""
	
	Do the coordinate transformation from Zemax to M1.
	
	Arguments:
		x {[float]} -- x coordinate.
		y {[float]} -- y coordinate.
		z {[float]} -- z coordinate.

	Returns:
		[float] -- x, y, z in M1 coordinate.
	"""

	return -x, y, -z

def M2CRS2ZCRS(x, y, z):
	"""
	
	Do the coordinate transformation from M2 to Zemax.
	
	Arguments:
		x {[float]} -- x coordinate.
		y {[float]} -- y coordinate.
		z {[float]} -- z coordinate.

	Returns:
		[float] -- x, y, z in Zemax coordinate.
	"""

	return -x, y, -z

def ZCRS2M2CRS(x, y, z):
	"""
	
	Do the coordinate transformation from Zemax to M2.
	
	Arguments:
		x {[float]} -- x coordinate.
		y {[float]} -- y coordinate.
		z {[float]} -- z coordinate.

	Returns:
		[float] -- x, y, z in M2 coordinate.
	"""

	return -x, y, -z
