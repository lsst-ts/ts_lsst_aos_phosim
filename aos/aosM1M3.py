#!/usr/bin/env python
##
# @authors: Bo Xin
# @       Large Synoptic Survey Telescope

import os, sys
import numpy as np
from scipy.interpolate import Rbf

from cwfs.Tool import ZernikeAnnularFit, ZernikeAnnularEval

from aos.aosCoTransform import M1CRS2ZCRS, ZCRS2M1CRS

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import time

class aosM1M3(object):

    # Outer radius of M1 mirror in m
    R = 4.180

    # Inner radius of M1 mirror in m
    Ri = 2.558

    # Outer radius of M3 mirror in m
    R3 = 2.508

    # Inner radius of M3 mirror in m    
    R3i = 0.550

    # M1 optical design
    r1 = -1.9835e4
    k1 = -1.215
    alpha1 = np.zeros((8, 1))
    alpha1[2] = 1.38e-24

    # M3 optical design
    r3 = -8344.5
    k3 = 0.155
    alpha3 = np.zeros((8, 1))
    alpha3[2] = -4.5e-22
    alpha3[3] = -8.2e-30 # This value is -8.15e-30 in p.69 in phosim_reference.pdf. Check with Bo.

    # Number of z-axis actuators
    nzActuator = 156

    # Number of actuators
    nActuator = 256

    def __init__(self, M1M3dir, debugLevel=0):
        """
        
        Initiate the AocM1M3 class.
        
        Arguments:
            M1M3dir {[str]} -- Directory to M1M3 data.
        
        Keyword Arguments:
            debugLevel {int} -- Debug level. The higher value gives more information. 
                                (default: {0})
        """

        # Get the bending mode information
        data = np.loadtxt(os.path.join(M1M3dir, "M1M3_1um_156_grid.DAT"))
        self.nodeID = data[:, 0].astype("int")
        bx = data[:, 1]
        by = data[:, 2]
        bz = data[:, 3:]
        
        # Get the actuator force
        data = np.loadtxt(os.path.join(M1M3dir, "M1M3_1um_156_force.DAT"))
        self.force = data[:, :]

        # Show the information
        if (debugLevel >= 3):
            print("-b13--  %f" % bx[33])
            print("-b13--  %f" % by[193])
            print("-b13--  %d" % np.sum(self.nodeID == 1))
            print("-b13--  %e" % bz[332, 15])
            print("-b13--  %e" % bz[4332, 15])

        # Do the coordinate transformation from M1 coordinate to Zemax coordinate
        self.bx, self.by, self.bz = M1CRS2ZCRS(bx, by, bz)

        # Data needed to determine gravitational print through
        data = np.loadtxt(os.path.join(M1M3dir, "M1M3_dxdydz_zenith.txt"))
        self.zdx = data[:, 0]
        self.zdy = data[:, 1]
        self.zdz = data[:, 2]

        data = np.loadtxt(os.path.join(M1M3dir, "M1M3_dxdydz_horizon.txt"))
        self.hdx = data[:, 0]
        self.hdy = data[:, 1]
        self.hdz = data[:, 2]
        
        self.zf = np.loadtxt(os.path.join(M1M3dir, "M1M3_force_zenith.txt"))
        self.hf = np.loadtxt(os.path.join(M1M3dir, "M1M3_force_horizon.txt"))
        self.G = np.loadtxt(os.path.join(M1M3dir, "M1M3_influence_256.txt"))
        self.LUTfile = os.path.join(M1M3dir, "M1M3_LUT.txt")

        # Data needed to determine thermal deformation
        data = np.loadtxt(os.path.join(M1M3dir, "M1M3_thermal_FEA.txt"), skiprows=1)

        # These are the normalized coordinates
        # n.b. these may not have been normalized correctly, b/c max(tx)=1.0
        # Bo tried to go back to the xls data, max(x)=164.6060 in,
        # while 4.18m = 164.5669 in.
        tx = data[:, 0]
        ty = data[:, 1]
        
        # Below are in M1M3 coordinate system, and in micron

        # Do the fitting in the normalized coordinate
        normX = bx/self.R
        normY = by/self.R

        # Fit the bulk
        self.tbdz = self.__fitData(tx, ty, data[:, 2], normX, normY)
        
        # Fit the x-grad
        self.txdz = self.__fitData(tx, ty, data[:, 3], normX, normY)
        
        # Fit the y-grad
        self.tydz = self.__fitData(tx, ty, data[:, 4], normX, normY)
        
        # Fit the z-grad
        self.tzdz = self.__fitData(tx, ty, data[:, 5], normX, normY)

        # Fit the r-gradß
        self.trdz = self.__fitData(tx, ty, data[:, 6], normX, normY)

    def __fitData(self, dataX, dataY, data, x, y):
        """
        
        Fit the data by radial basis function.
        
        Arguments:
            dataX {[float]} -- Data x.
            dataY {[float]} -- Data y.
            data {[float]} -- Data to fit.
            x {[float]} -- x coordinate.
            y {[float]} -- y coordinate.
        
        Returns:
            [float] -- Fitted data.
        """

        # Construct the fitting model
        rbfi = Rbf(dataX, dataY, data)

        # Return the fitted data
        return rbfi(x, y)

    def getPrintthz(self, zAngle):
        """
        
        Get the mirror print along z direction in specific zenith angle. 

        It is noted that in the real condition, zenith angle is published by TMA and 
        M1M3. And TMA is preferred to use for higher accuracy. 
        
        Arguments:
            zAngle {[float]} -- Zenith angle.
        
        Returns:
            [float] -- Corrected projection in z direction.
        """

        # Do the M1M3 gravitational correction.
        # Map the changes of dx, dy, and dz on a plane for certain zenith angle 
        printthx = self.zdx * np.cos(zAngle) + self.hdx * np.sin(zAngle)
        printthy = self.zdy * np.cos(zAngle) + self.hdy * np.sin(zAngle)
        printthz = self.zdz * np.cos(zAngle) + self.hdz * np.sin(zAngle)

        # The temperatural correction is not considered here. Check with Bo to do this.

        # Transform the ZEMAX coordinate to M1 coordinate
        x, y, z = ZCRS2M1CRS(self.bx, self.by, self.bz)
        
        # Calcualte the mirror ideal shape
        zRef = self.__idealShape(x*1000, y*1000, self.nodeID)/1000
        
        # Calcualte the mirror ideal shape with the displacement
        zpRef = self.__idealShape((x + printthx)*1000, (y + printthy)*1000, self.nodeID)/1000

        # Convert printthz into surface sag to get the estimated wavefront error
        # Do the zenith angle correction by the linear approximation with the idea shape 
        printthz = printthz - (zpRef - zRef)

        # Normalize the coordinate
        normX = x/self.R
        normY = y/self.R
        obs = self.Ri/self.R
        
        # Fit the annular Zernike polynomials z0-z2 (piton, x-tilt, y-tilt)
        zc = ZernikeAnnularFit(printthz, normX, normY, 3, obs)

        # Do the estimated wavefront error correction for the mirror projection 
        printthz -= ZernikeAnnularEval(zc, normX, normY, obs)
        
        return printthz

    def __idealShape(self, x, y, annulus, dr1=0, dr3=0, dk1=0, dk3=0):
        """
        
        Calculate the ideal shape of mirror along z direction, which is described by a series of 
        cylindrically-symmetric aspheric surfaces.
        
        Arguments:
            x {[float]} -- coordinate x in 1D array in mm.
            y {[float]} -- coordinate y in 1D array in mm.
            annulus {[int]} -- Node of mirror. M1: 1; M3: 3.
        
        Keyword Arguments:
            dr1 {float} -- Displacement of r in mirror 1. (default: {0})
            dr3 {float} -- Displacement of r in mirror 3. (default: {0})
            dk1 {float} -- Displacement of kappa (k) in mirror 1. (default: {0})
            dk3 {float} -- Displacement of kappa (k) in mirror 3. (default: {0})

        Returns:
            [float] -- Ideal mirror surface along z direction.
        """

        # Get the dimension of input x, y
        nr = x.shape
        mr = y.shape
        if (nr != mr):
            print("In the ideal shape calculation, x is [%d] while y is [%d]." % (nr, mr))
            sys.exit()

        # Calculation the curvature (c) and conic constant (kappa)

        # Mirror 1 (M1)
        c1 = 1/(self.r1 + dr1)
        k1 = self.k1 + dk1
        
        # Mirror 3 (M3)
        c3 = 1/(self.r3 + dr3)
        k3 = self.k3 + dk3

        # Get the mirror index
        idxM1 = (annulus == 1)
        idxM3 = (annulus == 3)

        # Construct the curvature, kappa, and alpha matrixes for the ideal shape calculation
        cMat = np.zeros(nr)        
        cMat[idxM1] = c1
        cMat[idxM3] = c3

        kMat = np.zeros(nr)
        kMat[idxM1] = k1
        kMat[idxM3] = k3

        alphaMat = np.tile(np.zeros(nr), (8, 1))        
        for ii in range(8):
            alphaMat[ii, idxM1] = self.alpha1[ii]
            alphaMat[ii, idxM3] = self.alpha3[ii]

        # Calculate the radius
        r2 = x**2 + y**2
        r = np.sqrt(r2)

        # Calculate the ideal surface

        # The optical elements of telescopes can often be described by a series of 
        # cylindrically-symmetric aspheric surfaces:
        # z(r) = c * r^2/[ 1 + sqrt( 1-(1+k) * c^2 * r^2 ) ] + sum(ai * r^(2*i)) + sum(Aj * Zj)
        # where i = 1-8, j = 1-N

        z0 = cMat * r2 / (1 + np.sqrt(1 - (1 + kMat) * cMat**2 * r2))
        for ii in range(8):
            z0 += alphaMat[ii, :] * r2**(ii+1)

        # M3 vertex offset from M1 vertex, values from Zemax model
        # M3voffset = (233.8 - 233.8 - 900 - 3910.701 - 1345.500 + 1725.701 + 3530.500 + 900 + 233.800)
        M3voffset = 233.8

        # Add the M3 offset (sum(Aj * Zj), j = 1 - N)
        z0[idxM3] = z0[idxM3] + M3voffset

        # In Zemax, z axis points from M1M3 to M2. the reversed direction (z0>0) is needed.
        # That means the direction of M2 to M1M3.
        return -z0

def plot3dInSurface(x, y, z, nPoint=201, title=""):
    """
    
    Plot the 3d data in surface. 
    
    Arguments:
        x {[float]} -- Input x data in array.
        y {[float]} -- Input y data in array.
        z {[float]} -- Input z data in array.
    
    Keyword Arguments:
        nPoint {number} -- Number of point in single axis. (default: {201})
        title {str} -- Title of figure. (default: {""})
    """

    # Generate the 3D map model by radial basis function (Rbf)
    rbfi = Rbf(x, y, z)

    # Generat the x, y grids
    newX = np.linspace(np.round(np.min(x)), np.round(np.max(x)), nPoint)
    newY = np.linspace(np.round(np.min(y)), np.round(np.max(y)), nPoint)
    newX, newY = np.meshgrid(newX, newY)

    # Get the fitted data based on the model
    data = rbfi(newX, newY)

    # Plot the image
    plt.figure()
    plt.imshow(data, extent = [np.min(newX), np.max(newX), np.min(newY), np.max(newY)], 
               origin="lower")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.colorbar()
    plt.show()

if __name__ == "__main__":

    # M1M3 directory
    M1M3dir = "../data/M1M3"

    # Instantiate the AocM1M3
    M1M3 = aosM1M3(M1M3dir)

    # Calculate the print thz
    x, y, temp = ZCRS2M1CRS(M1M3.bx, M1M3.by, M1M3.bz)
    x = x/M1M3.R
    y = y/M1M3.R

    printthz = M1M3.getPrintthz(3)
    print(printthz)

    printthz = M1M3.getPrintthz(-2)
    print(printthz)

    # Plot the surface
    # plot3dInSurface(x, y, printthz)
    
    
