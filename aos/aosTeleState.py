#!/usr/bin/env python

# @author: Bo Xin
# @      Large Synoptic Survey Telescope

import os
#import sys
import shutil
import glob
# import subprocess
import multiprocessing

import numpy as np
from astropy.io import fits
from astropy.time import Time
import aos.aosCoTransform as ct
from scipy.interpolate import Rbf

from lsst.cwfs.tools import ZernikeAnnularFit, ZernikeFit, ZernikeEval, extractArray

from aos.aosUtility import getInstName, isNumber, getLUTforce, hardLinkFile, runProgram, writeToFile

import matplotlib.pyplot as plt

# Active filter ID in PhoSim
phosimFilterID = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "y": 5}

class aosTeleState(object):

    # Instrument name
    LSST = "lsst"

    # Effective wavelength
    effwave = {"u": 0.365, "g": 0.480, "r": 0.622,
               "i": 0.754, "z": 0.868, "y": 0.973}

    # Gaussain quadrature (GQ) points xi
    # Check the condition of "u", "z", and "y" with Bo. What is the meaning of place holder?
    # Does that mean the analysis is not done yet? Or is the first order approximation good enough?
    # The same question of weighting ratio of GQ for Bo.
    GQwave = {"u": [0.365], # place holder
              "g": [0.41826, 0.44901, 0.49371, 0.53752],
              "r": [0.55894, 0.59157, 0.63726, 0.68020],
              "i": [0.70381, 0.73926, 0.78798, 0.83279],
              "z": [0.868], # place holder
              "y": [0.973]} # place holder

    # Gaussian quadrature (GQ) weighting ratio
    # Check with Bo for "u", "z", and "y"
    GQwt = {"u": [1],
            "g": [0.14436, 0.27745, 0.33694, 0.24125],
            "r": [0.15337, 0.28587, 0.33241, 0.22835],
            "i": [0.15810, 0.29002, 0.32987, 0.22201],
            "z": [1],
            "y": [1]}

    def __init__(self, simuParamDataDir, inst, simuParamFileName, iSim, ndofA, phosimDir,
                 pertDir, imageDir, band, wavelength, endIter, M1M3=None, M2=None,
                 opSimHisDir=None, debugLevel=0):
        """

        Initiate the aosTeleState class.

        Arguments:
            simuParamDataDir {[str]} -- Directory of simulation parameter files.
            inst {[str]} -- Instrument name (e.g. lsst or comcam10).
            simuParamFileName {[str]} -- Simulation parameter file name.
            iSim {[int]} -- Simulation number.
            ndofA {[int]} -- Number of degree of freedom.
            phosimDir {[str]} -- Directory of PhoSim.
            pertDir {[str]} -- Directory of perturbation files.
            imageDir {[str]} -- Directory of image files.
            band {[str]} -- Band of active filter.
            wavelength {[float]} -- Single chromatic light wavelength in um. If the value is 0, the
                                    effective wavelength in each band is used.
            endIter {[int]} -- End iteration number.

        Keyword Arguments:
            M1M3 {[aosM1M3]} -- aosM1M3 object. (default: {None})
            M2 {[aosM2]} -- aosM2 object. (default: {None})
            opSimHisDir {[str]} -- Directory of OpSim Observation history. (default: {None})
            debugLevel {int} -- Debug level. The higher value gives more information.
                                (default: {0})
        """

        # Iteration number
        self.iIter = None

        # Type of active filter. It is noted that if the monochromatic wavelength (0.5 um)
        # is used, the band is "g".
        self.band = band

        # Monochromatic wavelength. The referenced wavelength is 0.5 um. If the value is 0, the
        # effective wavelength in each band is used.
        self.wavelength = wavelength

        # Get the effective wavelength and number of OPD
        self.effwave, self.nOPDw = getTelEffWave(wavelength, band)

        # Telescope state in the basis of degree of freedom (DOF)
        self.stateV = np.zeros(ndofA)

        # Get the instrument name
        self.inst = getInstName(inst)[0]

        # Path to the simulation parameter data
        self.instruFile = os.path.join(simuParamDataDir, (simuParamFileName + ".inst"))

        # Parameters of telescope state
        self.time0 = None
        self.iqBudget = None
        self.eBudget = None
        self.zAngle = None
        self.camTB = None
        self.camRot = None
        self.M1M3ForceError = None
        self.M1M3TxGrad = None
        self.M1M3TyGrad = None
        self.M1M3TzGrad = None
        self.M1M3TrGrad = None
        self.M1M3TBulk = None
        self.M2TzGrad = None
        self.M2TrGrad = None
        self.znPert = None
        self.surfaceGridN = None
        self.opdSize = None
        self.eimage = None
        self.psfMag = None
        self.cwfsMag = None
        self.cwfsStampSize = None

        self.M1M3surf = None
        self.M2surf = None

        self.stateV0 = None

        # Read the file to get the parameters
        self.__readFile(self.instruFile)

        # Update zAngle for multiple iteration
        if (self.zAngle is not None):
            self.zAngle = self.__getZenithAngleForIteration(self.zAngle, endIter, opSimHisDir=opSimHisDir)

        # Change the unit to arcsec
        # Check with Bo for the original unit defined in GT.inst file because
        # I do not understand the meaning of 1e-3 here
        if (self.iqBudget is not None):
            self.iqBudget = self.iqBudget*1e-3

        # Get the size of point spread function (PSF) stamp
        # Check with Bo for the math here
        fno = 1.2335
        k = fno*self.effwave/0.2
        self.psfStampSize = int(self.opdSize + np.rint((self.opdSize * (k-1) + 1e-5) / 2) * 2)

        # Simulation number
        self.iSim = iSim

        # Directory of PhoSim
        self.phosimDir = phosimDir

        # Directory of perturbation files
        self.pertDir = pertDir

        # Directory of image files
        self.imageDir = imageDir

        # Actuator ID in PhoSim
        # M2 z, x, y, rx, ry (5, 6, 7, 8, 9)
        # Cam z, x, y, rx, ry (10, 11, 12, 13, 14)
        # M1M3 and M2 bending modes
        # Check with Bo for the index of "nodfA" in PhoSim. It is wield if M1M3 and M2 have different
        # bending modes. Does M1M3 begin from 15 and M2 begin from 35
        # It looks like the index in PhoSim has been hardcoded based on writePertFile(). Check with Bo.
        phoSimStartIdx = 5
        self.phosimActuatorID = range(phoSimStartIdx, ndofA+phoSimStartIdx)

        # x-, y-coordinate in the OPD image
        opdGrid1d = np.linspace(-1, 1, self.opdSize)
        self.opdx, self.opdy = np.meshgrid(opdGrid1d, opdGrid1d)

        # Show the information of OPD image grid or not
        if (debugLevel >= 3):
            print("in aosTeleState:")
            print(self.stateV)
            print(opdGrid1d.shape)
            print(opdGrid1d[0])
            print(opdGrid1d[-1])
            print(opdGrid1d[-2])

        # Get the spectral energy distribution (SED) file name
        self.sedfile = self.__getSedFileName(self.wavelength, phosimDir)

        # Assign the seed of random number by using the simulation number
        np.random.seed(self.iSim)

        # Get the print of M1M3 and M2 along z-axis and add the random value to mirror surfaces
        if (self.zAngle is not None):

            # Get the zenith angle in iteration 0
            zenithAngle0 = self.zAngle[0]

            # Set the mirror print along z direction of M1M3 in iteration 0
            M1M3.setPrintthz_iter0(M1M3.getPrintthz(zenithAngle0))

            # Get the actuator forces of M1M3 based on the look-up table (LUT)
            LUTforce = getLUTforce(zenithAngle0/np.pi*180, M1M3.LUTfile)

            # Add 5% force error (self.M1M3ForceError). This is for iteration 0 only.
            # This means from -5% to +5% of original actuator's force.
            myu = (1 + 2*(np.random.rand(M1M3.nActuator) - 0.5)*self.M1M3ForceError)*LUTforce

            # Balance forces along z-axis
            # Not sure this part. Need to check with Bo for the arangement of LUT force
            myu[M1M3.nzActuator-1] = np.sum(LUTforce[:M1M3.nzActuator]) - np.sum(myu[:M1M3.nzActuator-1])

            # Balance forces along y-axis
            # Not sure this part. Need to check with Bo for the arangement of LUT force
            myu[M1M3.nActuator-1] = np.sum(LUTforce[M1M3.nzActuator:]) - np.sum(myu[M1M3.nzActuator:-1])

            # Get the net force along the z-axis
            u0 = M1M3.zf*np.cos(zenithAngle0) + M1M3.hf*np.sin(zenithAngle0)

            # Add the error to the M1M3 surface and change the unit to um
            # It looks like there is the unit inconsistency between M1M3 and M2
            # This inconsistency comes from getPrintthz() in mirror. But I do not understand the condition
            # in M1M3 compared with M2.
            # Check this with Bo.
            # Suggeest to change the unit in aosM1M3 class at printthz_iter0() and G. Check with Bo.
            self.M1M3surf = (M1M3.printthz_iter0 + M1M3.G.dot(myu - u0))*1e6

            # Set the mirror print along z direction of M2 in iteration 0
            # When considering the update(), do we need this statement? Check with Bo.
            M2.setPrintthz_iter0(M2.getPrintthz(zenithAngle0))

            # This value should be copy of M2.printthz_iter0.
            self.M2surf = M2.printthz_iter0.copy()

        # Do the temperature correction for mirror surface
        # Need to check with Bo to put this in run-time change for continuing update temperature or not.
        if ((self.M1M3TBulk is not None) and (self.M1M3surf is not None)):

            self.M1M3surf += self.M1M3TBulk*M1M3.tbdz + self.M1M3TxGrad*M1M3.txdz \
                             + self.M1M3TyGrad*M1M3.tydz + self.M1M3TzGrad*M1M3.tzdz \
                             + self.M1M3TrGrad*M1M3.trdz
            self.M2surf += self.M2TzGrad*M2.tzdz + self.M2TrGrad*M2.trdz

        # Do the coordinate transformation (z-axis only) from M1M3 to Zemax.
        if (self.M1M3surf is not None):
            self.M1M3surf = ct.M1CRS2ZCRS(0, 0, self.M1M3surf)[2]

        # Do the coordinate transformation (z-axis only) from M2 to Zemax.
        if (self.M2surf is not None):
            self.M2surf = ct.M2CRS2ZCRS(0, 0, self.M2surf)[2]

        # Get the camera distortion
        # Check with Bo that we need to allow the run time change of this variable or not.
        # I believe the answer should be yes.
        # This part may be removed
        # It looks like only lsst camera will do the distortion correction. Is this true?
        # Check this part with Bo.
        if (self.camRot is not None):
            # Check the inputs here with Bo.
            self.__getCamDistortionAll(simuParamDataDir, self.zAngle[0], 0, 0, 0)

    def __getZenithAngleForIteration(self, zAngle, endIter, opSimHisDir=None):
        """

        Get the zenith angle for iteration use.

        Arguments:
            zAngle {[str]} -- Zenith angle setting.
            endIter {[int]} -- End iteratioin number.

        Keyword Arguments:
            opSimHisDir {[str]} -- Directory of OpSim Observation History. (default: {None})

        Returns:
            [ndarray] -- Zenith angles in iteration.
        """

        # Check the value is a number or not
        if isNumber(zAngle):
            # Change the unit from the degree to radian
            # When startIter>0, we still need to set M1M3.printthz_iter0 correctly
            zAngleIter = np.ones(endIter+1)*float(zAngle)/180*np.pi

        else:
            # zAngle is extracted from OpSim ObsHistory.
            # This is 90-block['altitude'].values[:100]/np.pi*180 --> Check the data format with Bo.
            zenithAngleFile = os.path.join(opSimHisDir, (zAngle + ".txt"))
            obsData = np.loadtxt(zenithAngleFile).reshape((-1, 1))

            # Make sure the data format is correct. Raise the error message if it is not.
            assert obsData.shape[0]>endIter, "OpSim data array length < iteration number"
            assert np.max(obsData)<90, "Maximum of zenith angle >= 90 degree"
            assert np.min(obsData)>0, "Minimum of zenith angle <= 0 degree"

            # Change the unit from the degree to radian
            zAngleIter = obsData[:endIter+1, 0]/180*np.pi

        return zAngleIter

    def __getSedFileName(self, wavelengthInUm, phosimDir):
        """

        Get the spectral energy distribution (SED) file name. For the single chromatic light, the
        related file will be generated if it does not exist.

        Arguments:
            wavelengthInUm {[float]} -- Wavelength in um.
            phosimDir {[str]} -- Directory of PhoSim.

        Returns:
            [str] -- SED file name.
        """

        if (wavelengthInUm == 0):
            sedfileName = "sed_flat.txt"
        else:
            # Unit is changed to nm
            sedfileName = "sed_%d.txt" % (wavelengthInUm*1e3)

            # Create the sed file for single chromatic light if the sed file does not exist
            pathToSedFile = os.path.join(phosimDir, "data", "sky", sedfileName)
            if not os.path.isfile(pathToSedFile):
                fsed = open(pathToSedFile, "w")
                fsed.write("%d   1.0\n" % (wavelengthInUm*1e3))
                fsed.close()

        return sedfileName

    def __readFile(self, filePath):
        """

        Read the AOS telescope simulation parameter file.

        Arguments:
            filePath {[str]} -- Path to telescope simulation paramter file (*.inst).
        """

        # Parameters used in reading the file
        iscomment = False

        # Read the file
        fid = open(filePath)
        for line in fid:

            # Strip the line
            line = line.strip()

            # Ignore the comment part
            if (line.startswith("###")):
                iscomment = ~iscomment

            if (not(line.startswith("#")) and (not iscomment) and len(line) > 0):

                # Used for the single degree of freedom (DOF) unit testing
                # Check with Bo the reason to have this
                if (line.startswith("dof")):

                    # Get the values
                    index, newValue, unit = line.split()[1:4]

                    # There is the "-1" here for the index. The index in python array begins from 0.
                    index -= 1

                    # Assign the value
                    self.stateV[index] = newValue

                    # By default, use micron and arcsec as units
                    if (unit == "mm"):
                        self.stateV[index] *= 1e3

                    elif (unit == "deg"):
                        self.stateV[index] *= 3600

                # Camera mjd
                elif (line.startswith("mjd")):
                    self.time0 = Time(float(line.split()[1]), format="mjd")

                # Budget of image quality
                # Check with Bo how to define this
                elif (line.startswith("iqBudget")):
                    # Read in mas, convert to arcsec (Check with Bo the meaning of "mas" here)
                    self.iqBudget = np.sqrt(np.sum(np.array(line.split()[1:], dtype=float)**2))

                # Budget of ellipticity
                # Check with Bo how to define this
                elif (line.startswith("eBudget")):
                    self.eBudget = float(line.split()[1])

                # Get the zenith angle or file name
                elif (line.startswith("zenithAngle")):
                    self.zAngle = line.split()[1]

                # Camera temperature in degree C
                # Ignore this if the instrument is comcam (Check with Bo for this ignorance)
                elif (line.startswith("camTB") and self.inst == self.LSST):
                    self.camTB = float(line.split()[1])

                    # Check with Bo how to get this compenstation of iqBudget
                    self.iqBudget = np.sqrt(self.iqBudget**2 + float(line.split()[3])**2)

                # Camera rotation angle
                # Ignore this if the instrument is comcam (Check with Bo for this ignorance)
                elif (line.startswith("camRotation") and self.inst == self.LSST):
                    self.camRot = float(line.split()[1])

                    # Check with Bo how to get this compenstation of iqBudget
                    self.iqBudget = np.sqrt(self.iqBudget**2 + float(line.split()[3])**2)

                # Check with Bo for the unit of this
                elif (line.startswith("M1M3ForceError")):
                    self.M1M3ForceError = float(line.split()[1])

                # Check with Bo how to get these temperature related parameters
                elif (line.startswith("M1M3TxGrad")):
                    self.M1M3TxGrad = float(line.split()[1])

                elif (line.startswith("M1M3TyGrad")):
                    self.M1M3TyGrad = float(line.split()[1])

                elif (line.startswith("M1M3TzGrad")):
                    self.M1M3TzGrad = float(line.split()[1])

                elif (line.startswith("M1M3TrGrad")):
                    self.M1M3TrGrad = float(line.split()[1])

                elif (line.startswith("M1M3TBulk")):
                    self.M1M3TBulk = float(line.split()[1])

                elif (line.startswith("M2TzGrad")):
                    self.M2TzGrad = float(line.split()[1])

                elif (line.startswith("M2TrGrad")):
                    self.M2TrGrad = float(line.split()[1])

                # Maximum number of Zn to define the surface in perturbation file
                # Check I can use "znPert = 20" or not. This is for the annular Zk in my WEP code.
                # Check the reason to use the spherical Zk. If I use annular Zk, what will happen?
                # From the phosim command, does it mean only spherical Zk is allowed?
                # https://bitbucket.org/phosim/phosim_release/wiki/Physics%20Commands
                # But the dof and uk is in the basis of annular Zk, which is different from PhoSim.
                # Check with Bo for this
                elif (line.startswith("znPert")):
                    self.znPert = int(line.split()[1])

                # Check with Bo for the meaning of this
                elif (line.startswith("surfaceGridN")):
                    self.surfaceGridN = int(line.split()[1])

                # Size of OPD image (n x n matrix)
                elif (line.startswith("opd_size")):
                    self.opdSize = int(line.split()[1])

                    # Make sure the value is odd
                    if (self.opdSize % 2 == 0):
                        self.opdSize -= 1

                # Use 0 for eimage only, without backgrounds
                elif (line.startswith("eimage")):
                    self.eimage = bool(int(line.split()[1]))

                # Magnitude of point spread function (PSF) star
                elif (line.startswith("psf_mag")):
                    self.psfMag = int(line.split()[1])

                # Magnitude of wavefront sensing star
                elif (line.startswith("cwfs_mag")):
                    self.cwfsMag = int(line.split()[1])

                # Wavefront sensing image stamp size
                elif (line.startswith("cwfs_stamp_size")):
                    self.cwfsStampSize = int(line.split()[1])

        fid.close()

    def setIterNum(self, iIter):
        """

        Set the iteration number.

        Arguments:
            iIter {[int]} -- Iteration number.
        """

        self.iIter = iIter

    def setPerFilePath(self, metr, wfs=None):
        """

        Set the path of purturbation files.

        Arguments:
            metr {[aosMetric]} -- aosMetric object.

        Keyword Arguments:
            wfs {[aosWFS]} -- aosWFS object. (default: {None})
        """

        # Generate the iteration folder
        iterFolder = "iter%d" % self.iIter
        self.__generateFolder(self.imageDir, iterFolder)
        self.__generateFolder(self.pertDir, iterFolder)

        # Set the path of perturbation file
        self.pertFile = os.path.join(self.pertDir, iterFolder,
                                     "sim%d_iter%d_pert.txt" % (self.iSim, self.iIter))

        # Replace the file extention to ".cmd"
        self.pertCmdFile = os.path.splitext(self.pertFile)[0] + ".cmd"

        # Replace the file extention to ".mat"
        self.pertMatFile = os.path.splitext(self.pertFile)[0] + ".mat"

        # Set the path of PSSN and ellipticity file path
        # Need to put this one to aosMetric class in the future
        metr.PSSNFile = os.path.join(self.imageDir, iterFolder,
                                     "sim%d_iter%d_PSSN.txt" % (self.iSim, self.iIter))
        metr.elliFile = os.path.join(self.imageDir, iterFolder,
                                     "sim%d_iter%d_elli.txt" % (self.iSim, self.iIter))

        # Set the path related to OPD information
        self.OPD_inst = os.path.join(self.pertDir, iterFolder,
                                     "sim%d_iter%d_opd%d.inst" % (self.iSim, self.iIter, metr.nFieldp4))

        self.OPD_cmd = os.path.splitext(self.OPD_inst)[0] + ".cmd"

        self.OPD_log = os.path.join(self.imageDir, iterFolder,
                                    "sim%d_iter%d_opd%d.log" % (self.iSim, self.iIter, metr.nFieldp4))

        self.zTrueFile = os.path.join(self.imageDir, iterFolder,
                                      "sim%d_iter%d_opd.zer" % (self.iSim, self.iIter))

        # Set the path of atmosphere for two exposures
        self.atmFile = [os.path.join(self.imageDir, iterFolder,
                            "sim%d_iter%d_E00%d.atm" % (self.iSim, self.iIter, iexp)) for iexp in [0, 1]]

        # Set the path of residue related to mirror surface
        if (self.M1M3surf is not None):
            self.M1M3zlist = os.path.join(self.pertDir, iterFolder, "sim%d_M1M3zlist.txt" % self.iSim)
            self.resFile1 = os.path.join(self.pertDir, iterFolder, "sim%d_M1res.txt" % self.iSim)
            self.resFile3 = os.path.join(self.pertDir, iterFolder, "sim%d_M3res.txt" % self.iSim)

        if (self.M2surf is not None):
            self.M2zlist = os.path.join(self.pertDir, iterFolder, "sim%d_M2zlist.txt" % self.iSim)
            self.resFile2 = os.path.join(self.pertDir, iterFolder, "sim%d_M2res.txt" % self.iSim)

        # Set the file path related to wavefront sensor (aosWFS class)
        if (wfs is not None):
            wfs.zFile = [os.path.join(self.imageDir, iterFolder,
                            "sim%d_iter%d_E00%d.z4c" % (self.iSim, self.iIter, iexp)) for iexp in [0, 1]]
            wfs.catFile = [os.path.join(self.pertDir, iterFolder,
                                        "wfs_catalog_E00%d.txt" % iexp) for iexp in [0, 1]]
            wfs.zCompFile = os.path.join(self.pertDir, iterFolder, "checkZ4C_iter%d.png" % self.iIter)

            # Instrument file contains the information of stars and DOF used in PhoSim
            if (wfs.nRun == 1):
                self.WFS_inst = [os.path.join(self.pertDir, iterFolder,
                                              "sim%d_iter%d_wfs%d.inst" % (self.iSim, self.iIter, wfs.nWFS))]
            else:
                # There is a problem here. wfs.nRun > 1 for ComCam. But ComCam has no half chips. Check this with Bo.
                self.WFS_inst = [os.path.join(self.pertDir, iterFolder,
                                 "sim%d_iter%d_wfs%d_%s.inst" % (self.iSim, self.iIter, wfs.nWFS, wfs.halfChip[irun]))
                                 for irun in range(wfs.nRun)]

            # Log file from PhoSim
            if (wfs.nRun == 1):
                self.WFS_log = [os.path.join(self.imageDir, iterFolder,
                                              "sim%d_iter%d_wfs%d.log" % (self.iSim, self.iIter, wfs.nWFS))]
            else:
                # There is a problem here. wfs.nRun > 1 for ComCam. But ComCam has no half chips. Check this with Bo.
                self.WFS_log = [os.path.join(self.imageDir, iterFolder,
                                "sim%d_iter%d_wfs%d_%s.log" % (self.iSim, self.iIter, wfs.nWFS, wfs.halfChip[irun]))
                                for irun in range(wfs.nRun)]

            # Command file used in PhoSim
            self.WFS_cmd = os.path.join(self.pertDir, iterFolder,
                                        "sim%d_iter%d_wfs%d.cmd" % (self.iSim, self.iIter, wfs.nWFS))

    def readPrePerFile(self, metr=None, wfs=None):
        """

        Read the perturbation data from the previous run.

        Keyword Arguments:
            metr {[aosMetric]} -- aosMetric object. (default: {None})
            wfs {[aosWFS]} -- aosWFS object. (default: {None})
        """

        # Read the data in the previous iteration
        if (self.iIter > 0):

            # Set the path of perturbation matrix file in the iteration 0
            self.pertMatFile_0 = os.path.join(self.pertDir, "iter0", "sim%d_iter0_pert.mat" % self.iSim)

            # Set the previous iteration folder
            preIterFolder = "iter%d" % (self.iIter-1)

            # Set the file path in the previous iteration
            self.zTrueFile_m1 = os.path.join(self.imageDir, preIterFolder,
                                             "sim%d_iter%d_opd.zer" % (self.iSim, self.iIter-1))

            self.pertMatFile_m1 = os.path.join(self.pertDir, preIterFolder,
                                               "sim%d_iter%d_pert.mat" % (self.iSim, self.iIter-1))

            # Read the file in the previous iteration
            self.stateV = np.loadtxt(self.pertMatFile_m1)
            self.stateV0 = np.loadtxt(self.pertMatFile_0)

            # Read the wavefront in zk in the previous iteration
            if (wfs is not None):
                wfs.zFile_m1 = [os.path.join(self.imageDir, preIterFolder,
                                "sim%d_iter%d_E00%d.z4c" % (self.iSim, self.iIter-1, iexp))
                                for iexp in [0, 1]]

            # PSSN from last iteration needs to be known for shiftGear
            if (metr is not None) and (metr.GQFWHMeff is None):
                PSSNFile_m1 = os.path.join(self.imageDir, preIterFolder,
                                           "sim%d_iter%d_PSSN.txt" % (self.iSim, self.iIter-1))
                data = np.loadtxt(PSSNFile_m1)
                metr.GQFWHMeff = data[1, -1]

    def __generateFolder(self, pathToFolder, newFolder):
        """

        Generate the folder if the folder does not exist.

        Arguments:
            pathToFolder {[str]} -- Path to folder.
            newFolder {[str]} -- Name of new folder.
        """

        # Generate the path of new folder
        pathToNewFolder = os.path.join(pathToFolder, newFolder)

        # Check the existence of folder
        if (not os.path.exists(pathToNewFolder)):
            # Create the folder if it does not exist
            os.makedirs(pathToNewFolder)

    def update(self, uk, ctrlRange, M1M3=None, M2=None):
        """

        Update the aggregated degree of freedom (DOF) and mirror surface if the elevation angle changes.

        Arguments:
            uk {[ndarray]} -- New estimated DOF from the optimal controller.
            ctrlRange {[ndarray]} -- Allowed moving range for each DOF.

        Keyword Arguments:
            M1M3 {[aosM1M3]} -- aosM1M3 object. (default: {None})
            M2 {[aosM2]} -- aosM2 object. (default: {None})

        Raises:
            RuntimeError -- Aggregated DOF overs the allowed range.
        """

        # Update the aggregated telescope state based on the estimated degree of freedom (DOF)
        # by the control algorithm
        self.stateV += uk

        # Check the aggegated correction in the basis of DOF is in the allowed range or not
        if np.any(self.stateV > ctrlRange):
            ii = (self.stateV > ctrlRange).argmax()
            raise RuntimeError("stateV[%d] = %e > its range = %e" % (ii, self.stateV[ii], ctrlRange[ii]))

        # If the elevation angle changes, the print through maps need to change also.
        if (self.M1M3surf is not None):
            # Check this statement with Bo. If there is the unit inconsistency in the initialization (* 1e6),
            # there should be it here also.

            # By the way, if considering the elevation angle change, the statement should be something like:
            # if (self.iIter >= 1):
            #     self.M1M3surf += (M1M3.getPrintthz(self.zAngle[self.iIter]) - M1M3.getPrintthz(self.zAngle[self.iIter-1]))*1e6

            self.M1M3surf += M1M3.getPrintthz(self.zAngle[self.iIter]) - M1M3.printthz_iter0

        if (self.M2surf is not None):
            self.M2surf += M2.getPrintthz(self.zAngle[self.iIter]) - M2.printthz_iter0

    def __getCamDistortionAll(self, simuParamDataDir, zAngle, pre_elev, pre_camR, pre_temp_camR):
        """

        Get the camera distortion parameters.

        Arguments:
            simuParamDataDir {[str]} -- Directory of simulation parameter files.
            zAngle {[float]} -- Zenith angle.
            pre_elev {[float]} -- ?? Check with Bo.
            pre_camR {[float]} -- ?? Check with Bo.
            pre_temp_camR {[float]} -- ?? Check with Bo.
        """

        # List of camera distortion
        # The first 5 types look like not to be used. Check with Bo.
        distTypeList = ["L1RB", "L2RB", "FRB", "L3RB", "FPRB",
                        "L1S1zer", "L2S1zer", "L3S1zer", "L1S2zer", "L2S2zer", "L3S2zer"]
        for distType in distTypeList:
            self.__getCamDistortion(simuParamDataDir, zAngle, distType, pre_elev, pre_camR, pre_temp_camR)

    def __getCamDistortion(self, simuParamDataDir, zAngle, distType, pre_elev, pre_camR, pre_temp_camR):
        """

        Get the camera distortion correction.

        Arguments:
            simuParamDataDir {[str]} -- Directory of simulation parameter files.
            zAngle {[float]} -- Zenith angle.
            distType {[str]} -- Distortion type.
            pre_elev {[float]} -- ?? Check with Bo.
            pre_camR {[float]} -- ?? Check with Bo.
            pre_temp_camR {[float]} -- ?? Check with Bo.
        """

        # Path to camera distortion parameter file
        dataFile = os.path.join(simuParamDataDir, "camera", (distType + ".txt"))

        # Read the distortion
        data = np.loadtxt(dataFile, skiprows=1)

        # Calculate the distortion (dx, dy, dz, rx, ry, rz)
        # Check with Bo for the math here
        distFun = lambda zenithAngle, camRotAngle: data[0, 3:]*np.cos(zenithAngle) + \
                    ( data[1, 3:]*np.cos(camRotAngle) + data[2, 3:]*np.sin(camRotAngle) )*np.sin(zenithAngle)
        distortion = distFun(zAngle, self.camRot) - distFun(pre_elev, pre_camR)

        # Do the temperature correction by the simple temperature interpolation/ extrapolation
        # If the temperature is too low, use the lowest listed temperature to do the correction.
        # List of data:
        # [ze. angle, camRot angle, temp (C), dx (mm), dy (mm), dz (mm), Rx (rad), Ry (rad), Rz (rad)]
        startTempRowIdx = 3
        endTempRowIdx = 10
        if (self.camTB <= data[startTempRowIdx, 2]):
            distortion += data[startTempRowIdx, 3:]

        # If the temperature is too high, use the highest listed temperature to do the correction.
        elif (self.camTB >= data[endTempRowIdx, 2]):
            distortion += data[endTempRowIdx, 3:]

        # Get the correction value by the linear fitting
        else:

            # Find the temperature boundary indexes
            p2 = (data[startTempRowIdx:, 2] > self.camTB).argmax() + startTempRowIdx
            p1 = p2-1

            # Calculate the linear weighting
            w1 = (data[p2, 2] - self.camTB) / (data[p2, 2] - data[p1, 2])
            w2 = 1-w1
            distortion += w1*data[p1, 3:] + w2*data[p2, 3:]

        # Minus the reference temperature correction. There is the problem here.
        # If the pre_temp_carR is not on the data list, this statement will fail/ get nothing.
        distortion -= data[(data[startTempRowIdx:, 2] == pre_temp_camR).argmax() + startTempRowIdx, 3:]

        # The order/ index of Zernike corrections by Andy in file is different from PhoSim use.
        # Reorder the correction here for PhoSim to use.
        if (distType[-3:] == "zer"):
            zidx = [1, 3, 2, 5, 4, 6, 8, 9, 7, 10, 13, 14, 12, 15, 11, 19,
                    18, 20, 17, 21, 16, 25, 24, 26, 23, 27, 22, 28]
            # The index of python begins from 0.
            distortion = distortion[[x - 1 for x in zidx]]

        # This statement is bad. Check where do we need these attribute. Do we really need the attribute?
        # Or we can just put it in the perturbation file writing.
        setattr(self, distType, distortion)

    def getPertFilefromBase(self, baserun):
        """

        Hard link the perturbation files to avoid the repeated calculation.

        Arguments:
            baserun {[int]} -- Source simulation number (basement run).
        """

        # File list to do the hard link
        hardLinkFileList = [self.pertFile, self.pertMatFile, self.pertCmdFile, self.M1M3zlist,
                            self.resFile1, self.resFile3, self.M2zlist, self.resFile2]

        # Hard link the file to avoid the repeated calculation
        for aFile in hardLinkFileList:
            hardLinkFile(aFile, baserun, self.iSim)

    def writePertFile(self, ndofA, M1M3=None, M2=None):
        """

        Write the perturbation file for the aggregated degree of freedom of telescope, mirror
        surface on z-axis in Zk, and camera lens distortion in Zk.

        Arguments:
            ndofA {[int]} -- Number of degree of freedom.

        Keyword Arguments:
            M1M3 {[aosM1M3]} -- aosM1M3 object. (default: {None})
            M2 {[aosM2]} -- aosM2 object. (default: {None})
        """

        # Write the perturbation file (aggregated degree of freedom of telescope)
        fid = open(self.pertFile, "w")
        for ii in range(ndofA):
            if (self.stateV[ii] != 0):
                # Don't add comments after each move command,
                # Phosim merges all move commands into one!
                fid.write("move %d %7.4f \n" % (self.phosimActuatorID[ii], self.stateV[ii]))

        fid.close()

        # Save the state of telescope into the perturbation matrix
        np.savetxt(self.pertMatFile, self.stateV)

        # Write the perturbation command file
        fid = open(self.pertCmdFile, "w")

        # M1M3 surface print
        if (self.M1M3surf is not None):
            # M1M3surf already converted into ZCRS
            writeM1M3zres(self.M1M3surf, M1M3.bx, M1M3.by, M1M3.Ri, M1M3.R, M1M3.R3i,
                          M1M3.R3, self.znPert, self.M1M3zlist, self.resFile1,
                          self.resFile3, M1M3.nodeID, self.surfaceGridN)

            # Not sure it is a good idea or not to use the M1M3zList for M1 and M3
            # correction together. This makes the z-axis offset to be the same.
            # I do not think this is coorect. Check with Bo for this.
            # But this also depends the initial FEA model study for the fitting.
            # Check with Bo.
            zz = np.loadtxt(self.M1M3zlist)
            for ii in range(self.znPert):
                fid.write("izernike 0 %d %s\n" % (ii, zz[ii]*1e-3))
            for ii in range(self.znPert):
                fid.write("izernike 2 %d %s\n" % (ii, zz[ii]*1e-3))

            fid.write("surfacemap 0 %s 1\n" % os.path.abspath(self.resFile1))
            fid.write("surfacemap 2 %s 1\n" % os.path.abspath(self.resFile3))

            # The meaning of surfacelink? Do not find it in:
            # https://bitbucket.org/phosim/phosim_release/wiki/Physics%20Commands
            # Check this with Bo
            fid.write("surfacelink 2 0\n")

        # M2 surface print
        if (self.M2surf is not None):
            # M2surf already converted into ZCRS
            writeM2zres(self.M2surf, M2.bx, M2.by, M2.R, M2.Ri, self.znPert,
                        self.M2zlist, self.resFile2, self.surfaceGridN)
            zz = np.loadtxt(self.M2zlist)
            for ii in range(self.znPert):
                fid.write("izernike 1 %d %s\n" % (ii, zz[ii]*1e-3))

            fid.write("surfacemap 1 %s 1\n" % os.path.abspath(self.resFile2))

        # Camera lens correction in Zk. ComCam has no such data now.
        if (self.camRot is not None) and (self.inst == self.LSST):
            for ii in range(self.znPert):
                # Andy uses mm, same unit as Zemax
                fid.write("izernike 3 %d %s\n" % (ii, self.L1S1zer[ii]))
                fid.write("izernike 4 %d %s\n" % (ii, self.L1S2zer[ii]))
                fid.write("izernike 5 %d %s\n" % (ii, self.L2S1zer[ii]))
                fid.write("izernike 6 %d %s\n" % (ii, self.L2S2zer[ii]))
                fid.write("izernike 9 %d %s\n" % (ii, self.L3S1zer[ii]))
                fid.write("izernike 10 %d %s\n" % (ii, self.L3S2zer[ii]))

        fid.close()

    def getOPDAll(self, obsID, metr, numproc, znwcs, obscuration, opdoff=False, debugLevel=0):
        """

        Get the optical path difference (OPD) by PhoSim.

        Arguments:
            obsID {[int]} -- Observation ID used in PhoSim.
            metr {[aosMetric]} -- aosMetric object.
            numproc {[int]} -- Number of processor for parallel calculation.
            znwcs {[int]} -- Number of terms of annular Zk (z1-z22 by default).
            obscuration {[float]} -- Obscuration of annular Zernike polynomial.

        Keyword Arguments:
            opdoff {bool} -- Calculate OPD or not. (default: {False})
            debugLevel {int} -- Debug level. The higher value gives more information.
                                (default: {0})
        """

        # The boundary of OPD image is dangerous for the calculation. Check with Bo.

        # Calculate OPD by PhoSim if necessary
        if (not opdoff):

            # Write the OPD instrument file for PhoSim use
            self.__writeOPDinst(obsID, metr)

            # Write the OPD command file for PhoSim use
            self.__writeOPDcmd(metr)

            # Source and destination file paths
            srcFile = os.path.join(self.phosimDir, "output", "opd_%d.fits.gz" % obsID)
            dstFile = os.path.join(self.imageDir, "iter%d" % self.iIter,
                                   "sim%d_iter%d_opd.fits.gz" % (self.iSim, self.iIter))

            # Collect the arguments needed for PhoSim use
            argList = []
            argList.append((self.OPD_inst, self.OPD_cmd, self.inst, self.eimage,
                            self.OPD_log, self.phosimDir, self.zTrueFile,
                            metr.nFieldp4, znwcs, obscuration, self.opdx, self.opdy,
                            srcFile, dstFile, self.nOPDw, numproc, debugLevel))

            # Calculate the OPD by parallel calculation
            runOPD(argList[0])

    def getOPDAllfromBase(self, baserun, nField):
        """

        Hard link the optical path difference (OPD) related files to avoid the repeated calculation.

        Arguments:
            baserun {[int]} -- Source simulation number (basement run).
            nField {[int]} -- Number of field points on focal plane. It can be 31 or 35 (31 + 4 WFS points).
        """

        # File list to do the hard link
        hardLinkFileList = [self.OPD_inst, self.OPD_log, self.zTrueFile, self.OPD_cmd]

        # Collect the file related to OPD
        iterFolder = "iter%d" % self.iIter
        for ii in range(self.nOPDw):
            for iField in range(nField):

                if (self.nOPDw == 1):
                    opdFile = os.path.join(self.imageDir, iterFolder,
                                           "sim%d_iter%d_opd%d.fits" % (self.iSim, self.iIter, iField))
                else:
                    opdFile = os.path.join(self.imageDir, iterFolder,
                                    "sim%d_iter%d_opd%d_w%d.fits" % (self.iSim, self.iIter, iField, ii))

                hardLinkFileList.append(opdFile)

        # Hard link the file to avoid the repeated calculation
        for aFile in hardLinkFileList:
            hardLinkFile(aFile, baserun, self.iSim)

    def __writeOPDinst(self, obsID, metr):
        """

        Write the instrument file of optical path difference (OPD) used in PhoSim.

        Arguments:
            obsID {[int]} -- Observation ID.
            metr {[aosMetric]} -- aosMetric object.
        """

        # Intrument condition used for PhoSim
        content = ""
        content += "Opsim_filter %d\n" % phosimFilterID[self.band]
        content += "Opsim_obshistid %d\n" % obsID
        content += "SIM_VISTIME 15.0\n"
        content += "SIM_NSNAP 1\n"

        # Write the aggregated degree of freedom of telescope
        writeToFile(self.OPD_inst, content=content, sourceFile=self.pertFile, mode="w")
       
        # Append the file with the OPD information contains the related field X, Y
        # The output of PhoSim simulation is the OPD image for inspected field points
        content = ""
        for irun in range(self.nOPDw):
            # Need to update "metr.nFieldp4" here. This is bad and confusing.
            for ii in range(metr.nFieldp4):

                if (self.nOPDw == 1):
                    opdWavelengthInNm = self.effwave*1e3
                else:
                    opdWavelengthInNm = self.GQwave[self.band][irun]*1e3

                content += "opd %2d\t%9.6f\t%9.6f %5.1f\n" % (irun*metr.nFieldp4 + ii, metr.fieldX[ii], 
                                                              metr.fieldY[ii], opdWavelengthInNm)

        # Append the file content
        writeToFile(self.OPD_inst, content=content, mode="a")

    def __writeOPDcmd(self, metr):
        """

        Write the command file of optical path difference (OPD) used in PhoSim.

        Arguments:
            metr {[aosMetric]} -- aosMetric object.
        """

        # Command condition used for PhoSim
        content = ""
        content += "backgroundmode 0\n"
        content += "raydensity 0.0\n"
        content += "perturbationmode 1\n"

        # Write the mirror surface print correction along the z-axis and camera lens distortion
        writeToFile(self.OPD_cmd, content=content, sourceFile=self.pertCmdFile, mode="w")

    def getWFSAll(self, obsID, timeIter, wfs, metr, numproc, debugLevel):

        self.__writeWFSinst(obsID, timeIter, wfs, metr)
        self.__writeWFScmd(wfs)
        argList = []
        for irun in range(wfs.nRun):
            argList.append((self.WFS_inst[irun], self.WFS_cmd, self.inst,
                                self.eimage, self.WFS_log[irun],
                                self.phosimDir, numproc, debugLevel))
        # test, pdb cannot go into the subprocess
        # runWFS1side(argList[0])

        pool = multiprocessing.Pool(numproc)
        pool.map(runWFS1side, argList)
        pool.close()
        pool.join()

        plt.figure(figsize=(10, 5))
        for i in range(metr.nFieldp4 - wfs.nWFS, metr.nFieldp4):
            chipStr, px, py = self.fieldXY2Chip(
                metr.fieldXp[i], metr.fieldYp[i], debugLevel)
            if wfs.nRun == 1: # phosim generates C0 & C1 already
                for ioffset in [0, 1]:
                    for iexp in [0, 1]:
                        src = glob.glob('%s/output/*%s_f%d_%s*%s*E00%d.fit*' %
                                    (self.phosimDir, obsID,
                                        phosimFilterID[self.band],
                                    chipStr, wfs.halfChip[ioffset], iexp))
                        if '.gz' in src[0]:
                            runProgram('gunzip -f %s' % src[0])
                        elif 'gz' in src[-1]:
                            runProgram('gunzip -f %s' % src[-1])
                        chipFile = src[0].replace('.gz', '')
                        runProgram('mv -f %s %s/iter%d' %
                                (chipFile, self.imageDir, self.iIter))
            else: # need to pick up two sets of fits.gz with diff phosim ID
                for ioffset in [0, 1]:
                    src = glob.glob('%s/output/*%s_f%d_%s*E000.fit*' %
                                    (self.phosimDir, obsID + ioffset,
                                        phosimFilterID[self.band],
                                    chipStr))
                    if '.gz' in src[0]:
                        runProgram('gunzip -f %s' % src[0])
                    elif 'gz' in src[-1]:
                        runProgram('gunzip -f %s' % src[-1])
                    chipFile = src[0].replace('.gz', '')
                    targetFile = os.path.split(chipFile.replace(
                        'E000', '%s_E000' % wfs.halfChip[ioffset]))[1]
                    runProgram('mv -f %s %s/iter%d/%s' %
                            (chipFile, self.imageDir, self.iIter,
                                targetFile))

    def __writeWFSinst(self, obsID, timeIter, wfs, metr):
        for irun in range(wfs.nRun):
            fid = open(self.WFS_inst[irun], 'w')
            fid.write('Opsim_filter %d\n\
Opsim_obshistid %d\n\
mjd %.10f\n\
SIM_VISTIME 33.0\n\
SIM_NSNAP 2\n\
SIM_SEED %d\n\
Opsim_rawseeing -1\n' % (phosimFilterID[self.band],
                             obsID + irun, timeIter.mjd,
                             obsID % 10000 + 4))
            fpert = open(self.pertFile, 'r')
            hasCamPiston = False #pertFile already includes move 10
            for line in fpert:
                if wfs.nRun > 1 and line.split()[:2] == ['move', '10']:
                    # move command follow Zemax coordinate system.
                    fid.write('move 10 %9.4f\n' %
                            (float(line.split()[2]) - wfs.offset[irun] * 1e3))
                    hasCamPiston = True
                else:
                    fid.write(line)
            if wfs.nRun > 1 and (not hasCamPiston):
                fid.write('move 10 %9.4f\n' % (-wfs.offset[irun] * 1e3))

            fpert.close()

            if self.inst[:4] == 'lsst':
                fid.write('SIM_CAMCONFIG 2\n')
            elif self.inst[:6] == 'comcam':
                fid.write('SIM_CAMCONFIG 1\n')

            ii = 0
            for i in range(metr.nFieldp4 - wfs.nWFS, metr.nFieldp4):
                if self.inst[:4] == 'lsst':
                    if i % 2 == 1:  # field 31, 33, R44 and R00
                        fid.write('object %2d\t%9.6f\t%9.6f %9.6f \
../sky/%s 0.0 0.0 0.0 0.0 0.0 0.0 star 0.0  none  none\n' % (
                            ii, metr.fieldXp[i] + 0.020, metr.fieldYp[i],
                            self.cwfsMag, self.sedfile))
                        ii += 1
                        fid.write('object %2d\t%9.6f\t%9.6f %9.6f \
../sky/%s 0.0 0.0 0.0 0.0 0.0 0.0 star 0.0  none  none\n' % (
                            ii, metr.fieldXp[i] - 0.020, metr.fieldYp[i],
                            self.cwfsMag, self.sedfile))
                        ii += 1
                    else:
                        fid.write('object %2d\t%9.6f\t%9.6f %9.6f \
../sky/%s 0.0 0.0 0.0 0.0 0.0 0.0 star 0.0  none  none\n' % (
                            ii, metr.fieldXp[i], metr.fieldYp[i] + 0.020,
                            self.cwfsMag, self.sedfile))
                        ii += 1
                        fid.write('object %2d\t%9.6f\t%9.6f %9.6f \
../sky/%s 0.0 0.0 0.0 0.0 0.0 0.0 star 0.0  none  none\n' % (
                            ii, metr.fieldXp[i], metr.fieldYp[i] - 0.020,
                            self.cwfsMag, self.sedfile))
                        ii += 1
                elif self.inst[:6] == 'comcam':
                    fid.write('object %2d\t%9.6f\t%9.6f %9.6f \
../sky/%s 0.0 0.0 0.0 0.0 0.0 0.0 star 0.0  none  none\n' % (
                        ii, metr.fieldXp[i], metr.fieldYp[i],
                        self.cwfsMag, self.sedfile))
                    ii += 1
            fid.close()
            fpert.close()

    def __writeWFScmd(self, wfs):
        fid = open(self.WFS_cmd, 'w')
        fid.write('backgroundmode 0\n\
raydensity 0.0\n\
perturbationmode 1\n\
trackingmode 0\n\
cleartracking\n\
clearclouds\n\
lascatprob 0.0\n\
contaminationmode 0\n\
diffractionmode 1\n\
straylight 0\n\
detectormode 0\n')
# airrefraction 0\n\
# coatingmode 0\n\ #this clears filter coating too
        fpert = open(self.pertCmdFile, 'r')
        fid.write(fpert.read())
        fpert.close()
        fid.close()

    def fieldXY2Chip(self, fieldX, fieldY, debugLevel):
        ruler = getChipBoundary(
            '%s/data/lsst/focalplanelayout.txt' % (self.phosimDir))

        # r for raft, c for chip, p for pixel
        rx, cx, px = fieldAgainstRuler(ruler, fieldX, 4000)
        ry, cy, py = fieldAgainstRuler(ruler, fieldY, 4072)

        if debugLevel >= 3:
            print('ruler:\n')
            print(ruler)
            print(len(ruler))

        return 'R%d%d_S%d%d' % (rx, ry, cx, cy), px, py

def getChipBoundary(fplayoutFile):

    mydict = dict()
    f = open(fplayoutFile)
    for line in f:
        line = line.strip()
        if (line.startswith('R')):
            mydict[line.split()[0]] = [float(line.split()[1]),
                                       float(line.split()[2])]

    f.close()
    ruler = sorted(set([x[0] for x in mydict.values()]))
    return ruler


def fieldAgainstRuler(ruler, field, chipPixel):

    field = field * 180000  # degree to micron
    p2 = (ruler >= field)
    if (np.count_nonzero(p2) == 0):  # too large to be in range
        p = len(ruler) - 1  # p starts from 0
    elif (p2[0]):
        p = 0
    else:
        p1 = p2.argmax() - 1
        p2 = p2.argmax()
        if (ruler[p2] - field) < (field - ruler[p1]):
            p = p2
        else:
            p = p1

    pixel = (field - ruler[p]) / 10  # 10 for 10micron pixels
    pixel += chipPixel / 2

    return np.floor(p / 3), p % 3, int(pixel)

def runOPD(argList):
    OPD_inst = argList[0]
    OPD_cmd = argList[1]
    inst = argList[2]
    eimage = argList[3]
    OPD_log = argList[4]
    phosimDir = argList[5]
    zTrueFile = argList[6]
    nFieldp4 = argList[7]
    znwcs = argList[8]
    obscuration = argList[9]
    opdx = argList[10]
    opdy = argList[11]
    srcFile = argList[12]
    dstFile = argList[13]
    nOPDw = argList[14]
    nthread = argList[15]
    debugLevel = argList[16]

    if debugLevel >= 3:
        runProgram('head %s' % OPD_inst)
        runProgram('head %s' % OPD_cmd)

    myargs = '%s -c %s -i %s -e %d -t %d > %s 2>&1' % (
        OPD_inst, OPD_cmd, inst, eimage, nthread,
        OPD_log)
    if debugLevel >= 2:
        print('*******Runnnig PHOSIM with following parameters*******')
        print('Check the log file below for progress')
        print('%s' % myargs)
        runProgram('date')
    runProgram('python %s/phosim.py' %
                   phosimDir, argstring=myargs)
    if debugLevel >= 2:
        print('DONE RUNNING PHOSIM FOR OPD: %s' % OPD_inst)
        runProgram('date')
    if os.path.isfile(zTrueFile):
        os.remove(zTrueFile)
    fz = open(zTrueFile, 'ab')
    for i in range(nFieldp4 * nOPDw):
        src = srcFile.replace('.fits.gz', '_%d.fits.gz' % i)
        if nOPDw == 1:
            dst = dstFile.replace('opd', 'opd%d' % i)
        else:
            dst = dstFile.replace('opd', 'opd%d_w%d' % (i%nFieldp4, int(i/nFieldp4)))
        shutil.move(src, dst)
        runProgram('gunzip -f %s' % dst)
        opdFile = dst.replace('.gz', '')
        IHDU = fits.open(opdFile)
        opd = IHDU[0].data  # Phosim OPD unit: um
        IHDU.close()
        idx = (opd != 0)
        Z = ZernikeAnnularFit(opd[idx], opdx[idx], opdy[idx],
                                  znwcs, obscuration)
        np.savetxt(fz, Z.reshape(1, -1), delimiter=' ')

    fz.close()

    if debugLevel >= 3:
        print(opdx)
        print(opdy)
        print(znwcs)
        print(obscuration)

def runWFS1side(argList):
    WFS_inst = argList[0]
    WFS_cmd = argList[1]
    inst = argList[2]
    eimage = argList[3]
    WFS_log = argList[4]
    phosimDir = argList[5]
    numproc = argList[6]
    debugLevel = argList[7]

    myargs = '%s -c %s -i %s -p %d -e %d > %s 2>&1' % (
        WFS_inst, WFS_cmd, inst, numproc, eimage,
        WFS_log)
    if debugLevel >= 2:
        print('********Runnnig PHOSIM with following parameters\
        ********')
        print('Check the log file below for progress')
        print('%s' % myargs)

    runProgram('python %s/phosim.py' %
               phosimDir, argstring=myargs)


def writeM1M3zres(surf, x, y, Ri, R, R3i, R3, n, zlist, resFile1, resFile3,
                      nodeID, surfaceGridN):

    zc = ZernikeFit(surf, x / R, y / R, n)
    res = surf - ZernikeEval(zc, x / R, y / R)
    np.savetxt(zlist, zc)

    idx1 = nodeID == 1
    idx3 = nodeID == 3

    # so far x and y are in meter, res is in micron
    # zemax wants everything in mm
    gridSamp(x[idx1] * 1e3, y[idx1] * 1e3, res[idx1] * 1e-3,
                 Ri * 1e3, R * 1e3, resFile1,
                 surfaceGridN, surfaceGridN, 1)
    gridSamp(x[idx3] * 1e3, y[idx3] * 1e3, res[idx3] * 1e-3,
                 R3i * 1e3, R3 * 1e3, resFile3,
                 surfaceGridN, surfaceGridN, 1)


def writeM2zres(surf, x, y, R, Ri, n, zlist, resFile2, surfaceGridN):

    # Fit the mirror surface
    zc = ZernikeFit(surf, x / R, y / R, n)

    res = surf - ZernikeEval(zc, x / R, y / R)
    np.savetxt(zlist, zc)

    # so far x and y are in meter, res is in micron
    # zemax wants everything in mm
    gridSamp(x * 1e3, y * 1e3, res * 1e-3, Ri * 1e3, R * 1e3, resFile2,
                 surfaceGridN, surfaceGridN, 1)


def gridSamp(xf, yf, zf, innerR, outerR, resFile, nx, ny, plots):

    Ff = Rbf(xf, yf, zf)
    #do not want to cover the edge? change 4->2 on both lines
    NUM_X_PIXELS = nx + 4  #alway extend 2 points on each side
    NUM_Y_PIXELS = ny + 4

    extFx = (NUM_X_PIXELS - 1) / (nx - 1) #this is spatial extension factor
    extFy = (NUM_Y_PIXELS - 1) / (ny - 1)
    extFr = np.sqrt(extFx * extFy)

    delx =  outerR*2 *extFx / (NUM_X_PIXELS - 1)
    dely =  outerR*2 *extFy / (NUM_Y_PIXELS - 1)

    minx = -0.5*(NUM_X_PIXELS-1)*delx
    miny = -0.5*(NUM_Y_PIXELS-1)*dely
    epsilon = .0001 * min(delx, dely)
    zp = np.zeros((NUM_X_PIXELS, NUM_Y_PIXELS))

    outid = open(resFile, 'w');
    # Write four numbers for the header line
    outid.write('%d %d %.9E %.9E\n' % (NUM_X_PIXELS, NUM_Y_PIXELS, delx, dely))

    #  Write the rows and columns
    for j in range(1, NUM_X_PIXELS + 1):
        for i in range(1, NUM_Y_PIXELS + 1):
            x =  minx + (i - 1) * delx
            y =  miny + (j - 1) * dely
            y = -y  # invert top to bottom, because Zemax reads (-x,-y) first

            # compute the sag */
            r = np.sqrt(x*x+y*y)

            if (r<innerR/extFr or r>outerR*extFr):
                z=0
                dx=0
                dy=0
                dxdy=0
            else:
                z=Ff(x,y)
                tem1=Ff((x+epsilon),y)
                tem2=Ff((x-epsilon),y)
                dx = (tem1 - tem2)/(2.0*epsilon)

                # compute dz/dy */
                tem1=Ff(x,(y+epsilon))
                tem2=Ff(x,(y-epsilon))
                dy = (tem1 - tem2)/(2.0*epsilon)

                # compute d2z/dxdy */
                tem1=Ff((x+epsilon),(y+epsilon))
                tem2=Ff((x-epsilon),(y+epsilon))
                tem3 = (tem1 - tem2)/(2.0*epsilon)
                tem1=Ff((x+epsilon),(y-epsilon))
                tem2=Ff((x-epsilon),(y-epsilon))
                tem4 = (tem1 - tem2)/(2.0*epsilon)
                dxdy = (tem3 - tem4)/(2.0*epsilon)

            zp[NUM_X_PIXELS+1-j-1, i-1]=z
            outid.write('%.9E %.9E %.9E %.9E\n'% (z, dx, dy, dxdy))

    outid.close()

    if plots:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        # the input data to gridSamp.m is in mm (zemax default)
        sc = ax[1].scatter(xf, yf, s=25, c=zf*1e6, marker='.', edgecolor='none')
        ax[1].axis('equal')
        ax[1].set_title('Surface map on FEA grid (nm)')
        ax[1].set_xlim([-outerR, outerR])
        ax[1].set_ylim([-outerR, outerR])
        ax[1].set_xlabel('x (mm)')

        xx = np.arange(minx, -minx + delx, delx)
        yy = np.arange(miny, -miny + dely, dely)
        xp, yp = np.meshgrid(xx, yy)
        xp = xp.reshape((NUM_X_PIXELS*NUM_Y_PIXELS,1))
        yp = yp.reshape((NUM_X_PIXELS*NUM_Y_PIXELS,1))
        zp = zp.reshape((NUM_X_PIXELS*NUM_Y_PIXELS,1))
        sc = ax[0].scatter(xp, yp, s=25, c=zp*1e6, marker='.',
                             edgecolor='none')
        ax[0].axis('equal')
        ax[0].set_title('grid input to ZEMAX (nm)')
        ax[0].set_xlim([-outerR, outerR])
        ax[0].set_ylim([-outerR, outerR])
        ax[0].set_xlabel('x (mm)')
        ax[0].set_ylabel('y (mm)')

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.1, 0.03, 0.8])
        fig.colorbar(sc, cax=cbar_ax)

        plt.savefig(resFile.replace('.txt','.png'))

def getTelEffWave(wavelength, band):
    """

    Get the effective wavelength and number of needed optical path difference (OPD).

    Arguments:
        wavelength {[float]} -- Wavelength in um.
        band {[str]} -- Type of active filter ("u", "g", "r", "i", "z", "y").

    Returns:
        [float] -- Effective wavelength.
        [int] -- Number of OPD.
    """

    # Check the type of band
    if band not in ("u", "g", "r", "i", "z", "y"):
        raise ValueError("No '%s' band available." % band)

    # Get the effective wavelength and number of OPD
    if (wavelength == 0):
        effwave = aosTeleState.effwave[band]
        nOPDw = len(aosTeleState.GQwave[band])
    else:
        effwave = wavelength
        nOPDw = 1

    return effwave, nOPDw

if __name__ == "__main__":

    pass
