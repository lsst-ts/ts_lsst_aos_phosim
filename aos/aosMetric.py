#!/usr/bin/env python
##
# @authors: Bo Xin
# @       Large Synoptic Survey Telescope

import os
import sys
import multiprocessing

import numpy as np
import scipy.special as sp
from astropy.io import fits

from lsst.cwfs.tools import padArray
from lsst.cwfs.tools import extractArray
from lsst.cwfs.tools import ZernikeAnnularFit
from lsst.cwfs.tools import ZernikeAnnularEval
from lsst.cwfs.errors import nonSquareImageError

from aos.aosErrors import psfSamplingTooLowError
from aos.aosTeleState import aosTeleState


import matplotlib.pyplot as plt

from aos.aosUtility import getInstName, hardLinkFile, feval


class aosMetric(object):

    # Instrument name
    LSST = "lsst"
    ComCam = "comcam"

    def __init__(self, aosDataDir, instName, pixelum=10, debugLevel=0):
        """
        
        Initiate the aosMetric object.
        
        Arguments:
            aosDataDir {[str]} -- Directory of metrology parameter file.
            inst {[str]} -- Instrument name (e.g. lsst or comcam10).
        
        Keyword Arguments:
            pixelum {int} -- Pixel to um. (default: {10})
            debugLevel {int} -- Debug level. The higher value gives more information.
                                (default: {0})
        """

        # Get Instrument name
        instName = getInstName(instName)[0]

        # Get the position and related weighting defined on Gaussian quadrature plane
        self.w, self.nField, self.nFieldp4, self.fieldX, self.fieldY = self.__getFieldCoor(instName)

        # Check the weighting ratio or not
        if (debugLevel >= 3):
            print(self.w.shape)
            print(self.w)

        # Full width at half maximum (FWHM) related files
        if (instName == self.LSST):
            self.fwhmModelFileBase = os.path.join(aosDataDir, "fwhmModel", "fwhm_vs_z_500nm")

        # Field point for the point spread function (PSF)
        self.fieldXp = self.fieldX.copy()
        self.fieldYp = self.fieldY.copy()

        # Add the offset (0.004 degree) for LSST wave front sensor (WFS)
        if (instName == self.LSST) and (pixelum == 10):
            self.fieldXp[19] += 0.004
            self.fieldXp[22] -= 0.004

        # Grt the alpha value of normalized point source sensitivity (PSSN)
        # PSSN ~ 1 - a * delta^2 
        # Eq (7.1) in Normalized Point Source Sensitivity for LSST (document-17242)
        # It is noted that there are 19 terms of alpha value here for z3-z22 to use.
        alphaData = np.loadtxt(os.path.join(aosDataDir, "pssn_alpha.txt"))
        self.pssnAlpha = alphaData[:, 0]

        # Parameters
        self.GQFWHMeff = None

        # File paths are assigned in aosTeleState class. Need to change this.
        self.PSSNFile = None
        self.elliFile = None

    def __getFieldCoor(self, instName):
        """
        
        Get the position and related weighting defined on Gaussian quadrature plane.
        
        Arguments:
            instName {[str]} -- Instrument name.

        Returns:
            [ndarray] -- Weighting of points on Gaussian quadrature plane.
            [int] -- Number of points on Gaussian quadrature plane.
            [int] -- Number of points with wavefront sensor (WFS) on Gaussian quadrature plane.
            [ndarray] -- Point x coordinate.
            [ndarray] -- Point y coordinate.
        """

        # Define the Gaussian quadrature plane
        if (instName == self.LSST):

            # The distance of point xi (used in Gaussian quadrature plane) to the origin
            # This value is in [-1.75, 1.75]
            armLen = [0.379, 0.841, 1.237, 1.535, 1.708]

            # Weighting of point xi (used in Gaussian quadrature plane) for each ring
            armW = [0.2369, 0.4786, 0.5689, 0.4786, 0.2369]

            # Number of ring on Gaussian quadrature plane
            nRing = len(armLen)

            # Number of points on each ring
            nArm = 6

            # Field x, y for 4 WFS
            fieldWFSx = [1.176, -1.176, -1.176, 1.176]
            fieldWFSy = [1.176, 1.176, -1.176, -1.176]

            # Get the weighting for all field points (31 for lsst camera)
            # Consider the first element is center (0)
            # Do not understand to put the centeral point's wrighting as 0. Check with Bo. 
            w = np.concatenate([np.zeros(1), np.kron(armW, np.ones(nArm))])
            w = w/sum(w)

            # Number of field points with the adding of center point
            nField = nArm*nRing + 1

            # Number of field points with the consideration of wavefront sensor
            nFieldWfs = nField + len(fieldWFSx)

            # Generate the fields point x, y coordinates
            pointAngle = np.arange(nArm) * (2*np.pi)/nArm
            fieldX = np.concatenate([np.zeros(1), np.kron(armLen, np.cos(pointAngle)), fieldWFSx])
            fieldY = np.concatenate([np.zeros(1), np.kron(armLen, np.sin(pointAngle)), fieldWFSy])

        # It looks like the weighting and position of Gaussian quardure plane here are just the 
        # 1st order approximation. Check this with Bo. Do we need the further study?
        elif (instName == self.ComCam):
            
            # ComCam is the cetral raft of LSST cam, which is composed of 3 x 3 CCDs.
            nRow = 3
            nCol = 3
            
            # Number of field points
            nField = nRow*nCol

            # Number of field points with the consideration of wavefront sensor
            nFieldWfs = nField

            # Get the weighting for all field points (9 for comcam)
            w = np.ones(nField)
            w = w / np.sum(w)

            # Distance to raft center in degree along x/y direction and the related relative position
            sensorD = 0.2347
            coorComcam = sensorD * np.array([-1, 0 ,1])

            # Generate the fields point x, y coordinates
            fieldX = np.kron(coorComcam, np.ones(nRow))
            fieldY = np.kron(np.ones(nCol), coorComcam)

        return w, nField, nFieldWfs, fieldX, fieldY

    def getPSSNandMorefromBase(self, baserun, iSim):
        """
        
        Hard link the PSSN file to avoid the repeated calculation.
        
        Arguments:
            baserun {[int]} -- Source simulation number (basement run).
            iSim {[int]} -- Simulation number.
        """

        # Hard link the file to avoid the repeated calculation
        hardLinkFile(self.PSSNFile, baserun, iSim)

        # Read the effective FWHM of Gaussian quadrature plane
        # This is needed for the shiftGear in the control algorithm
        self.setGQFWHMeff(loadFilePath=self.PSSNFile)

    def getEllipticityfromBase(self, baserun, iSim):
        """
        
        Hard link the ellipticity file to avoid the repeated calculation.
        
        Arguments:
            baserun {[int]} -- Source simulation number (basement run).
            iSim {[int]} -- Simulation number.
        """

        # Hard link the file to avoid the repeated calculation
        hardLinkFile(self.elliFile, baserun, iSim)

    def setGQFWHMeff(self, value=None, loadFilePath=None):
        """
        
        Set the value of effective full width at half maximum (FWHM) on Gaussian quadrature plane.
        
        Keyword Arguments:
            value {[ndarray]} -- Effective FWHM data. (default: {None})
            loadFilePath {[str]} -- File contains the FWHM data. (default: {None})
        
        Raises:
            ValueError -- One of inputs (value, loadFilePath) should be None.
        """

        if (value is not None) and (loadFilePath is not None):
            raise ValueError("One of inputs (value, loadFilePath) should be None.")

        if (value is not None):
            self.GQFWHMeff = value

        if (loadFilePath is not None):
            data = np.loadtxt(loadFilePath)
            self.GQFWHMeff = data[1, -1]

    def getEllipticity(self, state, ellioff=False, outElliFile=None, debugLevel=0):
        """
        
        Calculate the ellipticity for all field points.
        
        Arguments:
            state {[aosTeleState]} -- aosTeleState object.
        
        Keyword Arguments:
            ellioff {[bool]} -- Calculate the ellipticity or not. (default: {False})
            outElliFile {[str]} -- Output file path. (default: {None})
            debugLevel {int} -- Debug level. The higher value gives more information. (default: {0})
        """

        # Redirect the path of outPssnFile file if necessary
        if (outElliFile is None):
            outElliFile = self.elliFile

        # Calculate the ellipticity
        if (not ellioff):

            # Calculate the effective ellipticity
            elli = self.__analyzeOPD(state.imageDir, state.iSim, state.iIter, state.nOPDw, 
                                     state.wavelength, state.band, runEllipticity, 
                                     funcArgs=(state.opdx, state.opdy, debugLevel))

            # Show the calculated ellipticity or not
            for ii in range(self.nField):
                if (debugLevel >= 2):
                    print("--- Field #%d, elli = %7.4f." % (ii, elli[ii]))

            # Calculate the effective ellipticity on Gaussain quardure plane
            GQelli = np.sum(self.w*elli)

            # Use the list of effective GQ values for the need of np.concatenate
            a1 = np.concatenate((elli, [GQelli]))
            np.savetxt(outElliFile, a1)

            # Show the effective ellipticity on Gaussain quardure plane
            if (debugLevel >= 2):
                print(GQelli)

    def getPSSNandMore(self, state, pssnoff=False, outPssnFile=None, debugLevel=0):
        """
        
        Calculate the normalized point source sensitivity (PSSN), full width at half maximum (FWHM), and 
        loss of limiting depth (dm5) for all field points.
        
        Arguments:
            state {[aosTeleState]} -- aosTeleState object.
        
        Keyword Arguments:
            pssnoff {[bool]} -- Calculate the PSSN, FWHM, and dm5 or not. (default: {False})
            outPssnFile {[str]} -- Output file path. (default: {None})
            debugLevel {int} -- Debug level. The higher value gives more information. (default: {0})
        """

        # Redirect the path of outPssnFile file if necessary
        if (outPssnFile is None):
            outPssnFile = self.PSSNFile

        # Calculate the PSSN
        if (not pssnoff):

            # Calculate the effective PSSN
            PSSN = self.__analyzeOPD(state.imageDir, state.iSim, state.iIter, state.nOPDw, 
                                     state.wavelength, state.band, runPSSN, 
                                     funcArgs=(state.opdx, state.opdy, debugLevel))

            # Calculate the effective FWHM
            # FWHMeff_sys = FWHMeff_atm * sqrt(1/PSSN - 1). FWHMeff_atm = 0.6 arcsec. 
            # Another correction factor (eta = 1.086) is used to account for the difference between the 
            # simple RSS and the more proper convolution.
            # Follow page 7 (section 7.2 FWHM) in document-17242 for more information.
            eta = 1.086
            FWHMatm = 0.6
            FWHMeff = eta*FWHMatm*np.sqrt(1/PSSN - 1)

            # Calculate dm5 (the loss of limiting depth)
            # Check eq. (4.1) in page 4 in document-17242 for more information.
            dm5 = -1.25 * np.log10(PSSN)

            # Show the calculated PSSN and effective FWHM or not
            if (debugLevel >= 2):
                for ii in range(self.nField):
                    print("--- Field #%d, PSSN = %7.4f, FWHMeff = %5.0f mas." % (ii, PSSN[ii], FWHMeff[ii] * 1e3))

            # Calculate the effective PSSN, FWHM, and dm5 on Gaussain quardure plane
            GQPSSN = np.sum(self.w*PSSN)
            GQdm5 = np.sum(self.w*dm5)
            self.GQFWHMeff = np.sum(self.w*FWHMeff)

            # Use the list of effective GQ values for the need of np.concatenate
            allPssn = np.concatenate((PSSN, [GQPSSN]))
            allFwhm = np.concatenate((FWHMeff, [self.GQFWHMeff]))
            allDm5 = np.concatenate((dm5, [GQdm5]))

            # Write the data into the file
            np.savetxt(outPssnFile, np.vstack((allPssn, allFwhm, allDm5)))

            # Show the effective PSSN on Gaussain quardure plane
            if (debugLevel >= 2):
                print(GQPSSN)

        else:
            # Read the effective FWHM of Gaussian quadrature plane
            # This is needed for the shiftGear in the control algorithm
            self.setGQFWHMeff(loadFilePath=outPssnFile)

    def __analyzeOPD(self, imageDir, iSim, iIter, nOPDw, wavelength, band, func, funcArgs=()):
        """
        
        Analyze the optical path difference (OPD) based on the specified function.
        
        Arguments:
            imageDir {[str]} -- Directory to OPD images.
            iSim {[int]} -- Simulation number.
            iIter {[int]} -- Iteration number.
            nOPDw {[int]} -- Number of weighting ratio of specific band on Gaussian quardure.
            wavelength {[float]} -- Monochromatic light wavelength.
            band {[str]} -- Active filter band.
            func {[obj]} -- Function object to analyze the OPD.
        
        Keyword Arguments:
            funcArgs {tuple} -- Arguments needed for func to use. (default: {()})
        
        Returns:
            [ndarray] -- Result by the function on Gaussian quardure.
        """
        
        outputW = np.zeros((self.nField, nOPDw))
        for ii in range(self.nField):
            for irun in range(nOPDw):

                # Decide the wavelength in um
                if (nOPDw == 1):
                    wlum = wavelength
                else:
                    wlum = aosTeleState.GQwave[band][irun]
                
                # Input opd fits file
                if (nOPDw == 1):
                    inputFile = os.path.join(imageDir, "iter%d" % iIter, 
                                "sim%d_iter%d_opd%d.fits.gz" % (iSim, iIter, ii))
                
                else:
                    inputFile = os.path.join(imageDir, "iter%d" % iIter, 
                                "sim%d_iter%d_opd%d_w%d.fits.gz" % (iSim, iIter, ii, irun))

                # Arguments to do the calculation
                # Use the list to make the args to be iterable for feval() to use.
                args = [([inputFile], wlum, ) + funcArgs]

                # Calculate the value based on the specified function
                outputW[ii, irun] = feval(func, vars=args)

        # Repeat the weighting ratio of certain band by nField times on the axis-0 (row) 
        wt = np.tile(np.array(aosTeleState.GQwt[band]), (self.nField, 1))

        # Calculate the effective output on the Gaussian quardure plane 
        # Int f(x)dx in [-1, 1] = sum(wi * f(xi)) for all i 
        output = np.sum(wt*outputW, axis=1)

        return output

def runPSSN(argList):
    """
    
    Calculate the normalized point source sensitivity (PSSN) based on the optical path 
    difference (OPD) map.
    
    Arguments:
        argList {[tuple]} -- Arguments to do the calculation.
    
    Returns:
        [ndarray] -- Calculated PSSN.
    """

    # List of parameters
    inputFile = argList[0]
    wavelength = argList[1]
    opdx = argList[2]
    opdy = argList[3]
    debugLevel = argList[4]

    # Show the input file or not
    if (debugLevel >=2):
        print("runPSSN: %s." % inputFile)

    # Remove the affection of piston (z1), x-tilt (z2), and y-tilt (z3) from OPD map.
    opd = rmPTTfromOPD(inputFile[0], opdx, opdy)

    # Calculate the normalized point source sensitivity (PSSN)
    pssn = calc_pssn(opd, wavelength, debugLevel=debugLevel)

    return pssn

def runEllipticity(argList):
    """
    
    Calculate the ellipticity based on the optical path difference (OPD) map.
    
    Arguments:
        argList {[tuple]} -- Arguments to do the calculation.
    
    Returns:
        [ndarray] -- Calculated ellipticity.
    """

    # List of parameters
    inputFile = argList[0]
    wavelength = argList[1]
    opdx = argList[2]
    opdy = argList[3]    
    debugLevel = argList[4]

    # Show the input file or not
    if (debugLevel >= 2):
        print("runEllipticity: %s." % inputFile)

    # Remove the affection of piston (z1), x-tilt (z2), and y-tilt (z3) from OPD map.
    opd = rmPTTfromOPD(inputFile[0], opdx, opdy)

    # Calculate the ellipticity
    elli = psf2eAtmW(opd, wavelength, debugLevel=debugLevel)[0]

    return elli

def rmPTTfromOPD(inputFile, opdx, opdy):
    """
    
    Remove the afftection of piston (z1), x-tilt (z2), and y-tilt (z3) from the optical 
    map difference (OPD) map.
    
    Arguments:
        inputFile {[str]} -- OPD file path.
        opdx {[ndarray]} -- x positions of OPD map.
        opdy {[ndarray]} -- y positions of OPD map.
    
    Returns:
        [ndarray] -- OPD map after removing the affection of z1-z3.
    """

    # Before calc_pssn,
    # (1) Remove PTT (piston, x-tilt, y-tilt),
    # (2) Make sure outside of pupil are all zeros

    # Get the optical path difference (OPD) image data
    opd = fits.getdata(inputFile)

    # Find the index that the value of OPD is not 0
    idx = (opd != 0)

    # Do the annular Zernike fitting for the OPD map
    # Only fit the first three terms (z1-z3): piston, x-tilt, y-tilt
    # Do not understand why using the obscuration = 0 here. The OPD map does not look like this.
    # Check this obs=0 with Bo.
    # It might have no affection. Check with Bo.
    Z = ZernikeAnnularFit(opd[idx], opdx[idx], opdy[idx], 3, 0)
    
    # Make sure all valls after z4
    # Check to remove this
    Z[3:] = 0
    
    # Remove the PTT
    opd[idx] -= ZernikeAnnularEval(Z, opdx[idx], opdy[idx], 0)

    return opd

def calc_pssn(array, wlum, type='opd', D=8.36, r0inmRef=0.1382, zen=0,
              pmask=0, imagedelta=0, fno=1.2335, debugLevel=0):
    """
    array: the array that contains either opd or pdf
           opd need to be in microns
    wlum: wavelength in microns
    type: what is used to calculate pssn - either opd or psf
    psf doesn't matter, will be normalized anyway
    D: side length of OPD image in meter
    r0inmRef: fidicial atmosphere r0@500nm in meter, Konstantinos uses 0.20
    Now that we use vonK atmosphere, r0in=0.1382 -> fwhm=0.6"
    earlier, we used Kolm atmosphere, r0in=0.1679 -> fwhm=0.6"
    zen: telescope zenith angle

    The following are only needed when the input array is psf -
    pmask: pupil mask. when opd is used, it can be generated using opd image,
    we can put 0 or -1 or whatever here.
    when psf is used, this needs to be provided separately with same
    size as array.
    imagedelta and fno are only needed when psf is used. use 0,0 for opd

    THE INTERNAL RESOLUTION THAT FFTS OPERATE ON IS VERY IMPORTANT
    TO THE ACCUARCY OF PSSN.
    WHEN TYPE='OPD', NRESO=SIZE(ARRAY,1)
    WHEN TYPE='PSF', NRESO=SIZE(PMASK,1)
       for the psf option, we can not first convert psf back to opd then
       start over,
       because psf=|exp(-2*OPD)|^2. information has been lost in the | |^2.
       we need to go forward with psf->mtf,
       and take care of the coordinates properly.
    """

    if array.ndim == 3:
        array2D = array[0, :, :].squeeze()

    if type == 'opd':
        try:
            m = max(array2D.shape)
        except NameError:
            m = max(array.shape)
        k = 1
    else:
        m = max(pmask.shape)
        # pupil needs to be padded k times larger to get imagedelta
        k = fno * wlum / imagedelta

    mtfa = createMTFatm(D, m, k, wlum, zen, r0inmRef)

    if type == 'opd':
        try:
            iad = (array2D != 0)
        except NameError:
            iad = (array != 0)
    elif type == 'psf':
        mk = int(m + np.rint((m * (k - 1) + 1e-5) / 2) * 2)  # add even number
        iad = pmask  # padArray(pmask, m)

    # number of non-zero elements, used for normalization later
    # miad2 = np.count_nonzero(iad)

    # Perfect telescope
    opdt = np.zeros((m, m))
    psft = opd2psf(opdt, iad, wlum, imagedelta, 1, fno, debugLevel)
    otft = psf2otf(psft)  # OTF of perfect telescope
    otfa = otft * mtfa  # add atmosphere to perfect telescope
    psfa = otf2psf(otfa)
    pssa = np.sum(psfa**2)  # atmospheric PSS = 1/neff_atm

    # Error;
    if type == 'opd':
        if array.ndim == 2:
            ninst = 1
        else:
            ninst = array.shape[0]
        for i in range(ninst):
            if array.ndim == 2:
                array2D = array
            else:
                array2D = array[i, :, :].squeeze()
            psfei = opd2psf(array2D, iad, wlum, 0, 0, 0, debugLevel)
            if i == 0:
                psfe = psfei
            else:
                psfe += psfei
        psfe = psfe / ninst
    else:
        if array.shape[0] == mk:
            psfe = array
        elif array.shape[0] > mk:
            psfe = extractArray(array, mk)
        else:
            print('calc_pssn: image provided too small, %d < %d x %6.4f' % (
                array.shape[0], m, k))
            print('IQ is over-estimated !!!')
            psfe = padArray(array, mk)

        psfe = psfe / np.sum(psfe) * np.sum(psft)

    otfe = psf2otf(psfe)  # OTF of error
    otftot = otfe * mtfa  # add atmosphere to error
    psftot = otf2psf(otftot)
    pss = np.sum(psftot**2)  # atmospheric + error PSS

    pssn = pss / pssa  # normalized PSS
    if debugLevel >= 3:
        print('pssn = %10.8e/%10.8e = %6.4f' % (pss, pssa, pssn))

    return pssn


def createMTFatm(D, m, k, wlum, zen, r0inmRef):
    """
    m is the number of pixel we want to have to cover the length of D.
    If we want a k-times bigger array, we pad the mtf generated using k=1.
    """

    sfa = atmSF('vonK', D, m, wlum, zen, r0inmRef)
    mtfa = np.exp(-0.5 * sfa)

    N = int(m + np.rint((m * (k - 1) + 1e-5) / 2) * 2)  # add even number
    mtfa = padArray(mtfa, N)

    return mtfa


def atmSF(model, D, m, wlum, zen, r0inmRef):
    """
    create the atmosphere phase structure function
    model = 'Kolm'
             = 'vonK'
    """
    r0a = r0Wz(r0inmRef, zen, wlum)
    L0 = 30  # outer scale in meter, only used when model=vonK

    m0 = np.rint(0.5 * (m + 1) + 1e-5)
    aa = np.arange(1, m + 1)
    x, y = np.meshgrid(aa, aa)

    dr = D / (m - 1)  # frequency resolution in 1/rad
    r = dr * np.sqrt((x - m0)**2 + (y - m0)**2)

    if model == 'Kolm':
        sfa = 6.88 * (r / r0a)**(5 / 3)
    elif model == 'vonK':
        sfa_c = 2 * sp.gamma(11 / 6) / 2**(5 / 6) / np.pi**(8 / 3) *\
            (24 / 5 * sp.gamma(6 / 5))**(5 / 6) * (r0a / L0)**(-5 / 3)
        # modified bessel of 2nd/3rd kind
        sfa_k = sp.kv(5 / 6, (2 * np.pi / L0 * r))
        sfa = sfa_c * (2**(-1 / 6) * sp.gamma(5 / 6) -
                       (2 * np.pi / L0 * r)**(5 / 6) * sfa_k)

        # if we don't do below, everything will be nan after ifft2
        # midp = r.shape[0]/2+1
        # 1e-2 is to avoid x.49999 be rounded to x
        midp = int(np.rint(0.5 * (r.shape[0] - 1) + 1e-2))
        sfa[midp, midp] = 0  # at this single point, sfa_k=Inf, 0*Inf=Nan;

    return sfa


def r0Wz(r0inmRef, zen, wlum):
    zen = zen * np.pi / 180.  # telescope zenith angle, change here
    r0aref = r0inmRef * np.cos(zen)**0.6  # atmosphere reference r0
    r0a = r0aref * (wlum / 0.5)**1.2  # atmosphere r0, a function of wavelength
    return r0a


def psf2eAtmW(array, wlum, type='opd', D=8.36, pmask=0, r0inmRef=0.1382,
              sensorFactor=1,
              zen=0, imagedelta=0.2, fno=1.2335, debugLevel=0):
    """
    array: wavefront OPD in micron, or psf image
    unlike calc_pssn(), here imagedelta needs to be provided for type='opd'
        because the ellipticity calculation operates on psf.

    """
    k = fno * wlum / imagedelta
    if type == 'opd':
        m = array.shape[0] / sensorFactor
        psfe = opd2psf(array, 0, wlum, imagedelta,
                       sensorFactor, fno, debugLevel)
    else:
        m = max(pmask.shape)
        psfe = array
    mtfa = createMTFatm(D, m, k, wlum, zen, r0inmRef)

    otfe = psf2otf(psfe)  # OTF of error

    otf = otfe * mtfa
    psf = otf2psf(otf)

    if debugLevel >= 3:
        print('Below from the Gaussian weigting function on elli')
    e, q11, q22, q12 = psf2eW(psf, imagedelta, wlum, 'Gau', debugLevel)

    return e, q11, q22, q12


def psf2eW(psf, pixinum, wlum, atmModel, debugLevel=0):

    x, y = np.meshgrid(np.arange(1, psf.shape[0] + 1),
                       np.arange(1, psf.shape[1] + 1))
    xbar = np.sum(x * psf) / np.sum(psf)
    ybar = np.sum(y * psf) / np.sum(psf)

    r2 = (x - xbar)**2 + (y - ybar)**2

    fwhminarcsec = 0.6
    oversample = 1
    W = createAtm(atmModel, wlum, fwhminarcsec, r2, pixinum, oversample,
                  0, '', debugLevel)

    if debugLevel >= 3:
        print('xbar=%6.3f, ybar=%6.3f' % (xbar, ybar))
        # plot.plotImage(psf,'')
        # plot.plotImage(W,'')

    psf = psf * W  # apply weighting function

    Q11 = np.sum(((x - xbar)**2) * psf) / np.sum(psf)
    Q22 = np.sum(((y - ybar)**2) * psf) / np.sum(psf)
    Q12 = np.sum(((x - xbar) * (y - ybar)) * psf) / np.sum(psf)

    T = Q11 + Q22
    if T > 1e-20:
        e1 = (Q11 - Q22) / T
        e2 = 2 * Q12 / T

        e = np.sqrt(e1**2 + e2**2)
    else:
        e = 0

    return e, Q11, Q22, Q12


def createAtm(model, wlum, fwhminarcsec, gridsize, pixinum, oversample,
              cutOutput, outfile, debugLevel):
    """
    gridsize can be int or an array. When it is array, it is r2
    cutOutput only applies to Kolm and vonK
    """
    if isinstance(gridsize, (int)):
        nreso = gridsize * oversample
        nr = nreso / 2  # n for radius length
        aa = np.linspace(-nr + 0.5, nr - 0.5, nreso)
        x, y = np.meshgrid(aa)
        r2 = x * x + y * y
    else:
        r2 = gridsize

    if model[:4] == 'Kolm' or model[:4] == 'vonK':
        pass
    else:
        fwhminum = fwhminarcsec / 0.2 * 10
        if model == 'Gau':
            sig = fwhminum / 2 / np.sqrt(2 * np.log(2))  # in micron
            sig = sig / (pixinum / oversample)
            z = np.exp(-r2 / 2 / sig**2)
        elif model == '2Gau':
            # below is used to manually solve for sigma
            # let x = exp(-r^2/(2*alpha^2)), which results in 1/2*max
            # we want to get (1+.1)/2=0.55 from below
            # x=0.4673194304;printf('%20.10f\n'%x**.25*.1+x);
            sig = fwhminum / (2 * np.sqrt(-2 * np.log(0.4673194304)))
            sig = sig / (pixinum / oversample)  # in (oversampled) pixel
            z = np.exp(-r2 / 2 / sig**2) + 0.4 / 4 * np.exp(-r2 / 8 / sig**2)
        if debugLevel >= 3:
            print('sigma1=%6.4f arcsec' %
                  (sig * (pixinum / oversample) / 10 * 0.2))
    return z


def opd2psf(opd, pupil, wavelength, imagedelta, sensorFactor, fno, debugLevel):
    """
    wavefront OPD in micron
    imagedelta in micron, use 0 if pixel size is not specified
    wavelength in micron

    if pupil is a number, not an array, we will get pupil geometry from opd
    The following are not needed if imagedelta=0,
    sensorFactor, fno
    """

    opd[np.isnan(opd)] = 0
    try:
        if (pupil.shape == opd.shape):
            pass
        else:
            raise AttributeError
    except AttributeError:
        pupil = (opd != 0)

    if imagedelta != 0:
        try:
            if opd.shape[0] != opd.shape[1]:
                raise(nonSquareImageError)
        except nonSquareImageError:
            print('Error (opd2psf): Only square images are accepted.')
            print('image size = (%d, %d)' % (
                opd.shape[0], opd.shape[1]))
            sys.exit()

        k = fno * wavelength / imagedelta
        padding = k / sensorFactor
        try:
            if padding < 1:
                raise(psfSamplingTooLowError)
        except psfSamplingTooLowError:
            print('opd2psf: sampling too low, data inaccurate')
            print('imagedelta needs to be smaller than fno*wlum=%4.2f um' % (
                fno * wavelength))
            print('         so that the padding factor > 1')
            print('         otherwise we have to cut pupil to be < D')
            sys.exit()

        sensorSamples = opd.shape[0]
        # add even number for padding
        N = int(sensorSamples + \
            np.rint(((padding - 1) * sensorSamples + 1e-5) / 2) * 2)
        pupil = padArray(pupil, N)
        opd = padArray(opd, N)
        if debugLevel >= 3:
            print('padding=%8.6f' % padding)
    # if imagedelta = 0, we don't do any padding, and go with below
    z = pupil * np.exp(-2j * np.pi * opd / wavelength)
    z = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(z),
                                    s=z.shape))  # /sqrt(miad2/m^2)
    z = np.absolute(z**2)
    z = z / np.sum(z)

    if debugLevel >= 3:
        print('opd2psf(): imagedelta=%8.6f' % imagedelta, end='')
        if imagedelta == 0:
            print('0 means using OPD with padding as provided')
        else:
            print('')
        print('verify psf has been normalized: %4.1f' % np.sum(z))

    return z


def psf2otf(psf):
    otf = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(psf),
                                      s=psf.shape))
    return otf


def otf2psf(otf):
    psf = np.absolute(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(otf),
                                                   s=otf.shape)))
    return psf

