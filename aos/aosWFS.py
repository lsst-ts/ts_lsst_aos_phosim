#!/usr/bin/env python
##
# @authors: Bo Xin
# @       Large Synoptic Survey Telescope

import os
from glob import glob

import multiprocessing

import numpy as np
from astropy.io import fits
from scipy.ndimage.measurements import center_of_mass

from lsst.cwfs.algorithm import Algorithm
from lsst.cwfs.instrument import Instrument
from lsst.cwfs.image import Image, readFile

from aos.aosMetric import getInstName

import matplotlib.pylab as plt

import time

class aosWFS(object):

    # Instrument name
    LSST = "lsst"
    COMCAM = "comcam"

    # Type of defocus and chip name
    # C0: intra-focal WFS, C1: extra-focal WFS
    wfsName = ["intra", "extra"]
    halfChip = ["C0", "C1"]

    def __init__(self, cwfsDir, instruFile, algoFile, imgSizeinPix, band, wavelength, aosDataDir, debugLevel=0):
        """

        Initiate the aosWFS class.

        Arguments:
            cwfsDir {[str]} -- cwfs directory.
            instruFile {[str]} -- Instrument folder name.
            algoFile {[str]} -- Algorithm to solve the TIE.
            imgSizeinPix {[int]} -- Pix size in one dimension.
            band {[str]} -- Active filter ("u", "g", "r", "i", "z", "y").
            wavelength {[float]} -- Wavelength in um.
            aosDataDir {[str]} -- AOS data directory.

        Keyword Arguments:
            debugLevel {int} -- Debug level. The higher value gives more information.
                                (default: {0})
        """

        # Get the instrument name
        instName, defocalOffset = getInstName(instruFile)

        # Declare the Zk, catalog donut, and calculated z files
        self.zFile = None
        self.catFile = None
        self.zCompFile = None

        # Previous zFile (Zk), which means the z4-zn in the previous iteration
        self.zFile_m1 = None

        # Number of wavefront sensor
        self.nWFS = {self.LSST: 4, self.COMCAM: 9}[instName]

        # Number of run in each iteration of phosim
        self.nRun = {self.LSST: 1, self.COMCAM: 2}[instName]

        # Number of exposure in each run
        # For ComCam, only 1 set of intra and 1 set of extra for each iter
        self.nExp = {self.LSST: 2, self.COMCAM: 1}[instName]

        # Decide the defocal distance offset in mm
        self.offset = [-defocalOffset, defocalOffset]

        # Will refactorize the directory root problem.

        # Record the directory now
        aosDir = os.getcwd()

        # Assign the cwfs directory
        self.cwfsDir = cwfsDir

        # Change the directory now to the cwfs directory
        os.chdir(cwfsDir)

        # Read the instrument and algorithm data
        self.inst = Instrument(instruFile, int(imgSizeinPix))
        self.algo = Algorithm(algoFile, self.inst, debugLevel)

        # Change back the directory
        os.chdir(aosDir)

        # Terms of annular Zernike polynomials
        self.znwcs = self.algo.numTerms

        # Only consider the terms z4-zn. The first three terms are piston, x-tilt, and y-tilt
        self.znwcs3 = self.znwcs - 3

        # Construct the matrix of annular Zernike polynomials for all WFSs.
        self.myZn = np.zeros((self.znwcs3 * self.nWFS, 2))
        self.trueZn = self.myZn.copy()

        # Directory of intrinsic Zn file belong to the specific instrument
        intrinsicZnFileName = "intrinsic_zn_%s.txt" % band.upper()
        if (wavelength == 0.5):
            intrinsicZnFileName = "intrinsic_zn.txt"

        intrinsicFile = os.path.join(aosDataDir, instName, intrinsicZnFileName)

        # Read the intrinsic Zn data and times the wavelength
        intrinsicAll = np.loadtxt(intrinsicFile)*wavelength

        # Read the intrinsic Zk
        self.intrinsicWFS = intrinsicAll[-self.nWFS:, 3:self.znwcs].reshape((-1, 1))

        # Read the covariance matrix in unit of nm^2
        # Check this is the covariance matrix or not with Bo.
        covMFilePath = os.path.join(aosDataDir, "covM86.txt")
        self.covM = np.loadtxt(covMFilePath)

        # Reconstruct the covariance matrix if necessary (not baseline condition)
        # The way to construct the covariance matrix here is weird. Actually, there
        # is no 4x4 repeation in original "covM86.txt". Need to check with Bo for this.

        # Expand the covariance matrix by repeating the matrix
        if (self.nWFS > 4):
            nrepeat = int(np.ceil(self.nWFS/4))
            self.covM = np.tile(self.covM, (nrepeat, nrepeat))

        # Take the needed part
        if (self.nWFS != 4):
            self.covM = self.covM[:(self.znwcs3 * self.nWFS), :(self.znwcs3 * self.nWFS)]

        # Change the unit to um^2
        self.covM *= 1e-6

        # Show the debug information
        if (debugLevel >= 3):
            print("znwcs3=%d" % self.znwcs3)
            print(self.intrinsicWFS.shape)
            print(self.intrinsicWFS[:5])

    def preprocess(self, state, metr, debugLevel=0):
        """
        
        Analyze the phosim images to get donuts as separate small images and related 
        field x, y.
        
        Arguments:
            state {[aosTeleState]} -- State of telescope.
            metr {[aosMetric]} -- Metrology of telescope.
        
        Keyword Arguments:
            debugLevel {int} -- Debug level. The higher value gives more information.
                                (default: {0})
        """

        # Get the instrument name
        instName, defocalOffset = getInstName(state.inst)

        for iexp in range(self.nExp):

            for iField in range(metr.nFieldp4 - self.nWFS, metr.nFieldp4):

                # Get the chip name, pixel position x, y of dunut
                chipStr, px0, py0 = state.fieldXY2Chip(metr.fieldXp[iField], metr.fieldYp[iField], 
                                                       debugLevel)

                # Half chip ("C0", "C1") for ioffset (0, 1)
                for ioffset in [0, 1]:

                    # nRun = 1 for lsst and nRun = 2 for comcam
                    if (self.nRun == 1):
                        visit = state.obsID                    
                    else:
                        # Add the index for the comcam
                        visit = state.obsID + ioffset

                    # Get the list of paths matching the pathname pattern
                    patheNamePattern = os.path.join(state.imageDir, "iter%d" % state.iIter, 
                                            "*%d*%s*%s*E00%d.fits" % (visit, chipStr, self.halfChip[ioffset], iexp))

                    src = glob(patheNamePattern)

                    # Get the fit image and header 
                    chipFile = src[0]
                    chipImage, header = fits.getdata(chipFile, header=True)

                    # Shift the pixel x position by 0.02 deg for lsst
                    if (instName == self.LSST):

                        if (ioffset == 0):
                            # Intra image, C0, pulled 0.02 deg from right edge
                            # degree to micron then to pixel
                            px = int(px0 - 0.020 * 180000 / 10)
                        elif (ioffset == 1):
                            # Extra image, C1, pulled 0.02 deg away from left edge
                            px = int(px0 + 0.020 * 180000 / 10 - chipImage.shape[1])
                    
                    elif (instName == self.COMCAM):
                        px = px0

                    # Pixel y position keeps the same
                    py = py0

                    # Take the donut image
                    # psf here is 4 x (size of cwfsStampSize), to get centroid
                    psf = chipImage[np.max((0, py - 2 * state.cwfsStampSize)):
                                    py + 2 * state.cwfsStampSize,
                                    np.max((0, px - 2 * state.cwfsStampSize)):
                                    px + 2 * state.cwfsStampSize]

                    # The method here is to assume the clean background
                    # Need to replace here in the final
                    centroid = center_of_mass(psf)

                    # Calculate the offset between the centroid and cutted donut picture
                    offsety = centroid[0] - 2 * state.cwfsStampSize + 1
                    offsetx = centroid[1] - 2 * state.cwfsStampSize + 1
                    
                    # If the psf above has been cut on px=0 or py=0 side
                    if (py - 2*state.cwfsStampSize < 0):
                        offsety -= py - 2*state.cwfsStampSize
                    
                    if (px - 2*state.cwfsStampSize < 0):
                        offsetx -= px - 2*state.cwfsStampSize

                    # Retake the donut image that the centroid of donut is in image's center
                    psf = chipImage[int(py - state.cwfsStampSize/2 + offsety):int(py + state.cwfsStampSize/2 + offsety),
                                    int(px - state.cwfsStampSize/2 + offsetx):int(px + state.cwfsStampSize/2 + offsetx)]

                    # Take the corner images of LSST based on the correct orientation (Euler angel z)
                    if (instName == self.LSST):
                        # readout of corner raft are identical,
                        # cwfs knows how to handle rotated images
                        # note: rot90 rotates the array,
                        # not the image (as you see in ds9, or Matlab with "axis xy")
                        # that is why we need to flipud and then flip back
                        if (iField == metr.nField):
                            psf = np.flipud(np.rot90(np.flipud(psf), 2))
                        
                        elif (iField == metr.nField+1):
                            psf = np.flipud(np.rot90(np.flipud(psf), 3))
                        
                        elif (iField == metr.nField+3):
                            psf = np.flipud(np.rot90(np.flipud(psf), 1))

                    # Below, we have "0" in the front of "E00" b/c we may have many donuts in the future
                    # Hard coded the "0" here. It should be removed in the future.
                    stampFile = os.path.join(state.imageDir, "iter%d" % state.iIter, 
                                    "sim%d_iter%d_wfs%d_%s_0_E00%d.fits" % (state.iSim, state.iIter, iField, 
                                                                            self.wfsName[ioffset], iexp))

                    # Delete the existed file if necessary               
                    if os.path.isfile(stampFile):
                        os.remove(stampFile)
                    
                    # Declare a header data unit and write the "psf" image into the file (stamp file)
                    hdu = fits.PrimaryHDU(psf)
                    hdu.writeto(stampFile)

                    # Write the atomosphere data into the file
                    if ((iField == metr.nFieldp4 - self.nWFS) and (ioffset == 0)):
                        
                        fid = open(state.atmFile[iexp], "w")
                        fid.write("Layer# \t seeing \t L0 \t\t wind_v \t wind_dir\n")
                        
                        # Severn layers of atmosphere
                        for ilayer in range(7):
                            fid.write("%d \t %.6f \t %.5f \t %.6f \t %.6f\n" % (ilayer, 
                                        header["SEE%d" % ilayer], header["OSCL%d" % ilayer],
                                        header["WIND%d" % ilayer], header["WDIR%d" % ilayer]))
                        
                        fid.close()

                    # Show the information for the debug
                    if (debugLevel >= 3):
                        print("px = %d, py = %d" % (px, py))
                        print("offsetx = %d, offsety = %d" % (offsetx, offsety))
                        print("passed %d, %s" % (iField, self.wfsName[ioffset]))

            # Make an image of the 8 donuts
            for iField in range(metr.nFieldp4-self.nWFS, metr.nFieldp4):

                # Get the chip name, pixel position x, y of dunut
                chipStr, px, py = state.fieldXY2Chip(metr.fieldXp[iField], metr.fieldYp[iField], debugLevel)
                
                # Plot C0 and C1 images
                for ioffset in [0, 1]:
                    
                    # Define the pattern of name
                    patheNamePattern = os.path.join(state.imageDir, "iter%d" % state.iIter, 
                        "sim%d_iter%d_wfs%d_%s_*E00%d.fits" % (state.iSim, state.iIter, iField, 
                                                               self.wfsName[ioffset], iexp))

                    # Get the file list
                    src = glob(patheNamePattern)
                    
                    # Open the image fits file
                    IHDU = fits.open(src[0])

                    # Get the image data
                    psf = IHDU[0].data

                    # Close the image fits file
                    IHDU.close()
                    
                    # Arrange the donut images to a single figure
                    if (instName == self.LSST):
                        
                        nRow = 2
                        nCol = 4
                        
                        if (iField == metr.nField):
                            # 3 and 4
                            pIdx = 3 + ioffset
                        elif (iField == metr.nField + 1):
                            # 1 and 2
                            pIdx = 1 + ioffset  
                        elif (iField == metr.nField + 2):
                            # 5 and 6
                            pIdx = 5 + ioffset  
                        elif (iField == metr.nField + 3):
                            # 7 and 8
                            pIdx = 7 + ioffset  
                    
                    elif (instName == self.COMCAM):

                        nRow = 3
                        nCol = 6
                        
                        ic = np.floor(iField / nRow)
                        ir = iField % nRow
                        
                        # does iField=0 give 13 and 14?
                        pIdx = int((nRow - ir - 1) * nCol + ic * 2 + 1 + ioffset)     
                        # print('pIdx = %d, chipStr= %s'%(pIdx, chipStr))
                   
                    plt.subplot(nRow, nCol, pIdx)
                    plt.imshow(psf, origin="lower", interpolation="none")
                    plt.title("%s_%s" % (chipStr, self.wfsName[ioffset]), fontsize=10)
                    plt.axis("off")

            # Give the file name
            pngFile = os.path.join(state.imageDir, "iter%d" % state.iIter, 
                                   "sim%d_iter%d_wfs_E00%d.png" % (state.iSim, state.iIter, iexp))
            # Save the image to the file
            plt.savefig(pngFile, bbox_inches="tight")

            # Write out the catalog for good wfs stars (field x, y and the donut image file path)
            fid = open(self.catFile[iexp], "w")
            for ii in range(metr.nFieldp4-self.nWFS, metr.nFieldp4):

                # Get the intra- and extra-focal image file names
                fileName = lambda x: glob(os.path.join(state.imageDir, "iter%d" % state.iIter, 
                                          "sim%d_iter%d_wfs%d_%s_*E00%d.fits" % (state.iSim, 
                                            state.iIter, ii, self.wfsName[x], iexp)))[0]
                intraFile = fileName(0)
                extraFile = fileName(1)
                
                # Write the information into the file
                intraFieldX = metr.fieldXp[ii]
                intraFieldY = metr.fieldYp[ii]

                extraFieldX = intraFieldX
                extraFieldY = intraFieldY

                # The corner WFS has the Euler z = 90 degree rotation in LSST
                if (instName == self.LSST):

                    delta = 0.02
                    if (ii == 31):

                        intraFieldX -= delta
                        extraFieldX += delta

                    elif (ii == 32):

                        intraFieldY -= delta
                        extraFieldY += delta

                    elif (ii == 33):

                        intraFieldX += delta
                        extraFieldX -= delta

                    elif (ii == 34):

                        intraFieldY += delta
                        extraFieldY -= delta

                fid.write("%9.6f %9.6f %9.6f %9.6f %s %s\n" % (intraFieldX, intraFieldY, 
                                        extraFieldX, extraFieldY, intraFile, extraFile))

            fid.close()

    def parallelCwfs(self, cwfsModel, numproc, writeToFile=True):
        """
        
        Calculate the wave front error by parallel.
        
        Arguments:
            cwfsModel {[str]} -- Optical model.
            numproc {[int]} -- Numebr of processors.
        
        Keyword Arguments:
            writeToFile {bool} -- Write the calculated result into the file or not. 
                                  (default: {True})
        """

        # Do the parallel calculation
        for iexp in range(self.nExp):
            
            argList = []
            
            # Read the catalog file, which records the field x, y and file path of donut image
            fid = open(self.catFile[iexp])
            for line in fid:

                data = line.split()
                
                # Get the field x, y
                I1Field = [float(data[0]), float(data[1])]
                I2Field = [float(data[2]), float(data[3])]
                
                # Get the file path
                I1File = data[4]
                I2File = data[5]

                # Collect the information
                argList.append((I1File, I1Field, I2File, I2Field,
                                self.inst, self.algo, cwfsModel))
                
            fid.close()

            # Do the parallel calculation
            pool = multiprocessing.Pool(numproc)
            zcarray = pool.map(runcwfs, argList)
            pool.close()
            pool.join()

            # Get the data in array
            zcarray = np.array(zcarray)

            # Write to file or not
            if (writeToFile):
                np.savetxt(self.zFile[iexp], zcarray)
            else:
                print(zcarray)


    def checkZ4C(self, state, metr, debugLevel=0, writeToFile=True):
        """
        
        Compare the calculated wavefront error with the truth based on the optical path difference 
        (OPD) in the annular Zernike polynomials.
        
        Arguments:
            state {[aosTeleState]} -- State of telescope.
            metr {[aosMetric]} -- Metrology of telescope.
        
        Keyword Arguments:
            debugLevel {int} -- Debug level. The higher value gives more information.
                                (default: {0})
            writeToFile {bool} -- Write the calculated result into the file or not. 
                                  (default: {True})
        """

        # Read the calculated Zk file, the unit of Zk is um
        # Exposure 0
        z4c = np.loadtxt(self.zFile[0])
        
        # Exposure 1 --> Need tp check with Bo. For the ComCam, this file should not exist.
        z4cE001 = np.loadtxt(self.zFile[1])
        
        # Read the Zk truth file based on the fitting of OPD from PhoSim
        z4cTrue = np.zeros((metr.nFieldp4, self.znwcs, state.nOPDw))
        data = np.loadtxt(state.zTrueFile)
        for ii in range(state.nOPDw):
            z4cTrue[:, :, ii] = data[ii*metr.nFieldp4:(ii + 1)*metr.nFieldp4, :]

        # Get the instrument name
        instName, defocalOffset = getInstName(state.inst)

        # Set the size of figure as 10 inch by 8 inch
        plt.figure(figsize=(10, 8))

        # Arrange the row, column, wavefront sensor  
        if (instName == self.LSST):
            # Subplots go like this
            #  2 1
            #  3 4
            pIdx = [2, 1, 3, 4]

            nRow = 2
            nCol = 2

        elif (instName == self.COMCAM):
            # Subplots go like this
            #  7 4 1
            #  8 5 2
            #  9 6 3
            pIdx = [7, 4, 1, 8, 5, 2, 9, 6, 3]
            
            nRow = 3
            nCol = 3

        # Label of the Zk term (z4-zn)
        x = range(4, self.znwcs + 1)

        # Plot the figure of each WFS
        for ii in range(self.nWFS):

            # Get the chip name and pixel positions of donuts
            chipStr, px, py = state.fieldXY2Chip(metr.fieldXp[ii + metr.nFieldp4 - self.nWFS],
                                                 metr.fieldYp[ii + metr.nFieldp4 - self.nWFS], 
                                                 debugLevel)
            
            # Subplot of the figure
            plt.subplot(nRow, nCol, pIdx[ii])
            plt.plot(x, z4c[ii, :self.znwcs3], label="CWFS_E000", marker="*", color="r", markersize=6)
            plt.plot(x, z4cE001[ii, :self.znwcs3], label="CWFS_E001", marker="v", color="g", markersize=6)
            
            # Plot the true Zk based on the OPD
            for irun in range(state.nOPDw):
                
                # Title name
                mylabel = "Truth" if irun==0 else ""

                plt.plot(x, z4cTrue[ii + metr.nFieldp4 - self.nWFS, 3:self.znwcs, irun],
                         label=mylabel, marker=".", color="b", markersize=10)
            
            # Selected label x-axis and y-axis. The labels are only at the left and bottom of figure 
            if ((instName == self.LSST and ii in (1, 2)) or (instName == self.COMCAM and (ii <= 2))):
                plt.ylabel("$\mu$m")
            
            if ((instName == self.LSST and ii in (2, 3)) or (instName == self.COMCAM and (ii % nRow == 0))):
                plt.xlabel("Zernike Index")
            
            # Put the legend
            leg = plt.legend(loc="best")
            leg.get_frame().set_alpha(0.5)
            
            # Put the grid
            plt.grid()

            # Give the title
            plt.title('Zernikes %s' % chipStr, fontsize=10)

        # Write the image to file or not
        if (writeToFile):
            plt.savefig(self.zCompFile, bbox_inches='tight')
        else:
            plt.show()

    def getZ4CfromBase(self, baserun, stateSimNum):
        """
        
        Construct the zFile and zCompFile files by hard-linking to the related files in 
        the indicated base run.
        
        Arguments:
            baserun {[int]} -- Indicated simulation number.
            stateSimNum {[int]} -- Telescope simulation number.
        """

        # Get Zk from the specific file which is assigned as a base run (iter0)
        for iexp in range(self.nExp):
    
            if not os.path.isfile(self.zFile[iexp]):

                # Hard link the file to avoid the repeated calculation
                self.__hardLinkFile(self.zFile[iexp], baserun, stateSimNum)
    
        if not os.path.isfile(self.zCompFile):

            # Hard link the file to avoid the repeated calculation
            self.__hardLinkFile(self.zCompFile, baserun, stateSimNum)

    def __hardLinkFile(self, targetFilePath, sourceNum, targetNum):
        """
        
        Hard link the past calculation result instead of repeated calculation.
        
        Arguments:
            targetFilePath {[str]} -- Path of file that is intended to do the hard link 
                                      with the previous result.
            sourceNum {[int]} -- Source simulation number.
            targetNum {[int]} -- Target simulation number.
        """

        # Get the path of base run file by changing the simulation number
        sourceFilePath = targetFilePath.replace("sim%d" % targetNum, "sim%d" % sourceNum)

        # Construct a hard link
        os.link(sourceFilePath, targetFilePath)

def runcwfs(argList):
    """
    
    Calculate the wavefront error in sigle processor.
    
    Arguments:
        argList {[list]} -- Inputs of cwfs to calulate the wavefront front.
    
    Returns:
        [ndarray] -- z4-zn and flag of calculation.
    """

    # Intra-focal image and related field x, y
    I1File = argList[0]
    I1Field = argList[1]
    
    # Extra-focal image and related field x, y
    I2File = argList[2]
    I2Field = argList[3]
    
    # Instrument
    inst = argList[4]

    # Algorithm
    algo = argList[5]
    
    # Optical model
    model = argList[6]

    # Set the images
    I1 = Image(readFile(I1File), I1Field, "intra")
    I2 = Image(readFile(I2File), I2Field, "extra")
    
    # Run the algorithm to solve the TIE
    algo.reset(I1, I2)
    algo.runIt(inst, I1, I2, model)

    return np.append(algo.zer4UpNm * 1e-3, algo.caustic)

if __name__ == "__main__":

    # cwfs directory
    cwfsDir = "/Users/Wolf/Documents/github/cwfs"

    # Optical model
    cwfsModel="offAxis"

    # AOS data directory
    aosDataDir = "/Users/Wolf/Documents/stash/ts_lsst_aos_phosim/data"

    # Data position
    pertDir = "/Users/Wolf/Documents/aosOutput/pert/sim2"
    iIter = 0
    
    # Set the catalog file for locating donut images and related field x, y 
    catFile = ["%s/iter%d/wfs_catalog_E00%d.txt" % (pertDir, iIter, iexp) for iexp in [0, 1]]

    # Instantiate the AOS wavefront sensor estimator
    wfs = aosWFS(cwfsDir, "lsst", "exp", 128, "g", 0.5, aosDataDir, debugLevel=0)

    # Set the catalog file
    wfs.catFile = catFile

    # Calculate the wavefront error
    # t0 = time.time()
    # wfs.parallelCwfs(cwfsModel, 2, writeToFile=False)
    # t1 = time.time()
    # print(t1-t0)


