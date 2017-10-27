#!/usr/bin/env python
##
# @authors: Bo Xin
# @       Large Synoptic Survey Telescope

import os, re
import glob
import multiprocessing
import copy

import numpy as np
from astropy.io import fits
from scipy import ndimage
import matplotlib.pyplot as plt

from lsst.cwfs.algorithm import Algorithm
from lsst.cwfs.instrument import Instrument
from lsst.cwfs.image import Image, readFile

from wep.WFEstimator import WFEstimator

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

        Raises:
            RuntimeError -- No instrument found.
        """

        # Get the instrument name
        m = re.match(r"([a-z]+)(?:(\d+))?$", instruFile)
        if m is None:
             raise RuntimeError("Cannot get the instrument name: %s." % instruFile)
        instName = m.groups()[0]

        # Declare the Zk, catalog donut, and calculated z files
        self.zFile = None
        self.catFile = None
        self.zCompFile = None

        # Number of wavefront sensor
        self.nWFS = {self.LSST: 4, self.COMCAM: 9}[instName]

        # Number of run in each iteration of phosim
        self.nRun = {self.LSST: 1, self.COMCAM: 2}[instName]

        # Number of exposure in each run
        # For ComCam, only 1 set of intra and 1 set of extra for each iter
        self.nExp = {self.LSST: 2, self.COMCAM: 1}[instName]

        # Decide the defocal distance offset in mm
        defocalOffset = m.groups()[1]
        if (defocalOffset is not None):
            defocalOffset = float(defocalOffset)/10
        else:
            # Default defocal distance is 1.5 mm
            defocalOffset = 1.5

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

        # Read the convolution matrix in unit of nm^2
        covMFilePath = os.path.join(aosDataDir, "covM86.txt")
        self.covM = np.loadtxt(covMFilePath)

        # Reconstruct the convolution matrix if necessary (not baseline condition)
        # The way to construct the convolution matrix here is weird. Actually, there
        # is no 4x4 repeation in original "covM86.txt". Need to check with Bo for this.

        # Expand the convolution matrix by repeating the matrix
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

    def preprocess(self, state, metr, debugLevel):

        # Analyze the phosim data to get eight donuts

        for iexp in range(0, self.nExp):
            for iField in range(metr.nFieldp4 - self.nWFS, metr.nFieldp4):
                chipStr, px0, py0 = state.fieldXY2Chip(
                    metr.fieldXp[iField], metr.fieldYp[iField], debugLevel)
                for ioffset in [0, 1]:
                    if self.nRun == 1:
                        src = glob.glob('%s/iter%d/*%d*%s*%s*E00%d.fits' %
                                        (state.imageDir, state.iIter,
                                            state.obsID,
                                        chipStr, self.halfChip[ioffset], iexp))
                    else:
                        src = glob.glob('%s/iter%d/*%d*%s*%s*E00%d.fits' %
                                        (state.imageDir, state.iIter,
                                            state.obsID + ioffset,
                                        chipStr, self.halfChip[ioffset], iexp))
                    chipFile = src[0]
                    chipImage, header = fits.getdata(chipFile,header=True)

                    if state.inst[:4] == 'lsst':
                        if ioffset == 0:
                            # intra image, C0, pulled 0.02 deg from right edge
                            # degree to micron then to pixel
                            px = int(px0 - 0.020 * 180000 / 10)
                        elif ioffset == 1:
                            # extra image, C1, pulled 0.02 deg away from left edge
                            px = int(px0 + 0.020 * 180000 / 10 - chipImage.shape[1])
                    elif state.inst[:6] == 'comcam':
                        px = px0
                    py = copy.copy(py0)

                    # psf here is 4 x the size of cwfsStampSize, to get centroid
                    psf = chipImage[np.max((0, py - 2 * state.cwfsStampSize)):
                                    py + 2 * state.cwfsStampSize,
                                    np.max((0, px - 2 * state.cwfsStampSize)):
                                    px + 2 * state.cwfsStampSize]
                    centroid = ndimage.measurements.center_of_mass(psf)
                    offsety = centroid[0] - 2 * state.cwfsStampSize + 1
                    offsetx = centroid[1] - 2 * state.cwfsStampSize + 1
                    # if the psf above has been cut on px=0 or py=0 side
                    if py - 2 * state.cwfsStampSize < 0:
                        offsety -= py - 2 * state.cwfsStampSize
                    if px - 2 * state.cwfsStampSize < 0:
                        offsetx -= px - 2 * state.cwfsStampSize

                    psf = chipImage[
                        int(py - state.cwfsStampSize / 2 + offsety):
                        int(py + state.cwfsStampSize / 2 + offsety),
                        int(px - state.cwfsStampSize / 2 + offsetx):
                        int(px + state.cwfsStampSize / 2 + offsetx)]

                    if state.inst[:4] == 'lsst':
                        # readout of corner raft are identical,
                        # cwfs knows how to handle rotated images
                        # note: rot90 rotates the array,
                        # not the image (as you see in ds9, or Matlab with
                        #                  "axis xy")
                        # that is why we need to flipud and then flip back
                        if iField == metr.nField:
                            psf = np.flipud(np.rot90(np.flipud(psf), 2))
                        elif iField == metr.nField + 1:
                            psf = np.flipud(np.rot90(np.flipud(psf), 3))
                        elif iField == metr.nField + 3:
                            psf = np.flipud(np.rot90(np.flipud(psf), 1))

                    # below, we have 0 b/c we may have many
                    stampFile = '%s/iter%d/sim%d_iter%d_wfs%d_%s_0_E00%d.fits' % (
                        state.imageDir, state.iIter, state.iSim, state.iIter,
                        iField, self.wfsName[ioffset], iexp)
                    if os.path.isfile(stampFile):
                        os.remove(stampFile)
                    hdu = fits.PrimaryHDU(psf)
                    hdu.writeto(stampFile)

                    if ((iField == metr.nFieldp4 - self.nWFS) and (ioffset == 0)):
                        fid = open(state.atmFile[iexp], 'w')
                        fid.write('Layer# \t seeing \t L0 \t\t wind_v \t wind_dir\n')
                        for ilayer in range(7):
                            fid.write('%d \t %.6f \t %.5f \t %.6f \t %.6f\n'%(
                                ilayer,header['SEE%d'%ilayer],
                                header['OSCL%d'%ilayer],
                                header['WIND%d'%ilayer],
                                header['WDIR%d'%ilayer]))
                        fid.close()

                    if debugLevel >= 3:
                        print('px = %d, py = %d' % (px, py))
                        print('offsetx = %d, offsety = %d' % (offsetx, offsety))
                        print('passed %d, %s' % (iField, self.wfsName[ioffset]))

            # make an image of the 8 donuts
            for iField in range(metr.nFieldp4 - self.nWFS, metr.nFieldp4):
                chipStr, px, py = state.fieldXY2Chip(
                    metr.fieldXp[iField], metr.fieldYp[iField], debugLevel)
                for ioffset in [0, 1]:
                    src = glob.glob('%s/iter%d/sim%d_iter%d_wfs%d_%s_*E00%d.fits' % (
                        state.imageDir, state.iIter, state.iSim, state.iIter,
                        iField, self.wfsName[ioffset], iexp))
                    IHDU = fits.open(src[0])
                    psf = IHDU[0].data
                    IHDU.close()
                    if state.inst[:4] == 'lsst':
                        nRow = 2
                        nCol = 4
                        if iField == metr.nField:
                            pIdx = 3 + ioffset  # 3 and 4
                        elif iField == metr.nField + 1:
                            pIdx = 1 + ioffset  # 1 and 2
                        elif iField == metr.nField + 2:
                            pIdx = 5 + ioffset  # 5 and 6
                        elif iField == metr.nField + 3:
                            pIdx = 7 + ioffset  # 7 and 8
                    elif state.inst[:6] == 'comcam':
                        nRow = 3
                        nCol = 6
                        ic = np.floor(iField / nRow)
                        ir = iField % nRow
                        # does iField=0 give 13 and 14?
                        pIdx = int((nRow - ir - 1) * nCol + ic * 2 + 1 + ioffset)
                        # print('pIdx = %d, chipStr= %s'%(pIdx, chipStr))
                    plt.subplot(nRow, nCol, pIdx)
                    plt.imshow(psf, origin='lower', interpolation='none')
                    plt.title('%s_%s' %
                              (chipStr, self.wfsName[ioffset]), fontsize=10)
                    plt.axis('off')

            # plt.show()
            pngFile = '%s/iter%d/sim%d_iter%d_wfs_E00%d.png' % (
                state.imageDir, state.iIter, state.iSim, state.iIter, iexp)
            plt.savefig(pngFile, bbox_inches='tight')

            # write out catalog for good wfs stars
            fid = open(self.catFile[iexp], 'w')
            for i in range(metr.nFieldp4 - self.nWFS, metr.nFieldp4):
                intraFile = glob.glob('%s/iter%d/sim%d_iter%d_wfs%d_%s_*E00%d.fits' % (
                    state.imageDir, state.iIter, state.iSim, state.iIter, i,
                    self.wfsName[0], iexp))[0]
                extraFile = glob.glob('%s/iter%d/sim%d_iter%d_wfs%d_%s_*E00%d.fits' % (
                    state.imageDir, state.iIter, state.iSim, state.iIter, i,
                    self.wfsName[1], iexp))[0]
                if state.inst[:4] == 'lsst':
                    if i == 31:
                        fid.write('%9.6f %9.6f %9.6f %9.6f %s %s\n' % (
                            metr.fieldXp[i] - 0.020, metr.fieldYp[i],
                            metr.fieldXp[i] + 0.020, metr.fieldYp[i],
                            intraFile, extraFile))
                    elif i == 32:
                        fid.write('%9.6f %9.6f %9.6f %9.6f %s %s\n' % (
                            metr.fieldXp[i], metr.fieldYp[i] - 0.020,
                            metr.fieldXp[i], metr.fieldYp[i] + 0.020,
                            intraFile, extraFile))
                    elif i == 33:
                        fid.write('%9.6f %9.6f %9.6f %9.6f %s %s\n' % (
                            metr.fieldXp[i] + 0.020, metr.fieldYp[i],
                            metr.fieldXp[i] - 0.020, metr.fieldYp[i],
                            intraFile, extraFile))
                    elif i == 34:
                        fid.write('%9.6f %9.6f %9.6f %9.6f %s %s\n' % (
                            metr.fieldXp[i], metr.fieldYp[i] + 0.020,
                            metr.fieldXp[i], metr.fieldYp[i] - 0.020,
                            intraFile, extraFile))
                elif state.inst[:6] == 'comcam':
                    fid.write('%9.6f %9.6f %9.6f %9.6f %s %s\n' % (
                        metr.fieldXp[i], metr.fieldYp[i],
                        metr.fieldXp[i], metr.fieldYp[i],
                        intraFile, extraFile))
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


    def checkZ4C(self, state, metr, debugLevel):
        z4c = np.loadtxt(self.zFile[0])  # in micron
        z4cE001 = np.loadtxt(self.zFile[1])
        z4cTrue = np.zeros((metr.nFieldp4, self.znwcs, state.nOPDw))
        aa = np.loadtxt(state.zTrueFile)
        for i in range(state.nOPDw):
            z4cTrue[:, :, i] = aa[i*metr.nFieldp4:(i+1)*metr.nFieldp4, :]

        x = range(4, self.znwcs + 1)
        plt.figure(figsize=(10, 8))
        if state.inst[:4] == 'lsst':
            # subplots go like this
            #  2 1
            #  3 4
            pIdx = [2, 1, 3, 4]
            nRow = 2
            nCol = 2
        elif state.inst[:6] == 'comcam':
            pIdx = [7, 4, 1, 8, 5, 2, 9, 6, 3]
            nRow = 3
            nCol = 3

        for i in range(self.nWFS):
            chipStr, px, py = state.fieldXY2Chip(
                metr.fieldXp[i + metr.nFieldp4 - self.nWFS],
                metr.fieldYp[i + metr.nFieldp4 - self.nWFS], debugLevel)
            plt.subplot(nRow, nCol, pIdx[i])
            plt.plot(x, z4c[i, :self.znwcs3], label='CWFS_E000',
                     marker='*', color='r', markersize=6)
            plt.plot(x, z4cE001[i, :self.znwcs3], label='CWFS_E001',
                     marker='v', color='g', markersize=6)
            for irun in range(state.nOPDw):
                if irun==0:
                    mylabel = 'Truth'
                else:
                    mylabel = ''
                plt.plot(x, z4cTrue[i + metr.nFieldp4 - self.nWFS, 3:self.znwcs,
                                        irun],
                             label=mylabel,
                        marker='.', color='b', markersize=10)
            if ((state.inst[:4] == 'lsst' and (i == 1 or i == 2)) or
                    (state.inst[:6] == 'comcam' and (i <= 2))):
                plt.ylabel('$\mu$m')
            if ((state.inst[:4] == 'lsst' and (i == 2 or i == 3)) or
                    (state.inst[:6] == 'comcam' and (i % nRow == 0))):
                plt.xlabel('Zernike Index')
            leg = plt.legend(loc="best")
            leg.get_frame().set_alpha(0.5)
            plt.grid()
            plt.title('Zernikes %s' % chipStr, fontsize=10)

        plt.savefig(self.zCompFile, bbox_inches='tight')

    def getZ4CfromBase(self, baserun, state):
        for iexp in range(self.nExp):
            if not os.path.isfile(self.zFile[iexp]):
                baseFile = self.zFile[iexp].replace(
                    'sim%d' % state.iSim, 'sim%d' % baserun)
                os.link(baseFile, self.zFile[iexp])
        if not os.path.isfile(self.zCompFile):
            baseFile = self.zCompFile.replace(
                'sim%d' % state.iSim, 'sim%d' % baserun)
            os.link(baseFile, self.zCompFile)

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

# def runcwfsTemp(argList):

#     # Needed information
#     I1File = argList[0]
#     I1Field = argList[1]

#     I2File = argList[2]
#     I2Field = argList[3]

#     inst = argList[4]
#     algo = argList[5]

#     model = argList[6]

#     # Instantiate the WFEstimator
#     instruFolderPath = "/Users/Wolf/Documents/stash/ts_tcs_wep/instruData"
#     algoFolderPath = "/Users/Wolf/Documents/stash/ts_tcs_wep/algo"
#     wfsEst = WFEstimator(instruFolderPath, algoFolderPath)

#     # Set the image
#     wfsEst.setImg(I1Field, imageFile=I1File, defocalType="intra")
#     wfsEst.setImg(I2Field, imageFile=I2File, defocalType="extra")

#     # Set the configuration
#     wfsEst.config(solver=algo, instName=inst, opticalModel=model)

#     # Do the calculation
#     zer4UpNm = wfsEst.calWfsErr()

#     # Return the value in the unit of um
#     algoCaustic = {"True": 1, "False": 0}[str(wfsEst.algo.caustic)]

#     return np.append(zer4UpNm*1e-3, algoCaustic)


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
    catFile = ["%s/iter%d/wfs_catalog_E00%d.txt" % (pertDir, iIter, iexp) for iexp in [0, 1]]

    # Instantiate the AOS wavefront sensor estimator
    wfs = aosWFS(cwfsDir, "lsst", "exp", 128, "g", 0.5, aosDataDir, debugLevel=0)

    # Set the cat file
    wfs.catFile = catFile

    # Calculate the wavefront error
    t0 = time.time()
    wfs.parallelCwfs(cwfsModel, 2, writeToFile=False)
    t1 = time.time()
    print(t1-t0)

