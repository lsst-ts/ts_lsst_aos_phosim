#!/usr/bin/env python
##
# @authors: Bo Xin
# @       Large Synoptic Survey Telescope

import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

from aos.aosMetric import getInstName

class aosController(object):

    # Rigid body stroke
    # Check with Bo for the unit and how to get this
    rbStroke = np.array([5900, 6700, 6700, 432, 432, 8700, 7600, 7600, 864, 864])

    def __init__(self, ctrlDir, instruFile, paramFile, esti, metr, wfs, M1M3Force, M2Force,
                 effwave, gain, debugLevel=0):

        # Name of controller parameter file
        self.filename = os.path.join(ctrlDir, (paramFile + ".ctrl"))

        # Assign the gain value
        self.gain = gain

        # Parameters of controller
        self.strategy = None
        self.xref = None
        self.shiftGear = None
        self.shiftGearThres = None
        self.rhoM13 = None
        self.rhoM2 = None
        self.rho = None

        # Read the file to get the parameters
        self.__readFile(self.filename)

        # Get the instrument name
        instName, defocalOffset = getInstName(instruFile)

        # Read the y2 file of specific instrument. Check with Bo for this.
        # Do not understand the meaning of y2.
        y2FilePath = os.path.join(ctrlDir, instName, "y2*txt")
        self.y2File = glob(y2FilePath)[0]
        self.y2 = np.loadtxt(self.y2File)

        # Show the paths of control strategy and y2 files
        if (debugLevel >= 1):
            print("control strategy: %s" % self.strategy)
            print("Using y2 file: %s" % self.y2File)

        # Get the authority (H) of each subsystem (J = y.T * Q * y + rho * u.T * H * u). 
        # The matrix H defines the distribution of control authority among various 
        # actuator groups (rigid body abd shape actuators) such that 1 um or 1 arcsec 
        # rigid body displacement corresponds to 1 N actuator.  

        # Establish control authority of the DOFs

        # Get the normalized rigid body weighting
        # For the rigid body DOF (r for rigid), weight based on the total stroke
        # Need to check with Bo for the reason of normalization
        rbW = self.rbStroke[0]/self.rbStroke

        # Construct the authority array (H)
        self.Authority = self.__getAuthorityH(rbW, M1M3Force, M2Force, esti.dofIdx, esti.nB13Max, esti.nB2Max)

        # Calculate the range of motion of DOFs, which means there is no truncation of DOF.
        # range = 1/self.Authority*rbStroke[0]
        dofIdxNoTrunc = np.ones(len(esti.dofIdx), dtype=bool)

        authorityNoTrunc = self.__getAuthorityH(rbW, M1M3Force, M2Force, dofIdxNoTrunc, esti.nB13Max, esti.nB2Max)
        self.range = 1/authorityNoTrunc*self.rbStroke[0]




        if esti.strategy == 'pinv':
            if esti.normalizeA:
                esti.normA(self.Authority)
        elif esti.strategy == 'opti':
            if esti.fmotion > 0:
                esti.optiAinv(self.range, wfs.covM)

        if (self.strategy == 'optiPSSN'):
            # use rms^2 as diagnal
            self.mH = np.diag(self.Authority**2)
            if self.xref == 'x0xcor':
                idx1 = 10 + 3  # b3 of M1M3 bending
                idx2 = 10 + esti.nB13Max + 5  # b5 of M2 bending
                if esti.dofIdx[idx1] and esti.dofIdx[idx2]:
                    idx1 = sum(esti.dofIdx[:idx1]) - 1
                    idx2 = sum(esti.dofIdx[:idx2]) - 1
                    self.mH[idx1, idx2] = self.Authority[idx1] * \
                        self.Authority[idx2] * 100  # 10 times penalty

            # wavelength below in um,b/c output of A in um
            CCmat = np.diag(metr.pssnAlpha) * (2 * np.pi / effwave)**2
            self.mQ = np.zeros((esti.Ause.shape[1], esti.Ause.shape[1]))
            for iField in range(metr.nField):
                aa = esti.senM[iField, :, :]
                Afield = aa[np.ix_(esti.zn3Idx, esti.dofIdx)]
                mQf = Afield.T.dot(CCmat).dot(Afield)
                self.mQ = self.mQ + metr.w[iField] * mQf
            self.mF = np.linalg.pinv(self.mQ + self.rho**2 * self.mH)

            if debugLevel >= 3:
                print(self.mQ[0, 0])
                print(self.mQ[0, 9])

    def __getAuthorityH(self, rbW, M1M3Force, M2Force, dofIdx, nB13Max, nB2Max):
        """
        
        Get the authority matrix H, which is constructed by all subsystems.
        
        Arguments:
            rbW {[ndarray]} -- Weighting of rigid body (M2 hexapod + camera hexapod).
            M1M3Force {[ndarray]} -- Acturator force matrix of M1M3.
            M2Force {[ndarray]} -- Acturator force matrix of M2.
            dofIdx {[ndarray]} -- Available degree of freedom index.
            nB13Max {[int]} -- Maximum number of available bending mode of M1M3.
            nB2Max {[int]} -- Maximum number of available bending mode of M2.
        
        Returns:
            [ndarray] -- Authority matrix H.
        """

        # Get the elements of matrix H belong to rigid body
        mHr = rbW[dofIdx[:10]]

        # Get the elements of matrix H belong to M1M3
        # The sum of index length in DOF of M2 and camera hexapods is 10
        mHM13 = self.__getSubH(M1M3Force, dofIdx, 10, nB13Max)

        # Get the elements of matrix H belong to M2
        # The sum of index length in DOF of M2 and camera hexapods is 10
        mHM2 = self.__getSubH(M2Force, dofIdx, 30, nB2Max)

        # Construct the authority array (H)
        authority = np.concatenate((mHr, self.rhoM13*mHM13, self.rhoM2*mHM2))

        return authority

    def __getSubH(self, force, dofIdx, startIdx, targetSubIndexLen):
        """
        
        Calculate the distribution of control authority of specific subsystem. 
        
        Arguments:
            force {[ndarray]} -- Actuator force matrix (row: actuator, column: bending mode).
            dofIdx {[ndarray]} -- Available degree of freedom.
            startIdx {[int]} -- Start index of the subsystem in dofIdx.
            targetSubIndexLen {[int]} -- Length of index (bending mode) belong to the 
                                         subsystem in dofIdx.
        
        Returns:
            [ndarray] -- Distrubution of subsystem authority.
        """

        # Get the force data
        data = force[:, :int(targetSubIndexLen)]
        data = data[:, dofIdx[int(startIdx):int(startIdx + targetSubIndexLen)]]

        # Calculate the element of matrix H (authority) of specific subsystem
        # Check with Bo for the physical meaning of this
        mH = np.std(data, axis=0)

        return mH

    def __readFile(self, filePath):
        """
        
        Read the AOS controller parameter file.
        
        Arguments:
            filePath {[str]} -- Path of file of controller parameters.
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
            
                # Define the control strategy. Default is optiPSSN.
                # This means the target of control is to minimize the PSSN.
                if (line.startswith("control_strategy")):
                    self.strategy = line.split()[1]
            
                # The strategy in optiPSSN. There are three types: _0, _x0, and  _x00.
                # The offset will trace the real value and target for 0 in "_0" type.
                # The offset will only trace the previous one in "_x0" type.
                # The offset will only trace the relative changes of offset without 
                # regarding the real value in "_x00" type. 
                if (line.startswith("xref")):
                    self.xref = line.split()[1]
            
                # The new gain value once the system is stable. That means PSSN is less 
                # than the threshold, and the system is judged to have corrected the 
                # wavefront error.
                # Not sure need to add the turn-back-original-gain mechanism or not.
                # Check with Bo for this.
                elif (line.startswith("shift_gear")):
                    self.shiftGear = bool(int(line.split()[1]))
            
                    # The threshold of PSSN to change the gain value
                    if (self.shiftGear):
                        self.shiftGearThres = float(line.split()[2])
            
                # M1M3 actuator penalty factor
                # Not really understand the meaning of penalty here. Check with Bo.
                elif (line.startswith("M1M3_actuator_penalty")):
                    self.rhoM13 = float(line.split()[1])
            
                # M2 actuator penalty factor
                # Not really understand the meaning of penalty here. Check with Bo.
                elif (line.startswith("M2_actuator_penalty")):
                    self.rhoM2 = float(line.split()[1])
            
                # Penalty on control motion as a whole
                # Not really understand the meaning of penalty here. Check with Bo.
                elif (line.startswith("Motion_penalty")):
                    self.rho = float(line.split()[1])

        # Close the file
        fid.close()

    def getMotions(self, esti, metr, wfs, state):
        self.uk = np.zeros(esti.ndofA)
        self.gainUse = self.gain
        if hasattr(self, 'shiftGear'):
            if self.shiftGear and (metr.GQFWHMeff > self.shiftGearThres):
                self.gainUse = 1
        if (self.strategy == 'null'):
            y2 = np.zeros(sum(esti.zn3Idx))
            for iField in range(metr.nField):
                y2f = self.y2[iField, esti.zn3Idx]
                y2 = y2 + metr.w[iField] * y2f
            y2c = np.repeat(y2, wfs.nWFS)
            x_y2c = esti.Ainv.dot(y2c)
            if esti.normalizeA:
                x_y2c = x_y2c / esti.dofUnit
            self.uk[esti.dofIdx] = - self.gainUse * \
                (esti.xhat[esti.dofIdx] + x_y2c)

        elif (self.strategy == 'optiPSSN'):
            CCmat = np.diag(metr.pssnAlpha) * (2 * np.pi / state.effwave)**2
            Mx = np.zeros(esti.Ause.shape[1])
            for iField in range(metr.nField):
                aa = esti.senM[iField, :, :]
                Afield = aa[np.ix_(esti.zn3Idx, esti.dofIdx)]
                y2f = self.y2[iField, esti.zn3Idx]
                yf = Afield.dot(esti.xhat[esti.dofIdx]) + y2f
                Mxf = Afield.T.dot(CCmat).dot(yf)
                Mx = Mx + metr.w[iField] * Mxf
            if self.xref == 'x0' or self.xref == 'x0xcor':
                self.uk[esti.dofIdx] = - self.gainUse * self.mF.dot(Mx)
            elif self.xref == '0':
                self.uk[esti.dofIdx] = self.gainUse * self.mF.dot(
                    -self.rho**2 * self.mH.dot(state.stateV[esti.dofIdx]) -
                    Mx)
            elif self.xref == 'x00':
                self.uk[esti.dofIdx] = self.gainUse * self.mF.dot(
                    self.rho**2 * self.mH.dot(state.stateV0[esti.dofIdx] -
                                              state.stateV[esti.dofIdx]) -
                    Mx)

    def drawControlPanel(self, esti, state):

        plt.figure(figsize=(15, 10))

        # rigid body motions
        axm2rig = plt.subplot2grid((4, 4), (0, 0))
        axm2rot = plt.subplot2grid((4, 4), (0, 1))
        axcamrig = plt.subplot2grid((4, 4), (0, 2))
        axcamrot = plt.subplot2grid((4, 4), (0, 3))

        myxticks = [1, 2, 3]
        myxticklabels = ['%d' % (myxticks[i])
                         for i in np.arange(len(myxticks))]
        axm2rig.plot(myxticks, self.uk[[(i - 1)
                                        for i in myxticks]], 'ro', ms=8)
        axm2rig.set_xticks(myxticks)
        axm2rig.set_xticklabels(myxticklabels)
        axm2rig.grid()
        axm2rig.annotate('M2 dz,dx,dy', xy=(0.3, 0.4),
                         xycoords='axes fraction', fontsize=16)
        axm2rig.set_ylabel('um')
        axm2rig.set_xlim(np.min(myxticks) - 0.5, np.max(myxticks) + 0.5)

        myxticks = [4, 5]
        myxticklabels = ['%d' % (myxticks[i])
                         for i in np.arange(len(myxticks))]
        axm2rot.plot(myxticks, self.uk[[(i - 1)
                                        for i in myxticks]], 'ro', ms=8)
        axm2rot.set_xticks(myxticks)
        axm2rot.set_xticklabels(myxticklabels)
        axm2rot.grid()
        axm2rot.annotate('M2 rx,ry', xy=(0.3, 0.4),
                         xycoords='axes fraction', fontsize=16)
        axm2rot.set_ylabel('arcsec')
        axm2rot.set_xlim(np.min(myxticks) - 0.5, np.max(myxticks) + 0.5)

        myxticks = [6, 7, 8]
        myxticklabels = ['%d' % (myxticks[i])
                         for i in np.arange(len(myxticks))]
        axcamrig.plot(myxticks, self.uk[[(i - 1)
                                         for i in myxticks]], 'ro', ms=8)
        axcamrig.set_xticks(myxticks)
        axcamrig.set_xticklabels(myxticklabels)
        axcamrig.grid()
        axcamrig.annotate('Cam dz,dx,dy', xy=(0.3, 0.4),
                          xycoords='axes fraction', fontsize=16)
        axcamrig.set_ylabel('um')
        axcamrig.set_xlim(np.min(myxticks) - 0.5, np.max(myxticks) + 0.5)

        myxticks = [9, 10]
        myxticklabels = ['%d' % (myxticks[i])
                         for i in np.arange(len(myxticks))]
        axcamrot.plot(myxticks, self.uk[[(i - 1)
                                         for i in myxticks]], 'ro', ms=8)
        axcamrot.set_xticks(myxticks)
        axcamrot.set_xticklabels(myxticklabels)
        axcamrot.grid()
        axcamrot.annotate('Cam rx,ry', xy=(0.3, 0.4),
                          xycoords='axes fraction', fontsize=16)
        axcamrot.set_ylabel('arcsec')
        axcamrot.set_xlim(np.min(myxticks) - 0.5, np.max(myxticks) + 0.5)

        # m13 and m2 bending
        axm13 = plt.subplot2grid((4, 4), (1, 0), colspan=2)
        axm2 = plt.subplot2grid((4, 4), (1, 2), colspan=2)

        myxticks = range(1, esti.nB13Max + 1)
        myxticklabels = ['%d' % (myxticks[i])
                         for i in np.arange(len(myxticks))]
        axm13.plot(myxticks, self.uk[[(i - 1 + 10)
                                      for i in myxticks]], 'ro', ms=8)
        axm13.set_xticks(myxticks)
        axm13.set_xticklabels(myxticklabels)
        axm13.grid()
        axm13.annotate('M1M3 bending', xy=(0.3, 0.4),
                       xycoords='axes fraction', fontsize=16)
        axm13.set_ylabel('um')
        axm13.set_xlim(np.min(myxticks) - 0.5, np.max(myxticks) + 0.5)

        myxticks = range(1, esti.nB2Max + 1)
        myxticklabels = ['%d' % (myxticks[i])
                         for i in np.arange(len(myxticks))]
        axm2.plot(myxticks, self.uk[[(
            i - 1 + 10 + esti.nB13Max) for i in myxticks]], 'ro', ms=8)
        axm2.set_xticks(myxticks)
        axm2.set_xticklabels(myxticklabels)
        axm2.grid()
        axm2.annotate('M2 bending', xy=(0.3, 0.4),
                      xycoords='axes fraction', fontsize=16)
        axm2.set_ylabel('um')
        axm2.set_xlim(np.min(myxticks) - 0.5, np.max(myxticks) + 0.5)

        # OPD zernikes before and after the FULL correction
        # it goes like
        #  2 1
        #  3 4
        axz1 = plt.subplot2grid((4, 4), (2, 2), colspan=2)
        axz2 = plt.subplot2grid((4, 4), (2, 0), colspan=2)
        axz3 = plt.subplot2grid((4, 4), (3, 0), colspan=2)
        axz4 = plt.subplot2grid((4, 4), (3, 2), colspan=2)

        z4up = range(4, esti.znMax + 1)
        axz1.plot(z4up, esti.yfinal[:esti.zn3Max],
                  label='iter %d' % (state.iIter - 1),
                  marker='*', color='b', markersize=10)
        axz1.plot(z4up, esti.yresi[:esti.zn3Max],
                  label='if full correction applied',
                  marker='*', color='r', markersize=10)
        axz1.grid()
        axz1.annotate('Zernikes R44', xy=(0.3, 0.4),
                      xycoords='axes fraction', fontsize=16)
        axz1.set_ylabel('um')
        axz1.legend(loc="best", shadow=True, fancybox=True)
        axz1.set_xlim(np.min(z4up) - 0.5, np.max(z4up) + 0.5)

        axz2.plot(z4up, esti.yfinal[esti.zn3Max:2 * esti.zn3Max],
                  marker='*', color='b', markersize=10)
        axz2.plot(z4up, esti.yresi[esti.zn3Max:2 * esti.zn3Max],
                  marker='*', color='r', markersize=10)
        axz2.grid()
        axz2.annotate('Zernikes R40', xy=(0.3, 0.4),
                      xycoords='axes fraction', fontsize=16)
        axz2.set_ylabel('um')
        axz2.set_xlim(np.min(z4up) - 0.5, np.max(z4up) + 0.5)

        axz3.plot(z4up, esti.yfinal[2 * esti.zn3Max:3 * esti.zn3Max],
                  marker='*', color='b', markersize=10)
        axz3.plot(z4up, esti.yresi[2 * esti.zn3Max:3 * esti.zn3Max],
                  marker='*', color='r', markersize=10)
        axz3.grid()
        axz3.annotate('Zernikes R00', xy=(0.3, 0.4),
                      xycoords='axes fraction', fontsize=16)
        axz3.set_ylabel('um')
        axz3.set_xlim(np.min(z4up) - 0.5, np.max(z4up) + 0.5)

        axz4.plot(z4up, esti.yfinal[3 * esti.zn3Max:4 * esti.zn3Max],
                  marker='*', color='b', markersize=10)
        axz4.plot(z4up, esti.yresi[3 * esti.zn3Max:4 * esti.zn3Max],
                  marker='*', color='r', markersize=10)
        axz4.grid()
        axz4.annotate('Zernikes R04', xy=(0.3, 0.4),
                      xycoords='axes fraction', fontsize=16)
        axz4.set_ylabel('um')
        axz4.set_xlim(np.min(z4up) - 0.5, np.max(z4up) + 0.5)

        plt.tight_layout()

        # plt.show()
        pngFile = '%s/iter%d/sim%d_iter%d_ctrl.png' % (
            state.pertDir, state.iIter, state.iSim, state.iIter)
        plt.savefig(pngFile, bbox_inches='tight')
        plt.close()

    def drawSummaryPlots(self, state, metr, esti, M1M3, M2,
                         startIter, endIter, debugLevel):
        allPert = np.zeros((esti.ndofA, endIter - startIter + 1))
        allPSSN = np.zeros((metr.nField + 1, endIter - startIter + 1))
        allFWHMeff = np.zeros((metr.nField + 1, endIter - startIter + 1))
        alldm5 = np.zeros((metr.nField + 1, endIter - startIter + 1))
        allelli = np.zeros((metr.nField + 1, endIter - startIter + 1))
        allseeing = np.zeros((endIter - startIter + 1))
        allseeingvk = np.zeros((endIter - startIter + 1))
        for iIter in range(startIter, endIter + 1):
            filename = state.pertMatFile.replace(
                'iter%d' % endIter, 'iter%d' % iIter)
            allPert[:, iIter - startIter] = np.loadtxt(filename)
            filename = metr.PSSNFile.replace(
                'iter%d' % endIter, 'iter%d' % iIter)
            allData = np.loadtxt(filename)
            allPSSN[:, iIter - startIter] = allData[0, :]
            allFWHMeff[:, iIter - startIter] = allData[1, :]
            alldm5[:, iIter - startIter] = allData[2, :]
            filename = metr.elliFile.replace(
                'iter%d' % endIter, 'iter%d' % iIter)
            allelli[:, iIter - startIter] = np.loadtxt(filename)

            filename = state.atmFile[0].replace(
                'iter%d' % endIter, 'iter%d' % iIter)
            seeingdata = np.loadtxt(filename, skiprows =1)
            w = seeingdata[:,1]
            # according to John, seeing = quadrature sum (each layer)
            allseeing[iIter - startIter] = np.sqrt(np.sum(w**2))*\
              2*np.sqrt(2*np.log(2)) #convert sigma into FWHM
            # according to John, weight L0 using seeing^2
            L0eff =  np.sum(seeingdata[:,2]*w**2) /np.sum(w**2)
            r0_500 = 0.976*0.5e-6/(allseeing[iIter - startIter]/3600/180*np.pi)
            r0 = r0_500*(state.wavelength/0.5)**1.2
            allseeingvk[iIter - startIter] = 0.976*state.wavelength*1e-6\
              /r0*np.sqrt(1-2.183*(r0/L0eff)**0.356)\
              /np.pi*180*3600

        f, ax = plt.subplots(3, 3, figsize=(15, 10))
        myxticks = np.arange(startIter, endIter + 1)
        myxticklabels = ['%d' % (myxticks[i])
                         for i in np.arange(len(myxticks))]
        colors = ('r', 'b', 'g', 'c', 'm', 'y', 'k')

        # 1: M2, cam dz
        ax[0, 0].plot(myxticks, allPert[0, :], label='M2 dz',
                      marker='.', color='r', markersize=10)
        ax[0, 0].plot(myxticks, allPert[5, :], label='Cam dz',
                      marker='.', color='b', markersize=10)
        ax[0, 0].set_xlim(np.min(myxticks) - 0.5, np.max(myxticks) + 0.5)
        ax[0, 0].set_xticks(myxticks)
        ax[0, 0].set_xticklabels(myxticklabels)
        ax[0, 0].set_xlabel('iteration')
        ax[0, 0].set_ylabel('$\mu$m')
        ax[0, 0].set_title('M2 %d/$\pm$%d$\mu$m; Cam %d/$\pm$%d$\mu$m' % (
            round(np.max(np.absolute(allPert[0, :]))), self.range[0],
            round(np.max(np.absolute(allPert[5, :]))), self.range[5]))
        # , shadow=True, fancybox=True)
        leg = ax[0, 0].legend(loc="lower left")
        leg.get_frame().set_alpha(0.5)

        # 2: M2, cam dx,dy
        ax[0, 1].plot(myxticks, allPert[1, :], label='M2 dx',
                      marker='.', color='r', markersize=10)
        ax[0, 1].plot(myxticks, allPert[2, :], label='M2 dy',
                      marker='*', color='r', markersize=10)
        ax[0, 1].plot(myxticks, allPert[6, :], label='Cam dx',
                      marker='.', color='b', markersize=10)
        ax[0, 1].plot(myxticks, allPert[7, :], label='Cam dy',
                      marker='*', color='b', markersize=10)
        ax[0, 1].set_xlim(np.min(myxticks) - 0.5, np.max(myxticks) + 0.5)
        ax[0, 1].set_xticks(myxticks)
        ax[0, 1].set_xticklabels(myxticklabels)
        ax[0, 1].set_xlabel('iteration')
        ax[0, 1].set_ylabel('$\mu$m')
        ax[0, 1].set_title('M2 %d/$\pm$%d$\mu$m; Cam %d/$\pm$%d$\mu$m' % (
            round(np.max(np.absolute(allPert[1:3, :]))), self.range[1],
            round(np.max(np.absolute(allPert[6:8, :]))), self.range[6]))
        # , shadow=True, fancybox=True)
        leg = ax[0, 1].legend(loc="lower left")
        leg.get_frame().set_alpha(0.5)

        # 3: M2, cam rx,ry
        ax[0, 2].plot(myxticks, allPert[3, :], label='M2 rx',
                      marker='.', color='r', markersize=10)
        ax[0, 2].plot(myxticks, allPert[4, :], label='M2 ry',
                      marker='*', color='r', markersize=10)
        ax[0, 2].plot(myxticks, allPert[8, :], label='Cam rx',
                      marker='.', color='b', markersize=10)
        ax[0, 2].plot(myxticks, allPert[9, :], label='Cam ry',
                      marker='*', color='b', markersize=10)
        ax[0, 2].set_xlim(np.min(myxticks) - 0.5, np.max(myxticks) + 0.5)
        ax[0, 2].set_xticks(myxticks)
        ax[0, 2].set_xticklabels(myxticklabels)
        ax[0, 2].set_xlabel('iteration')
        ax[0, 2].set_ylabel('arcsec')
        ax[0, 2].set_title('M2 %d/$\pm$%darcsec; Cam %d/$\pm$%darcsec' % (
            round(np.max(np.absolute(allPert[3:5, :]))), self.range[3],
            round(np.max(np.absolute(allPert[8:10, :]))), self.range[8]))
        # , shadow=True, fancybox=True)
        leg = ax[0, 2].legend(loc="lower left")
        leg.get_frame().set_alpha(0.5)

        # 4: M1M3 bending
        rms = np.std(allPert[10:esti.nB13Max + 10, :], axis=1)
        idx = np.argsort(rms)
        for i in range(1, 4 + 1):
            ax[1, 0].plot(myxticks, allPert[idx[-i] + 10, :],
                          label='M1M3 b%d' %
                          (idx[-i] + 1), marker='.', color=colors[i - 1],
                          markersize=10)
        for i in range(4, esti.nB13Max + 1):
            ax[1, 0].plot(myxticks, allPert[idx[-i] + 10, :],
                          marker='.', color=colors[-1], markersize=10)
        ax[1, 0].set_xlim(np.min(myxticks) - 0.5, np.max(myxticks) + 0.5)
        ax[1, 0].set_xticks(myxticks)
        ax[1, 0].set_xticklabels(myxticklabels)
        ax[1, 0].set_xlabel('iteration')
        ax[1, 0].set_ylabel('$\mu$m')
        allF = M1M3.force[:, :esti.nB13Max].dot(
            allPert[10:esti.nB13Max + 10, :])
        stdForce = np.std(allF, axis=0)
        maxForce = np.max(allF, axis=0)
        ax[1, 0].set_title('Max %d/$\pm$%dN; RMS %dN' % (
            round(np.max(maxForce)), round(self.range[0] / self.rhoM13),
            round(np.max(stdForce))))
        # , shadow=True, fancybox=True)
        leg = ax[1, 0].legend(loc="lower left")
        leg.get_frame().set_alpha(0.5)

        # 5: M2 bending
        rms = np.std(allPert[10 + esti.nB13Max:esti.ndofA, :], axis=1)
        idx = np.argsort(rms)
        for i in range(1, 4 + 1):
            ax[1, 1].plot(myxticks, allPert[idx[-i] + 10 + esti.nB13Max, :],
                          label='M2 b%d' %
                          (idx[-i] + 1), marker='.', color=colors[i - 1],
                          markersize=10)
        for i in range(4, esti.nB2Max + 1):
            ax[1, 1].plot(myxticks, allPert[idx[-i] + 10 + esti.nB13Max,
                                            :], marker='.', color=colors[-1],
                          markersize=10)
        ax[1, 1].set_xlim(np.min(myxticks) - 0.5, np.max(myxticks) + 0.5)
        ax[1, 1].set_xticks(myxticks)
        ax[1, 1].set_xticklabels(myxticklabels)
        ax[1, 1].set_xlabel('iteration')
        ax[1, 1].set_ylabel('$\mu$m')
        allF = M2.force[:, :esti.nB2Max].dot(
            allPert[10 + esti.nB13Max:esti.ndofA, :])
        stdForce = np.std(allF, axis=0)
        maxForce = np.max(allF, axis=0)
        ax[1, 1].set_title('Max %d/$\pm$%dN; RMS %dN' % (
            round(np.max(maxForce)), round(self.range[0] / self.rhoM2),
            round(np.max(stdForce))))
        # , shadow=True, fancybox=True)
        leg = ax[1, 1].legend(loc="lower left")
        leg.get_frame().set_alpha(0.5)

        # 6: PSSN
        for i in range(metr.nField):
            ax[1, 2].semilogy(myxticks, 1 - allPSSN[i, :],
                              marker='.', color='b', markersize=10)
        ax[1, 2].semilogy(myxticks, 1 - allPSSN[-1, :],
                          label='GQ(1-PSSN)',
                          marker='.', color='r', markersize=10)
        ax[1, 2].set_xlim(np.min(myxticks) - 0.5, np.max(myxticks) + 0.5)
        ax[1, 2].set_xticks(myxticks)
        ax[1, 2].set_xticklabels(myxticklabels)
        ax[1, 2].set_xlabel('iteration')
        # ax[1, 2].set_ylabel('um')
        ax[1, 2].grid()
        if allPSSN.shape[1] > 1:
            ax[1, 2].set_title('Last 2 PSSN: %5.3f, %5.3f' %
                               (allPSSN[-1, -2], allPSSN[-1, -1]))
        else:
            ax[1, 2].set_title('Last PSSN: %5.3f' % (allPSSN[-1, -1]))

        # , shadow=True, fancybox=True)
        leg = ax[1, 2].legend(loc="upper right")
        leg.get_frame().set_alpha(0.5)

        # 7: FWHMeff
        if debugLevel>-1:
            for i in range(metr.nField):
                ax[2, 0].plot(myxticks, allFWHMeff[i, :],
                            marker='.', color='b', markersize=10)
        ax[2, 0].plot(myxticks, allFWHMeff[-1, :],
                      label='GQ($FWHM_{eff}$)',
                      marker='.', color='r', markersize=10)
        ax[2, 0].plot(myxticks, allseeingvk,label='seeing',
                          marker='.', color='g', markersize=10)
        xmin = np.min(myxticks) - 0.5
        xmax = np.max(myxticks) + 0.5
        ax[2, 0].set_xlim([xmin, xmax])
        ax[2, 0].set_xticks(myxticks)
        ax[2, 0].set_xticklabels(myxticklabels)
        ax[2, 0].set_xlabel('iteration')
        ax[2, 0].set_ylabel('arcsec')
        ax[2, 0].grid()
        ax[2, 0].plot([xmin, xmax], state.iqBudget *
                      np.ones((2, 1)), label='Error Budget', color='k')
        if debugLevel == -1:
            ax[2, 0].set_title('$FWHM_{eff}$')
        else:
            if allFWHMeff.shape[1] > 1:
                ax[2, 0].set_title('Last 2 $FWHM_{eff}$: %5.3f, %5.3f arcsec' % (
                    allFWHMeff[-1, -2], allFWHMeff[-1, -1]))
            else:
                ax[2, 0].set_title(
                    'Last $FWHM_{eff}$: %5.3f arcsec' % (allFWHMeff[-1, -1]))
        # , shadow=True, fancybox=True)
        leg = ax[2, 0].legend(loc="upper right")
        leg.get_frame().set_alpha(0.5)

        # 8: dm5
        for i in range(metr.nField):
            ax[2, 1].plot(myxticks, alldm5[i, :], marker='.',
                          color='b', markersize=10)
        ax[2, 1].plot(myxticks, alldm5[-1, :], label='GQ($\Delta$m5)',
                      marker='.', color='r', markersize=10)
        ax[2, 1].set_xlim(np.min(myxticks) - 0.5, np.max(myxticks) + 0.5)
        ax[2, 1].set_xticks(myxticks)
        ax[2, 1].set_xticklabels(myxticklabels)
        ax[2, 1].set_xlabel('iteration')
        # ax[2, 1].set_ylabel('arcsec')
        ax[2, 1].grid()
        if alldm5.shape[1] > 1:
            ax[2, 1].set_title('Last 2 $\Delta$m5: %5.3f, %5.3f' %
                               (alldm5[-1, -2], alldm5[-1, -1]))
        else:
            ax[2, 1].set_title('Last $\Delta$m5: %5.3f' % (alldm5[-1, -1]))
        # , shadow=True, fancybox=True)
        leg = ax[2, 1].legend(loc="upper right")
        leg.get_frame().set_alpha(0.5)

        # 9: elli
        if debugLevel>-1:
            for i in range(metr.nField):
                ax[2, 2].plot(myxticks, allelli[i, :] * 100,
                            marker='.', color='b', markersize=10)
        ax[2, 2].plot(myxticks, allelli[-1, :] * 100,
                      label='GQ(ellipticity)',
                      marker='.', color='r', markersize=10)
        ax[2, 2].set_xlim(np.min(myxticks) - 0.5, np.max(myxticks) + 0.5)
        ax[2, 2].set_xticks(myxticks)
        ax[2, 2].set_xticklabels(myxticklabels)
        ax[2, 2].set_xlabel('iteration')
        ax[2, 2].set_ylabel('percent')
        ax[2, 2].plot([xmin, xmax], state.eBudget * 100 #in percent
                      * np.ones((2, 1)), label='SRD Spec (Median)', color='k')
        ax[2, 2].grid()
        if debugLevel == -1:
            ax[2, 2].set_title('Ellipticity')
        else:
            if allelli.shape[1] > 1:
                ax[2, 2].set_title('Last 2 e: %4.2f%%, %4.2f%%' %
                                (allelli[-1, -2] * 100, allelli[-1, -1] * 100))
            else:
                ax[2, 2].set_title('Last 2 e: %4.2f%%' % (allelli[-1, -1] * 100))
            # , shadow=True, fancybox=True)
        leg = ax[2, 2].legend(loc="upper right")
        leg.get_frame().set_alpha(0.5)

        plt.tight_layout()
        # plt.show()

        for i in range(startIter, endIter + 1):
            for j in range(i, endIter + 1):
                sumPlotFile = '%s/sim%d_iter%d-%d.png' % (
                    state.pertDir, state.iSim, i, j)
                if (i == startIter and j == endIter):
                    plt.savefig(sumPlotFile, bbox_inches='tight', dpi=500)
                else:
                    # remove everything else in between startIter and endIter
                    if os.path.isfile(sumPlotFile):
                        os.remove(sumPlotFile)

if __name__ == "__main__":

    pass
