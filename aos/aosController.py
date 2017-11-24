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

    def __init__(self, ctrlDir, instruFile, paramFile, esti, metr, M1M3Force, M2Force,
                 effwave, gain, covM=None, debugLevel=0):
        """

        Initiate the aosController class.

        Arguments:
            ctrlDir {[str]} -- Directory of controller parameter file.
            instruFile {[str]} -- Instrument folder name.
            paramFile {[str]} -- Controller parameter file (*.ctrl) to read.
            esti {[aosEstimator]} -- aosEstimator object.
            metr {[aosMetric]} -- aosMetric object.
            M1M3Force {[ndarray]} -- Actuator force matrix of M1M3.
            M2Force {[ndarray]} -- Actuator force matrix of M2.
            effwave {[float]} -- Effective wavelength in um.
            gain {[float]} -- Gain value feedback in control algorithm.

        Keyword Arguments:
            covM {[ndarray]} -- Covariance matrix of WFS. (default: {None})
            debugLevel {int} -- Debug level. The higher value gives more information.
                                (default: {0})
        """

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

        # Predicted movement at time k
        self.uk = None
        self.CCmat = None

        # Read the file to get the parameters
        self.__readFile(self.filename)

        # Get the instrument name
        instName, defocalOffset = getInstName(instruFile)

        # Read the y2 file of specific instrument. y = A * x + A2 * x2 + w. y2 = A2 * x2
        # x2 is the uncontrolled perturbations.
        # Check with Bo for why all values are zeros. How to handle the real condition?
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
        self.Authority = self.__getAuthorityH(rbW, M1M3Force, M2Force, esti.dofIdx, 
                                              esti.nB13Max, esti.nB2Max)

        # Calculate the range of motion of DOFs, which means there is no truncation of DOF.
        # range = 1/self.Authority*rbStroke[0]
        dofIdxNoTrunc = np.ones(len(esti.dofIdx), dtype=bool)
        authorityNoTrunc = self.__getAuthorityH(rbW, M1M3Force, M2Force, dofIdxNoTrunc, 
                                                esti.nB13Max, esti.nB2Max)
        self.range = 1/authorityNoTrunc*self.rbStroke[0]

        # Decide to modify the sensitivity matrix A based on the strategy or not
        # Check with Bo for this part
        # There should be "crude_opti" and "kalman" in aosEstimator.py strategy.
        if (esti.strategy == "pinv"):
            # Decide to normalize sensitivity matrix A or not
            if (esti.normalizeA):
                esti.normA(self.Authority)
        elif (esti.strategy == "opti"):
            # Decide to add a diagonal perpetubation term in sensitivity matrix A or not.
            if (esti.fmotion > 0):
                esti.optiAinv(self.range, covM)

        if (self.strategy == "optiPSSN"):

            # Calculate the matrix H by diagonalize the array of authority in rms^2 unit
            # This means the square of authority
            self.mH = np.diag(self.Authority**2)

            # Update the matrix H for the strategy of "x0xcor"
            if (self.xref == "x0xcor"):

                # b3 of M1M3 bending, 10 is for the DOF of M2 and camera hexapod
                idx1 = 10 + 3

                # b5 of M2 bending, 20 is for the maximum bending mode of M1M3
                idx2 = 10 + 20 + 5

                # Update mH if M1M3 b3 and M2 b5 are both available
                # Check with Bo for this idea
                if (esti.dofIdx[idx1] and esti.dofIdx[idx2]):

                    # Do not understand this part.
                    # Check with Bo the reason to do this.
                    idx1 = sum(esti.dofIdx[:idx1]) - 1
                    idx2 = sum(esti.dofIdx[:idx2]) - 1

                    # 10 times penalty
                    # Do not understand here. Check with Bo.
                    self.mH[idx1, idx2] = self.Authority[idx1] * self.Authority[idx2] * 100

            # Calcualte the CCmat by diagonalizing the PSSN with unit of um
            # Do not understand the meaning of CCmat. Check with Bo for this.

            # wavelength below in um,b/c output of A in um
            self.CCmat = np.diag(metr.pssnAlpha) * (2 * np.pi / effwave)**2

            # Calculate the matrix Q (q^2 = y.T * Q * y)
            # where q is the image quality metric
            # Notice here: q^2 ~ y.T * Q * y + c^2
            # c is the fitting of PSSN of image quality
            # Check eqs: (3.2 and 3.3) in "Real Time Wavefront Control System for the Large
            # Synoptic Survey Telescope (LSST)"

            # Q = sum_i (w_i * Q_i)
            self.mQ = np.zeros((esti.Ause.shape[1], esti.Ause.shape[1]))

            # nField is the number of field points to describe the image quality on a focal plane.
            # For LSST, it is 1 + 6*5 = 31 field pints.
            for iField in range(metr.nField):

                # Get the sensitivity matrix for the related field point
                # The first dimension of sensitivity matrix is the field point
                senMfield = esti.senM[iField, :, :]

                # Get the Afield based on the available Zk and DOF indexes
                Afield = senMfield[np.ix_(esti.zn3Idx, esti.dofIdx)]

                # Calculate the numerator of F = A.T * Q * A / (A.T * Q * A + rho * H)
                # Q := A.T * Q * A actually
                mQf = Afield.T.dot(self.CCmat).dot(Afield)

                # Consider the weighting of Q = sum_i (w_i * Q_i)
                self.mQ = self.mQ + metr.w[iField] * mQf

            # Calculate the denominator of F = A.T * Q * A / (A.T * Q * A + rho * H)
            # F := (A.T * Q * A + rho * H)^(-1) actually
            # Because the unit is rms^2, the square of rho read from the *.ctrl file is needed.
            self.mF = np.linalg.pinv(self.mQ + self.rho**2 * self.mH)

            # Show the elements of matrix Q for the debug or not
            if (debugLevel >= 3):
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
                if (line.startswith("xref")):
                    self.xref = line.split()[1]

                # The new gain value once the system is stable. That means PSSN is less
                # than the threshold, and the system is judged to have corrected the
                # wavefront error.
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

    def getMotions(self, esti, metr, nWFS=None, state=None):
        """

        Get the offset of degree of freedom (DOF).

        Arguments:
            esti {[aosEstimator]} -- aosEstimator object.
            metr {[aosMetric]} -- aosMetric object.

        Keyword Arguments:
            nWFS {[int]} -- Number of wavefront sensor. (default: {None})
            state {[aosTeleState]} -- aosTeleState object. (default: {None})
    
        Returns:
            [ndarray] -- Predicted offsets to subsystems.
        """

        # Initialize the values.
        uk = np.zeros(esti.ndofA)

        # Gain value to use in the run time
        # Use the higher gain value if the image quality is not good enough.
        gainUse = self.gain
        if (self.shiftGear and (metr.GQFWHMeff > self.shiftGearThres)):
            gainUse = 1

        # Calculate uk = - gain * (xhat + c) based on different strategy
        # For the negative sign, follow:
        # https://confluence.lsstcorp.org/pages/viewpage.action?pageId=64698465

        # Check with Bo for the reason to use "null"
        if (self.strategy == "null"):

            # Calculate y2 = sum_i (w_i * y2f), which i is ith field point
            y2 = np.zeros(sum(esti.zn3Idx))
            for iField in range(metr.nField):
                # Get the avialble y2 values based on the available Zk index
                y2f = self.y2[iField, esti.zn3Idx]
                y2 = y2 + metr.w[iField]*y2f

            # Repeat matrix for all WFSs
            y2c = np.repeat(y2, nWFS)

            # Calculate x = inv(A) * y
            x_y2c = esti.Ainv.dot(y2c)

            # Change the unit back to the correct one because of the normalized A.
            if (esti.normalizeA):
                x_y2c = x_y2c * self.Authority

            # Calculate uk
            uk[esti.dofIdx] = -gainUse * (esti.xhat[esti.dofIdx] + x_y2c)

        elif (self.strategy == "optiPSSN"):

            # Construct the array x = inv(A) * y
            Mx = np.zeros(esti.Ause.shape[1])
            for iField in range(metr.nField):

                # Get the sensitivity matrix in specific field point with available
                # Zk and DOF index
                senMfield = esti.senM[iField, :, :]
                Afield = senMfield[np.ix_(esti.zn3Idx, esti.dofIdx)]

                # Get the avialble y2 values based on the available Zk index
                # There should be a update of y2 before/ after this. Check with Bo for this.
                y2f = self.y2[iField, esti.zn3Idx]

                # Calculate y_{k+1} = A * x_{k} + y2_{k}
                yf = Afield.dot(esti.xhat[esti.dofIdx]) + y2f

                # Calcualte A.T * Q * y_{k+1} and considering the weighting
                Mxf = Afield.T.dot(self.CCmat).dot(yf)
                Mx = Mx + metr.w[iField] * Mxf

            # Estimate the motion/ offset based on the feedback algorithm to use.
            # x_{k+1} = x_{k} + u_{k} + d_{k+1}
            # u_{k+1} = - gain * (A.T * Q * A + pho * H)^(-1) * A.T * Q * y_{k+1}
            # For the negative sign above, follow:
            # https://confluence.lsstcorp.org/pages/viewpage.action?pageId=64698465

            # The offset will only trace the previous one in "_x0" type.
            # The offset will trace the real value and target for 0 in "_0" type.
            # The offset will only trace the relative changes of offset without
            # regarding the real value in "_x00" type.

            # For the notaions in the following equations:
            # F = (A.T * Q * A + pho * H)^(-1)
            # x = A.T * Q * y_{k+1}. It is noted that the weighting is included in x for the simplification

            # Check the idea of "0" and "x00" with Bo. It does not look like the description in
            # https://confluence.lsstcorp.org/pages/viewpage.action?pageId=60948764
            if self.xref in ("x0", "x0xcor"):
                # uk = u_{k+1} = - gain * F * xhat_{k+1}
                uk[esti.dofIdx] = -gainUse * self.mF.dot(Mx)

            elif (self.xref == "0"):
                # uk = gain * F * ( -rho^2 * H * x_{k} - xhat_{k+1} )
                uk[esti.dofIdx] = gainUse * self.mF.dot(-self.rho**2 * self.mH.dot(state.stateV[esti.dofIdx]) - Mx)

            elif (self.xref == "x00"):
                # uk = gain * F * [ rho^2 * H * (x_{0} - x_{k}) - xhat_{k+1} ]
                # Check this one with Bo. state.stateV0 is hard coded to use "iter0_pert.mat", which is zero actually.
                # Based on this point, there is no different between "0" and "x00" here.
                uk[esti.dofIdx] = gainUse * self.mF.dot(
                            self.rho**2 * self.mH.dot(state.stateV0[esti.dofIdx] - state.stateV[esti.dofIdx]) - Mx)

        return uk

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
        rms = np.std(allPert[10: 10 + esti.nB13Max, :], axis=1)
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
            ax[1, 1].plot(myxticks, allPert[idx[-i] + 10 + 20,
                                            :], marker='.', color=colors[-1],
                          markersize=10)
        ax[1, 1].set_xlim(np.min(myxticks) - 0.5, np.max(myxticks) + 0.5)
        ax[1, 1].set_xticks(myxticks)
        ax[1, 1].set_xticklabels(myxticklabels)
        ax[1, 1].set_xlabel('iteration')
        ax[1, 1].set_ylabel('$\mu$m')
        allF = M2.force[:, :esti.nB2Max].dot(
            allPert[10 + 20:esti.ndofA, :])
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

def showSummaryPlots(dataDir, dofRange=None, iSim=0, ndofA=50, nField=31, startIter=0, endIter=5, saveFilePath=None, doWrite=True):

    # Number of iteration
    numOfIter = endIter-startIter+1

    # Data array initialization
    allPert = np.zeros((ndofA, numOfIter))
    allPSSN = np.zeros((nField+1, numOfIter))
    allFWHMeff = np.zeros((nField+1, numOfIter))
    alldm5 = np.zeros((nField+1, numOfIter))
    allelli = np.zeros((nField+1, numOfIter))
    allseeing = np.zeros(numOfIter)
    allseeingvk = np.zeros(numOfIter)

    # Perturbation directory
    pertDir = "pert"

    # Read the data
    for iIter in range(startIter, endIter + 1):

        # Simulation directory
        simDir = "sim%d" % iSim

        # Iteration directory
        iterDir = "iter%d" % iIter

        # Perturbation file name
        fileName = "_".join((simDir, iterDir, pertDir))+".mat"
        filePath = os.path.join(dataDir, pertDir, simDir, iterDir, fileName)
        
        # Read the data
        allPert[:, iIter-startIter] = np.loadtxt(filePath)





    # Draw the control panel to show each subsystem's offset
    plt.figure(figsize=(15, 10))

    # Arrangement of figure
    axM2CamDz = plt.subplot2grid((3, 3), (0, 0))
    axM2CamDxDy = plt.subplot2grid((3, 3), (0, 1))
    axM2CamRxRy = plt.subplot2grid((3, 3), (0, 2))
    axM1M3B = plt.subplot2grid((3, 3), (1, 0))
    axM2B = plt.subplot2grid((3, 3), (1, 1))
    axPSSN = plt.subplot2grid((3, 3), (1, 2))
    axFWHMeff = plt.subplot2grid((3, 3), (2, 0))
    axDm5 = plt.subplot2grid((3, 3), (2, 1))
    axElli = plt.subplot2grid((3, 3), (2, 2))

    # x tick
    xticks = np.arange(startIter, endIter+1)

    # Plot the figures
    # 1: M2, cam dz
    if (dofRange is not None):
        title = "M2 %d/$\pm$%d$\mu$m; Cam %d/$\pm$%d$\mu$m" % (
                round(np.max(np.absolute(allPert[0, :]))), dofRange[0],
                round(np.max(np.absolute(allPert[5, :]))), dofRange[5])
    else:
        title = "M2 %d$\mu$m; Cam %d$\mu$m" % (round(np.max(np.absolute(allPert[0, :]))),
                                               round(np.max(np.absolute(allPert[5, :]))))

    __subsystemFigure(axM2CamDz, index=xticks, data=allPert[0, :], marker="r.-", 
                      label="M2 dz", xlabel="iteration", ylabel="$\mu$m", 
                      title=title, grid=False)
    __subsystemFigure(axM2CamDz, index=xticks, data=allPert[5, :], marker="b.-", 
                      label="Cam dz", grid=False)

    # 2: M2, cam dx,dy
    if (dofRange is not None):
        title = "M2 %d/$\pm$%d$\mu$m; Cam %d/$\pm$%d$\mu$m" % (
                round(np.max(np.absolute(allPert[1:3, :]))), dofRange[1],
                round(np.max(np.absolute(allPert[6:8, :]))), dofRange[6])
    else:
        title = "M2 %d$\mu$m; Cam %d$\mu$m" % (round(np.max(np.absolute(allPert[1:3, :]))),
                                               round(np.max(np.absolute(allPert[6:8, :]))))

    __subsystemFigure(axM2CamDxDy, index=xticks, data=allPert[1, :], marker="r.-", 
                      label="M2 dx", xlabel="iteration", ylabel="$\mu$m", title=title, 
                      grid=False)
    __subsystemFigure(axM2CamDxDy, index=xticks, data=allPert[2, :], marker="r*-", 
                      label="M2 dy", grid=False)
    __subsystemFigure(axM2CamDxDy, index=xticks, data=allPert[6, :], marker="b.-", 
                      label="Cam dx", grid=False)
    __subsystemFigure(axM2CamDxDy, index=xticks, data=allPert[7, :], marker="b*-", 
                      label="Cam dy", grid=False)

    # 3: M2, cam rx,ry

  


    __subsystemFigure(axM2CamRxRy, index=xticks, xlabel="iteration", ylabel="arcsec", title="M2CamRxRy", grid=False)
    __subsystemFigure(axM1M3B, index=xticks, xlabel="iteration", ylabel="$\mu$m", title="M1M3 Bending", grid=False)
    __subsystemFigure(axM2B, index=xticks, xlabel="iteration", ylabel="$\mu$m", title="M2 Bending", grid=False)
    __subsystemFigure(axPSSN, index=xticks, xlabel="iteration", title="PSSN")
    __subsystemFigure(axFWHMeff, index=xticks, xlabel="iteration", ylabel="arcsec", title="FWHM")
    __subsystemFigure(axDm5, index=xticks, xlabel="iteration", title="Dm5")
    __subsystemFigure(axElli, index=xticks, xlabel="iteration", ylabel="percent", title="Elli")

    # Save the image or not
    __saveFig(plt, saveFilePath=saveFilePath, doWrite=doWrite)

def showControlPanel(uk=None, yfinal=None, yresi=None, iterNum=None, saveFilePath=None, doWrite=True):
    """
    
    Plot the figure of degree of freedom for each subsystem and the wavefront error on each wavefront 
    sensor. It is noted that this function has been hardcoded for LSST wavefront sensor to use. Need to 
    update this function for random WFS later.
    
    Keyword Arguments:
        uk {[ndarray]} -- Predicted offset for each subsystem in the basis of degree of freedom. 
                          (default: {None})
        yfinal {[ndarray]} -- Wavefront error in the basis of Zk. (default: {None})
        yresi {[ndarray]} -- Residue of wavefront error in the basis of Zk if full correction is applied. 
                            (default: {None})
        iterNum {[int]} -- Iteration number. (default: {None})
        saveFilePath {[str]} -- File path to save the figure. (default: {None})
        doWrite {bool} -- Write the figure into the file or not. (default: {True})
    """

    # Draw the control panel to show each subsystem's offset
    plt.figure(figsize=(15, 10))

    # Arrangement of figure
    # The following part is hard-coded for LSST WFS actually. Need to update this later.

    # Rigid body motions: piston, x-tilt, y-tilt, x-rotation, y-rotation
    # M2 hexapod: (dz, dx, dy) and (rx, ry)
    axm2rig = plt.subplot2grid((4, 4), (0, 0))
    axm2rot = plt.subplot2grid((4, 4), (0, 1))

    # Camera hexapod: (dz, dx, dy) and (rx, ry)
    axcamrig = plt.subplot2grid((4, 4), (0, 2))
    axcamrot = plt.subplot2grid((4, 4), (0, 3))

    # M1M3 and M2 bending modes
    axm13 = plt.subplot2grid((4, 4), (1, 0), colspan=2)
    axm2 = plt.subplot2grid((4, 4), (1, 2), colspan=2)

    # OPD zernikes before and after the FULL correction
    # it goes like
    #  2 1
    #  3 4
    axz1 = plt.subplot2grid((4, 4), (2, 2), colspan=2)
    axz2 = plt.subplot2grid((4, 4), (2, 0), colspan=2)
    axz3 = plt.subplot2grid((4, 4), (3, 0), colspan=2)
    axz4 = plt.subplot2grid((4, 4), (3, 2), colspan=2)

    # Calculate the terms of annular Zernike polynomials
    termZk = None
    if (yfinal is not None):
        termZk = int(len(yfinal)/4)
    elif (yresi is not None):
        termZk = int(len(yfinal)/4)

    # Plot the degree of freedom for each subsystem
    __subsystemFigure(axm2rig, xticksStart=1, index=range(0,3), data=uk, marker="ro", 
                      annotation="M2 dz, dx, dy", ylabel="$\mu$m")
    __subsystemFigure(axm2rot, xticksStart=4, index=range(3,5), data=uk, marker="ro", 
                      annotation="M2 rx, ry", ylabel="arcsec")
    __subsystemFigure(axcamrig, xticksStart=6, index=range(5,8), data=uk, marker="ro", 
                      annotation="Cam dz, dx, dy", ylabel="$\mu$m")
    __subsystemFigure(axcamrot, xticksStart=9, index=range(8,10), data=uk, marker="ro", 
                      annotation="Cam rx, ry", ylabel="arcsec")
    __subsystemFigure(axm13, xticksStart=1, index=range(10,30), data=uk, marker="ro", 
                      annotation="M1M3 bending", ylabel="$\mu$m")
    __subsystemFigure(axm2, xticksStart=1, index=range(30,50), data=uk, marker="ro", 
                      annotation="M2 bending", ylabel="$\mu$m")

    # Plot the wavefront error
    subPlotList = [axz1, axz2, axz3, axz4]
    annotationList = ["Zernikes R44", "Zernikes R40", "Zernikes R00", "Zernikes R04"]

    # Plot the final wavefront error in the basis of Zk
    if ((yfinal is not None) and (termZk is not None)):
        label = None
        if (iterNum is not None):
            label = "iter %d" % (iterNum-1)
        __wavefrontFigure(subPlotList, annotationList, yfinal, termZk, marker="*b-", 
                          xticksStart=4, label=label)

    # Plot the residue of wavefront error if full correction of wavefront error is applied
    # This is for the performance prediction only
    if ((yresi is not None) and (termZk is not None)):
        label = "if full correction applied"
        __wavefrontFigure(subPlotList, annotationList, yresi, termZk, marker="*r-", 
                          xticksStart=4, label=label)

    # Save the image or not
    __saveFig(plt, saveFilePath=saveFilePath, doWrite=doWrite)

def __saveFig(plotFig, saveFilePath=None, doWrite=True):
    """
    
    Save the figure to specific path or just show the figure.
    
    Arguments:
        plotFig {[matplotlib.pyplot]} -- Pyplot figure object.
    
    Keyword Arguments:
        saveFilePath {[str]} -- File path to save the figure. (default: {None})
        doWrite {bool} -- Write the figure into the file or not. (default: {True})
    """

    if (doWrite):
        if (saveFilePath):
            
            # Adjust the space between xlabel and title for neighboring sub-figures
            plotFig.tight_layout()

            # Save the figure to file
            plotFig.savefig(saveFilePath, bbox_inches="tight")

            # Close the figure
            plotFig.close()
    else:
        # Show the figure only
        plotFig.show()

def __wavefrontFigure(subPlotList, annotationList, wavefront, termZk, marker="b", 
                      xticksStart=None, label=None):
    """
    
    Plot the wavefront error in the basis of annular Zk for each wavefront sensor (WFS).
    
    Arguments:
        subPlotList {[list]} -- The list of subplots of WFS.
        annotationList {[list]} -- The annotation list of WFS. The idea is to use the name of WFS.
        wavefront {[ndarray]} -- Wavefront error in the basis of annular Zk.
        termZk {[int]} -- Number of terms of annular Zk.
    
    Keyword Arguments:
        marker {str} -- Maker of data point. (default: {"b"})
        xticksStart {[float]} -- x-axis start sticks. (default: {None})
        label {[str]} -- Label of data. (default: {None})
    
    Raises:
        RuntimeError -- The lengths of subPlotList and nameList do not match.
    """

    if (len(subPlotList) != len(annotationList)):
        raise RuntimeError("The lengths of subPlotList and nameList do not match.")

    for ii in range(len(subPlotList)):
        __subsystemFigure(subPlotList[ii], xticksStart=xticksStart, 
                          index=range(ii*int(termZk), (ii+1)*int(termZk)), data=wavefront, 
                          annotation=annotationList[ii], ylabel="um", marker=marker, label=label)

def __subsystemFigure(subPlot, xticksStart=None, index=None, data=None, marker="b", 
                      annotation=None, xlabel=None, ylabel=None, label=None, title=None,
                      grid=True):
    """
    
    Sublplot the figure of evaluated offset (uk) for each subsystem (M2 haxapod, camera hexapod, 
    M1M3 bending mode, M2 bending mode).
    
    Arguments:
        subPlot {[matplotlib.axes]} -- Subplot of subsystem.
    
    Keyword Arguments:
        xticksStart {[float]} -- x-axis start sticks. (default: {None})
        index {[list]} -- Index of values needed in the data. (default: {None})
        data {[list]} -- Data to show. (default: {None})
        marker {str} -- Marker of plot. (default: {"b"})
        annotation {[str]} -- Annotation put in figure. (default: {None})
        xlabel {[str]} -- Label in x-axis. (default: {None})
        ylabel {[str]} -- Label in y-axis. (default: {None})
        label {[str]} -- Label of plot. (default: {None})
        title {[str]} -- Title of plot. (default: {None})
        grid {[bool]} -- Show the grid or not. (default: {True})
    """

    # Get the data
    if ((index is not None) and (data is not None)):
        # Plot the figure
        subPlot.plot(index, data[index], marker, ms=8, label=label)

    # Set x_ticks
    if (index is not None):
        subPlot.set_xticks(index)
        subPlot.set_xlim(np.min(index) - 0.5, np.max(index) + 0.5)

        # Label the x ticks
        if (xticksStart is not None):
            # Shift the labeling
            index = [ii-index[0]+xticksStart for ii in index]

        xticklabels = [str(ii) for ii in index]
        subPlot.set_xticklabels(xticklabels)

    # Set the x lable
    if (xlabel is not None):
        subPlot.set_xlabel(xlabel)

    # Set the y label
    if (ylabel is not None):
        subPlot.set_ylabel(ylabel)

    # Set the annotation
    if (annotation is not None):
        subPlot.annotate(annotation, xy=(0.3, 0.4), xycoords="axes fraction", fontsize=16)

    # Set the legend
    if (label is not None):
        subPlot.legend(loc="best", shadow=False, fancybox=True)

    # Set the title
    if (title is not None):
        subPlot.set_title(title)

    # Set the grid
    if (grid):
        subPlot.grid()

if __name__ == "__main__":

    # uk = np.arange(50)
    # yfinal = np.arange(19*4)
    # iterNum = 5
    # yresi = np.arange(19*4)*3

    dofRange = np.arange(0,51)*1e3

    dataDir = "/Users/Wolf/Documents/aosOutput"
    saveFilePath = "/Users/Wolf/Desktop/temp.png"
    showSummaryPlots(dataDir, dofRange=None, iSim=6, startIter=0, endIter=5, saveFilePath=saveFilePath, doWrite=True)
