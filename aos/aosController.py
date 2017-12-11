#!/usr/bin/env python
##
# @authors: Bo Xin
# @       Large Synoptic Survey Telescope

import os
from glob import glob
import numpy as np

from aos.aosUtility import getInstName

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
            debugLevel {[int]} -- Debug level. The higher value gives more information.
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

if __name__ == "__main__":

    pass