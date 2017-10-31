#!/usr/bin/env python
##
# @authors: Bo Xin
# @       Large Synoptic Survey Telescope

import os, re
from glob import glob

import numpy as np
from aos.aosTeleState import aosTeleState

from aos.aosMetric import getInstName

class aosEstimator(object):

    def __init__(self, estiDir, paramFile, instruFile, wfs, icomp=None, izn3=None, debugLevel=0):
        """
        
        Initiate the aosEstimator class.
        
        Arguments:
            estiDir {[str]} -- Directory of estimation parameter file.
            paramFile {[str]} -- Estimator parameter file (*.esti) to read.
            instruFile {[str]} -- Instrument folder name.
            wfs {[aosWFS]} -- AOS wavefront sensor object.
        
        Keyword Arguments:
            icomp {[int]} -- Index of available DOF to use defined in *.esti. (default: {None})
            izn3 {[int]} -- Index of available Zk (>Z3) to use defined in *.esti. (default: {None})
            debugLevel {int} -- Debug level. The higher value gives more information.
                                (default: {0})
        """

        # Name of estimator parameter file
        self.filename = os.path.join(estiDir, (paramFile + ".esti"))

        # Parameters of estimator
        self.strategy = None
        self.nB13Max = None
        self.nB2Max = None
        self.znMax = None
        self.normalizeA = None
        self.nSingularInf = None
        self.fmotion = None
        self.reguMu = None
        self.dofIdx = None
        self.zn3Idx = None

        # Read the file to get the parameters
        self.__readFile(self.filename, icomp, izn3)

        # Nubmer of Zn terms higher than Z3
        self.zn3Max = self.znMax - 3

        # Number of available degree of freedom (DOF) equals 
        # M1M3 bending mode + M2 bending mode + camera hexapod (5) + M2 hexapod (5)
        self.ndofA = self.nB13Max + self.nB2Max + 10

        # Construct the estimation of degree of freedom (x hat)
        # xhat = A.T * (A A.T)^(-1) * y
        self.xhat = np.zeros(self.ndofA)

        # Get the instrument name
        instName, defocalOffset = getInstName(instruFile)

        # Find the instrument sensitivity matrix file
        pathFile = os.path.join(estiDir, instName, "senM*txt")
        src = glob(pathFile)
        self.senMFile = src[0]

        # Show the path of sensitivity matrix or not
        if (debugLevel >= 1):
            print("Using senM file: %s" % self.senMFile)

        # Read the file of sensitivity matrix
        self.senM = np.loadtxt(self.senMFile)

        # Reshape the sensitivity matrix --> Check with Bo for the meaning of first dimension
        # For lsst, the shape of senM is (35, 19, 50)
        self.senM = self.senM.reshape((-1, self.zn3Max, self.ndofA))

        # Need to check the arrangement of sensitivity matrix of M1M3 and M2 with Bo.
        # If the arrangement of senM is [(hexapod dof: 10), (M1M3), (M2)], 
        # it should be range(10 + intended M1M3) + range(10 + max M1M3, 10 + max M1M3 + intended M2)
        self.senM = self.senM[:, :, np.concatenate( (range(10 + self.nB13Max), 
                            range(10 + self.nB13Max, 10 + self.nB13Max + self.nB2Max)))]
        
        # Show the strategy and shape of sensitivity matrix or not
        if (debugLevel >= 3):
            print(self.strategy)
            print(self.senM.shape)

        # Get the sensitivity matrix of WFS. It is 4 for LSST and 9 for ComCam.
        # Need to check with Bo for the meaning of dimension 1 in senM
        A = self.senM[-wfs.nWFS:, :, :].reshape((-1, self.ndofA))

        # Repeat the matrix of Zk by the times of number of WFS 
        self.zn3IdxAxnWFS = np.repeat(self.zn3Idx, wfs.nWFS)
        
        # Construct an open mesh matrix and get the sensitivity matrix A that is in use
        self.Ause = A[np.ix_(self.zn3IdxAxnWFS, self.dofIdx)]

        # Show the information related to the sensitivity matrix or not        
        if (debugLevel >= 3):
            print('---checking estimator related:')
            print(self.dofIdx)
            print(self.zn3Idx)
            print(self.zn3Max)
            print(self.Ause.shape)
            print(self.Ause[21, -1])
            if (self.strategy == "pinv"):
                print(self.normalizeA)

        # Construct the normalized sensitivity matrix A
        # Check with Bo for this statement. It should be self.Anorm = self.Ause.copy()
        # because self.Ause is a ndarray object.
        self.Anorm = self.Ause
        
        # Show the information of sensitivity matrix A or not
        # Check with Bo we need the Anorm to show ot not.
        if (debugLevel >= 3):
            print("---checking Anorm (actually Ause):")
            print(self.Anorm[:5, :5])
            print(self.Ause[:5, :5])
        
        # Get the pseudo-inverse matrix of A
        if (self.strategy == "pinv"):
            # Do the truncation if needed. 
            # Need to check with Bo why not decide the truncation terms automatically
            self.Ainv = pinv_truncate(self.Anorm, self.nSingularInf)
        
        # Check with Bo for these: "opti" and "kalman" and "crude_opti"
        elif self.strategy in ("opti", "kalman"):

            # Empirical estimates (by Doug M.), not used when self.fmotion<0
            aa = [0.5, 2, 2, 0.1, 0.1, 0.5, 2, 2, 0.1, 0.1]
            dX = np.concatenate((aa, 0.01 * np.ones(20), 0.005 * np.ones(20)))**2
            X = np.diag(dX)

            if (self.strategy == "opti"):
                self.Ainv = X.dot(self.Anorm.T).dot(np.linalg.pinv(self.Anorm.dot(X).dot(self.Anorm.T) + wfs.covM))
            
            elif (self.strategy == "kalman"):
                self.P = np.zeros((self.ndofA, self.ndofA))
                self.Q = X
                self.R = wfs.covM*100
                
        elif (self.strategy == "crude_opti"):
            self.Ainv = self.Anorm.T.dot(np.linalg.pinv(self.Anorm.dot(self.Anorm.T) + 
                        self.reguMu * np.identity(self.Anorm.shape[0])))

    def __readFile(self, filePath, icomp, izn3):
        """
        
        Read the AOS estimator parameter file.
        
        Arguments:
            filePath {[str]} -- Path of file of estimator parameters.
            icomp {[int]} -- Decide the available DOFs defined in *.esti file. 
            izn3 {[int]} -- Decide the available Zks defined in *.esti file.
        """

        # Parameters used in reading the file
        iscomment = False
        arrayCountComp = 0
        arrayCountZn3 = 0

        # Read the file
        fid = open(filePath)
        for line in fid:
        
            # Strip the line
            line = line.strip()

            # Ignore the comment part
            if (line.startswith("###")):
                iscomment = ~iscomment
        
            if (not(line.startswith("#")) and (not iscomment) and len(line) > 0):

                # The way to estimate the optical state. (Default: pseudo-inverse)
                if (line.startswith("estimator_strategy")):
                    self.strategy = line.split()[1]

                # Number of bending mode for M1M3
                elif (line.startswith("n_bending_M1M3")):
                    self.nB13Max = int(line.split()[1])

                # Number of bending mode for M2
                elif (line.startswith("n_bending_M2")):
                    self.nB2Max = int(line.split()[1])

                # Maximum number of terms of annular Zernike polynomials
                elif (line.startswith("znmax")):
                    self.znMax = int(line.split()[1])

                # Normalize the sensitivity matrix A or not. (Default: False)
                elif (line.startswith("normalize_A")):
                    self.normalizeA = bool(int(line.split()[1]))

                # Number of singular values to set to infinity --> Need to check with Bo for this
                elif (line.startswith("n_singular_inf")):
                    self.nSingularInf = int(line.split()[1])

                # Average range of motion (in fraction of total range)
                # Only kalman.esti and opti.esti use this. --> Do we need it?
                elif (line.startswith("range_of_motion")):
                    self.fmotion = float(line.split()[1])

                # Only crude_opti.esti uses this. --> Do we need it? And what is this?
                elif (line.startswith("regularization")):
                    self.reguMu = float(line.split()[1])

                # Which line will be used as dofIdx?
                elif (line.startswith("icomp")):

                    # Use the default value in *.esti file
                    if (icomp is None):
                        icomp = int(line.split()[1])

                # Which line will be used as zn3Idx?
                elif (line.startswith("izn3")):

                    # Use the default value in *.esti file
                    if (izn3 is None):
                        izn3 = int(line.split()[1])
                
                # Decide the available index of DOF and Zk
                elif (line.startswith(("0", "1"))):

                    # Get the available index of DOF
                    arrayCountComp, data = self.__getAvailableIdx(line, arrayCountComp, icomp, "icomp")
                    if (data is not None):
                        self.dofIdx = data.astype(bool)

                    # Get the available index of Zk
                    arrayCountZn3, data = self.__getAvailableIdx(line, arrayCountZn3, izn3, "izn3")
                    if (data is not None):
                        self.zn3Idx = data.astype(bool)

        # Close the file                
        fid.close()

    def __getAvailableIdx(self, line, readIdx, targetIdx, arrayType):
        """
        
        Get the availble array data in *.esti file.
        
        Arguments:
            line {[str]} -- Input string.
            readIdx {[int]} -- Read index in specific array type.
            targetIdx {[int]} -- Target index in specific array type.
            arrayType {[str]} -- Specific array type ("icomp" or "izn3").
        
        Returns:
            [int] -- Updated read index in specific array type.
        """

        # Find the groups of repeated pattern
        m = re.findall(r"(\d+)", line)

        # Get the type of line
        aType = None 
        if (len(m) == 10):
            aType = "icomp"
        elif (len(m) == 19):
            aType = "izn3"

        # Compare with the target array type
        if (aType == arrayType):
            readIdx = readIdx + 1

        # Check this is the needed information or not
        data = None
        if (readIdx == targetIdx and aType == arrayType):

            data = np.array([])
            for value in m:
                # Change the value such as "11111" to ["1", "1", "1", "1", "1"]
                value = list(value)
                data = np.append(data, np.array(value))

        return readIdx, data
            
    def normA(self, ctrl):
        self.dofUnit = 1 / ctrl.Authority
        dofUnitMat = np.repeat(self.dofUnit.reshape(
            (1, -1)), self.Ause.shape[0], axis=0)

        self.Anorm = self.Ause / dofUnitMat
        self.Ainv = pinv_truncate(self.Anorm, self.nSingularInf)

    def optiAinv(self, ctrl, wfs):
        dX = (ctrl.range * self.fmotion)**2
        X = np.diag(dX)
        self.Ainv = X.dot(self.Anorm.T).dot(
            np.linalg.pinv(self.Anorm.dot(X).dot(self.Anorm.T) + wfs.covM))

    def estimate(self, state, wfs, ctrl, sensor):
        if sensor == 'ideal' or sensor == 'covM':
            bb = np.zeros((wfs.znwcs, state.nOPDw))
            if state.nOPDw == 1:
                aa = np.loadtxt(state.zTrueFile_m1)
                self.yfinal = aa[-wfs.nWFS:, 3:self.znMax].reshape((-1, 1))
            else:
                for irun in range(state.nOPDw):
                    aa = np.loadtxt(state.zTrueFile_m1.replace('.zer','_w%d.zer'%irun))
                    bb[:, irun] = aa[-wfs.nWFS:, 3:self.znMax].reshape((-1, 1))
                self.yfinal = np.sum(aosTeleState.GQwt * bb)
            if sensor == 'covM':
                mu = np.zeros(self.zn3Max * 4)
                np.random.seed(state.obsID)
                self.yfinal += np.random.multivariate_normal(
                    mu, wfs.covM).reshape(-1, 1)
        else:
            aa = np.loadtxt(wfs.zFile_m1[0]) #[0] for exp No. 0
            self.yfinal = aa[:, :self.zn3Max].reshape((-1, 1))

        self.yfinal -= wfs.intrinsicWFS

        # subtract y2c
        aa = np.loadtxt(ctrl.y2File)
        self.y2c = aa[-wfs.nWFS:, 0:self.znMax - 3].reshape((-1, 1))

        z_k = self.yfinal[self.zn3IdxAxnWFS] - self.y2c
        if self.strategy == 'kalman':
            # the input to each iteration (in addition to Q and R) :
            #         self.xhat[:, state.iIter - 1]
            #         self.P[:, :, state.iIter - 1]

            if state.iIter>1: #for iIter1, iter0 initialized by estimator
                Kalman_xhat_km1_File = '%s/iter%d/sim%d_iter%d_Kalman_xhat.txt' % (
                    self.pertDir, self.iIter-1, self.iSim, self.iIter-1)
                Kalman_P_km1_File = '%s/iter%d/sim%d_iter%d_Kalman_P.txt' % (
                    self.pertDir, self.iIter-1, self.iSim, self.iIter-1)
                self.xhat = np.loadtxt(Kalman_xhat_km1_File)
                self.P = np.loadtxt(Kalman_P_km1_File)
            # time update
            xhatminus_k = self.xhat
            Pminus_k = self.P + self.Q
            # measurement update
            K_k = Pminus_k.dot(self.Anorm.T).dot(
                pinv_truncate(
                    self.Anorm.dot(Pminus_k).dot(self.Anorm.T) + self.R, 5))
            self.xhat[self.dofIdx] = self.xhat[self.dofIdx] + \
              K_k.dot(z_k - np.reshape(self.Anorm.dot(xhatminus_k),(-1,1)))
            self.P[np.ix_(self.dofIdx, self.dofIdx)] = \
              (1-K_k.dot(self.Anorm)).dot(Pminus_k)
              
            Kalman_xhat_k_File = '%s/iter%d/sim%d_iter%d_Kalman_xhat.txt' % (
                state.pertDir, state.iIter, state.iSim, state.iIter)
            Kalman_P_k_File = '%s/iter%d/sim%d_iter%d_Kalman_P.txt' % (
                state.pertDir, state.iIter, state.iSim, state.iIter)
            np.savetxt(Kalman_xhat_k_File, self.xhat)
            np.savetxt(Kalman_P_k_File, self.P)
        else:
            self.xhat[self.dofIdx] = np.reshape(self.Ainv.dot(z_k), [-1])
            if self.strategy == 'pinv' and self.normalizeA:
                self.xhat[self.dofIdx] = self.xhat[self.dofIdx] / self.dofUnit
        self.yresi = self.yfinal.copy()
        self.yresi -= self.y2c
        self.yresi += np.reshape(
            self.Ause.dot(-self.xhat[self.dofIdx]), (-1, 1))

def pinv_truncate(A, n=0):
    """
    
    Get the pseudo-inversed matrix based on the singular value decomposition (SVD) 
    with intended truncation.
    
    Arguments:
        A {[ndarray]} -- Matrix to do the pseudo-inverse.
    
    Keyword Arguments:
        n {int} -- Number of terms to do the truncation in sigma values. (default: {0})
    
    Returns:
        [ndarray] -- Pseudo-inversed matrix.
    """

    # Do the singular value decomposition (A = U * S * V.T)
    Ua, Sa, VaT = np.linalg.svd(A)

    # Get the inverse of sigma
    siginv = 1/Sa
    
    # Do the truncation. The output of Sa is in the decending order. 
    # If there is the near-degenearcy in input matrix, that means the sigma value is closing to zero. 
    # And the inverse of it is closing to infinity.
    # Put such kind of value as 0 to do the truncation.
    # Ref: https://egret.psychol.cam.ac.uk/statistics/local_copies_of_sources_Cardinal_and_Aitken_ANOVA/svd.htm
    if (n > 1):
        siginv[-n:] = 0

    # Construct the inversed sigma matrix by the diagonalization matrix 
    Sainv = np.diag(siginv)

    # Construct the inversed sigma to the correct dimensions.
    Sainv = np.concatenate((Sainv, np.zeros((VaT.shape[0], Ua.shape[0] - Sainv.shape[1]))), axis=1)
   
    # A^(-1) = V * S^(-1) * U.T
    Ainv = VaT.T.dot(Sainv).dot(Ua.T)
   
    return Ainv


if __name__ == "__main__":

    pass