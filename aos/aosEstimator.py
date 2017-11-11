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

        self.yfinal = None
        self.yresi = None

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

        # Reshape the sensitivity matrix
        
        # There are 35 field points. 35 = 1 + 6*5 + 4 --> This is to map all focal plane.
        # The reference is at page 16, figure 12 in "An Integrated Modeling Framework for 
        # the Large Synoptic Survey Telescope (LSST)".

        # For LSST, the shape of senM is (35, 19, 50)
        self.senM = self.senM.reshape((-1, self.zn3Max, self.ndofA))

        # The arrangement of senM is [(M2 hexapod), (Camera haxapod), (M1M3), (M2)]. 
        # It should be range(10 + intended M1M3) + range(10 + max M1M3, 10 + max M1M3 + intended M2)
        # max M1M3 index length is 20 here
        self.senM = self.senM[:, :, np.concatenate( (range(10 + self.nB13Max), 
                            range(30, 30 + self.nB2Max)))]
        
        # Show the strategy and shape of sensitivity matrix or not
        if (debugLevel >= 3):
            print(self.strategy)
            print(self.senM.shape)

        # Get the sensitivity matrix of WFS. It is 4 for LSST and 9 for ComCam.
        # It is noted that for LSST and ComCam, different sensitivity matrix files will be used.
        # This is why the index of "-wfs.nWFS" can be used directly.
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
        self.Anorm = self.Ause.copy()
        
        # Show the information of sensitivity matrix A or not
        if (debugLevel >= 3):
            print("---checking Anorm (actually Ause):")
            print(self.Anorm[:5, :5])
        
        # Get the pseudo-inverse matrix of A
        if (self.strategy == "pinv"):
            # Do the truncation if needed. 
            # This command needs more study to check how to decide the inf number.
            self.Ainv = pinv_truncate(self.Anorm, self.nSingularInf)
        
        # "opti" = "optimal estimator"
        elif self.strategy in ("opti", "kalman"):

            # Empirical estimates (by Doug M.), not used when self.fmotion<0
            aa = [0.5, 2, 2, 0.1, 0.1, 0.5, 2, 2, 0.1, 0.1]
            dX = np.concatenate((aa, 0.01 * np.ones(20), 0.005 * np.ones(20)))**2
            X = np.diag(dX)

            if (self.strategy == "opti"):

                # A^(-1) = X * A.T * ( A * X * A.T + M )^(-1)
                # M is W in paper actually.
                # There is another optiAinv() with different X 
                # and fluctuation motion.
                self.Ainv = X.dot(self.Anorm.T).dot(np.linalg.pinv(self.Anorm.dot(X).dot(self.Anorm.T) + wfs.covM))
            
            # The ref. paper of Kalman filter is "An Introduction to the Kalman Filter" 
            # by Greg Welch and Gary Bishop at 2006
            # https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf
            elif (self.strategy == "kalman"):

                # Refactor here by following the paper
                self.P = np.zeros((self.ndofA, self.ndofA))
                self.Q = X
                self.R = wfs.covM*100
                
        elif (self.strategy == "crude_opti"):
            # A^(-1) = A.T * (A * A.T + perturbation * I)^(-1)
            # This perturbation looks like to remove the problem of near-degeneracy.
            # There is the left/ right inverse problems here.
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

                # Number of singular values to set to infinity
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
            
    def normA(self, authority):
        """
        
        Normalize the sensitivity matrix A based on the authority of each degree of freedom (DOF).
        
        Arguments:
            authority {[ndarray]} -- Authority array for each DOF.
        """

        # This is to avoid the unit difference to define the authority. The unit might affect 
        # the pseudo-inverse numerical calculation. This is to avoid the condition such as 
        # nm and um difference. 
        # This needs more study to be the baseline. 
        
        # Normalize the element of A based on the level of authority
        # The dimension of Ause is n x m. The authority array is 1 x m after the reshape.
        # Numpy supports this statement for matrix element multiplication.
        self.Anorm = self.Ause * authority.reshape((1, -1))

        # Get the pseudo-inversed sensitivity matrix A with the truncation
        self.Ainv = pinv_truncate(self.Anorm, self.nSingularInf)

    def optiAinv(self, ctrlRange, covM):
        """
        
        Construct the pseudo-inversed sensitivity matrix A in the stratege of opti.esti with the fmotion 
        that is greater than zero.  
        
        Arguments:
            ctrlRange {[ndarray]} -- Range of controller for each degree of freedom.
            covM {[ndarray]} -- Covariance matrix of wavefront sensor.
        """

        # Additional scope for the optimization. This needs more study to check the reliability.

        # Construct the X matrix
        dX = (ctrlRange * self.fmotion)**2
        X = np.diag(dX)

        # A^(-1) = X * A.T * ( A * X * A.T + M )^(-1)
        self.Ainv = X.dot(self.Anorm.T).dot(np.linalg.pinv(self.Anorm.dot(X).dot(self.Anorm.T) + covM))

    def estimate(self, state, wfs, ctrlY2File, sensor, authority=None):
        """
        
        Estimate the degree of freedom by "x^{hat} = pinv(A) * y". A is the sensitivity matrix 
        and y is the annular Zernike polynomials.
        
        Arguments:
            state {[aosTeleState]} -- AOS telescope state object.
            wfs {[aosWFS]} -- AOS wavefront sensor object.
            ctrlY2File {[str]} -- AOS controller y2 file of specific instrument.
            sensor {[str]} -- Wavefront sensor type ("ideal", "covM", "phosim", "cwfs", "check", "pass").
        
        Keyword Arguments:
            authority {[ndarray]} -- Authority array for each DOF. (default: {None})
        """

        # Get z4-zk and put as y. This data is needed in "x = pinv(A) * y".
        if sensor in ("ideal",  "covM"):

            if (state.nOPDw == 1):
                
                # Read the file of true z4-zk based on the OPD om the previous iteration
                data = np.loadtxt(state.zTrueFile_m1)

                # We only need z4-zk.
                self.yfinal = data[-wfs.nWFS:, 3:self.znMax].reshape((-1, 1))

            else:

                # Collect all z4-zk in the specific band defined in GQwt (Gaussain quadrature weights).
                bb = np.zeros((wfs.znwcs, state.nOPDw))

                for irun in range(state.nOPDw):

                    # Load the data in single run of specific band
                    data = np.loadtxt(state.zTrueFile_m1.replace(".zer", "_w%d.zer" % irun))
                    
                    # Collect the data
                    bb[:, irun] = data[-wfs.nWFS:, 3:self.znMax].reshape((-1, 1))
  
                # There is the problem here. GQwt is a dictionary and bb is a ndarray.
                # At least, use aosTeleState.GQwt[bend] instead.
                # Check this statement again.
                self.yfinal = np.sum(aosTeleState.GQwt * bb)

            if (sensor == "covM"):

                # The centors of random numbers are zero.
                mu = np.zeros(self.zn3Max * 4)

                # Seed the generator. By using this statement, the user can always get 
                # the same random numbers.
                np.random.seed(state.obsID)
                
                # Add the random samples from a multivariate normal distribution.
                self.yfinal += np.random.multivariate_normal(mu, wfs.covM).reshape(-1, 1)
        else:

            # Read the file of z4-zk in the previous iteration
            # [0] for exp No. 0
            data = np.loadtxt(wfs.zFile_m1[0])
            self.yfinal = data[:, :self.zn3Max].reshape((-1, 1))

        # Get rid of the intrinsic WFS error
        self.yfinal -= wfs.intrinsicWFS

        # Subtract y2c.
        # y2c: correction. Zk offset between corner and center.
        y2cData = np.loadtxt(ctrlY2File)

        # Check with Bo for this. What is the meaning of dimension 1 in y2cData?
        y2c = y2cData[-wfs.nWFS:, 0:self.znMax - 3].reshape((-1, 1))

        # Get the zk after the removing of affection from y2c
        z_k = self.yfinal[self.zn3IdxAxnWFS] - y2c

        # Do not sure to keep this part of Kalman filter or not. Check with Bo.
        if (self.strategy == "kalman"):

            # the input to each iteration (in addition to Q and R) :
            #         self.xhat[:, state.iIter - 1]
            #         self.P[:, :, state.iIter - 1]

            # For iIter1, iter0 initialized by estimator
            if (state.iIter>1):
                
                Kalman_xhat_km1_File = "%s/iter%d/sim%d_iter%d_Kalman_xhat.txt" % (
                                            self.pertDir, self.iIter-1, self.iSim, self.iIter-1)
                
                Kalman_P_km1_File = "%s/iter%d/sim%d_iter%d_Kalman_P.txt" % (
                                            self.pertDir, self.iIter-1, self.iSim, self.iIter-1)
                
                # Get the "xhat" and "P" in the previous iteration/ run  
                self.xhat = np.loadtxt(Kalman_xhat_km1_File)
                self.P = np.loadtxt(Kalman_P_km1_File)
            
            # Time update
            xhatminus_k = self.xhat
            Pminus_k = self.P + self.Q
            
            # Measurement update
            K_k = Pminus_k.dot(self.Anorm.T).dot(
                                pinv_truncate(self.Anorm.dot(Pminus_k).dot(self.Anorm.T) + self.R, 5))
            self.xhat[self.dofIdx] = self.xhat[self.dofIdx] + \
                                        K_k.dot(z_k - np.reshape(self.Anorm.dot(xhatminus_k),(-1,1)))
            self.P[np.ix_(self.dofIdx, self.dofIdx)] = (1-K_k.dot(self.Anorm)).dot(Pminus_k)
              
            Kalman_xhat_k_File = '%s/iter%d/sim%d_iter%d_Kalman_xhat.txt' % (
                                        state.pertDir, state.iIter, state.iSim, state.iIter)
            Kalman_P_k_File = '%s/iter%d/sim%d_iter%d_Kalman_P.txt' % (
                                        state.pertDir, state.iIter, state.iSim, state.iIter)

            np.savetxt(Kalman_xhat_k_File, self.xhat)
            np.savetxt(Kalman_P_k_File, self.P)

        else:
        
            # Calculate xhat_{k} = pinv(A) * y_{k}
            # The output of xhat here is the best solution with minimum norm
            # This solution will be used as the input of controller to estimate 
            # the next movement (getMotions()).
            self.xhat[self.dofIdx] = np.reshape(self.Ainv.dot(z_k), -1)
        
            # Put the affection of authority back
            # Anorm = A * authority
            # xhat = pinv(Anorm) * y * authority
            if (self.strategy == "pinv" and self.normalizeA):
                self.xhat[self.dofIdx] = self.xhat[self.dofIdx]*authority
       
        # Define the y residure
        self.yresi = self.yfinal - y2c

        # y_resi := y_resi - A * xhat, where y = A * xhat
        self.yresi += np.reshape(self.Ause.dot(-self.xhat[self.dofIdx]), (-1, 1))

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