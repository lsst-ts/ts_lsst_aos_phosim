import os, re, subprocess
import numpy as np
import matplotlib.pyplot as plt

def getInstName(instruFile):
    """
    
    Get the instrument name from the instrument file.
    
    Arguments:
        instruFile {[str]} -- Instrument folder name.
    
    Returns:
        [str] -- Instrument name.
        [float] -- Defocal offset in mm.
    
    Raises:
        RuntimeError -- No instrument found.
    """

    # Get the instrument name
    m = re.match(r"([a-z]+)(?:(\d+))?$", instruFile)
    if m is None:
         raise RuntimeError("Cannot get the instrument name: %s." % instruFile)
    instName = m.groups()[0]

    # Decide the defocal distance offset in mm
    defocalOffset = m.groups()[1]
    if (defocalOffset is not None):
        defocalOffset = float(defocalOffset)/10
    else:
        # Default defocal distance is 1.5 mm
        defocalOffset = 1.5

    return instName, defocalOffset

def pinv_truncate(A, n=0):
    """
    
    Get the pseudo-inversed matrix based on the singular value decomposition (SVD) 
    with intended truncation.
    
    Arguments:
        A {[ndarray]} -- Matrix to do the pseudo-inverse.
    
    Keyword Arguments:
        n {[int]} -- Number of terms to do the truncation in sigma values. (default: {0})
    
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

def showSummaryPlots(dataDir, dofRange=None, iSim=0, ndofA=50, nField=31, 
                     startIter=0, endIter=5, nB13Max=20, nB2Max=20, interestedBend=4, 
                     rhoM13=5.9, M1M3ActForce=None, rhoM2=5.9, M2ActForce=None, wavelength=0.5, 
                     iqBudget=0.2, eBudget=None, dpi=None, saveFilePath=None, doWrite=True, debugLevel=0):
    """
    
    Draw the summary of simulation.
    
    Arguments:
        dataDir {[str]} -- Output files directory.
    
    Keyword Arguments:
        dofRange {[ndarray]} -- Range of each degree of freedom. (default: {None})
        iSim {[int]} -- Simulation number. (default: {0})
        ndofA {[int]} -- Number of degree of freedom. (default: {50})
        nField {[int]} -- Number of field points on camera focal plane. (default: {31})
        startIter {[int]} -- Start number of iteration. (default: {0})
        endIter {[int]} -- End number of iteration. (default: {5})
        nB13Max {[int]} -- Maximum number of bending mode in M1M3 mirror. (default: {20})
        nB2Max {[int]} -- Maximum number of bending mode in M2 mirror. (default: {20})
        interestedBend {[int]} -- Interested starting bending mode of mirrors. (default: {4})
        rhoM13 {[float]} -- Penalty of M1M3 mirror. (default: {5.9})
        M1M3ActForce {[ndaray]} -- Actuator forces of M1M3 mirror. (default: {None})
        rhoM2 {[float]]} -- Penalty of M2 mirror. (default: {5.9})
        M2ActForce {[ndarray]} -- Actuator forces of M2 mirror. (default: {None})
        wavelength {[float]} -- Wavelength in um. (default: {0.5})
        iqBudget {[float]} -- Budget of image quality. (default: {0.2})
        eBudget {[float]} -- Budget of ellipticity. (default: {None})
        dpi {[int]} -- The resolution in dots per inch. (default: {None})
        saveFilePath {[str]} -- File path to save the figure. (default: {None})
        doWrite {[bool]} -- Write the figure into the file or not. (default: {True})
        debugLevel {[int]} -- Debug level. The higher value gives more information. (default: {0})
    """

    # Number of iteration
    numOfIter = endIter-startIter+1

    # Data array initialization
    allPert = np.zeros((ndofA, numOfIter))
    allPSSN = np.zeros((nField+1, numOfIter))
    allFWHMeff = np.zeros((nField+1, numOfIter))
    allDm5 = np.zeros((nField+1, numOfIter))
    allSeeingVk = np.zeros(numOfIter)
    allElli = np.zeros((nField+1, numOfIter))

    # Perturbation directory
    pertDir = "pert"
    imgDir = "image"

    # Read the data
    for iIter in range(startIter, endIter + 1):

        # Simulation directory
        simDir = "sim%d" % iSim

        # Iteration directory
        iterDir = "iter%d" % iIter

        # Perturbation file name
        fileName = "_".join((simDir, iterDir, pertDir)) + ".mat"
        filePath = os.path.join(dataDir, pertDir, simDir, iterDir, fileName)
        
        # Read the perturbation data
        allPert[:, iIter-startIter] = np.loadtxt(filePath)

        # Read the PSSN/ opd related data
        fileName = "_".join((simDir, iterDir, "PSSN")) + ".txt"
        filePath = os.path.join(dataDir, imgDir, simDir, iterDir, fileName)
        allData = np.loadtxt(filePath)

        # Read the PSSN data
        allPSSN[:, iIter-startIter] = allData[0, :]

        # Read the effective FWHM data
        allFWHMeff[:, iIter-startIter] = allData[1, :]

        # Read the dm5 data
        allDm5[:, iIter-startIter] = allData[2, :]

        # Read the seeing data
        fileName = "_".join((simDir, iterDir, "E000")) + ".atm"
        filePath = os.path.join(dataDir, imgDir, simDir, iterDir, fileName)
        seeingData = np.loadtxt(filePath, skiprows=1)

        # Get the width of seeing
        # Check the meaning of seeing with Bo for the following parts and related formula.
        # Does it mean the diameter of star on focal plane? 
        w = seeingData[:,1]

        # According to John, seeing = quadrature sum (each layer)
        # Convert sigma into FWHM by multiplying with 2*sqrt(2*log(2))
        allSeeing = np.sqrt(np.sum(w**2))*2*np.sqrt(2*np.log(2))

        # According to John, weight L0 using seeing^2
        L0eff =  np.sum(seeingData[:,2]*w**2) / np.sum(w**2)

        # Calculate the radius at referenced wavelength (500 nm)
        r0_500 = 0.976*0.5e-6/(allSeeing/3600/180*np.pi)
        
        # Calculate the radius at specific wavelength 
        r0 = r0_500*(wavelength/0.5)**1.2
        
        # Get the seeing in arcsec
        allSeeingVk[iIter-startIter] = 0.976*wavelength*1e-6\
                                         /r0*np.sqrt(1-2.183*(r0/L0eff)**0.356)\
                                         /np.pi*180*3600

        # Read the ellipticity data
        fileName = "_".join((simDir, iterDir, "elli")) + ".txt"
        filePath = os.path.join(dataDir, imgDir, simDir, iterDir, fileName)
        allElli[:, iIter-startIter] = np.loadtxt(filePath)

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

    # Plot the data
    __subsystemFigure(axM2CamDz, index=xticks, data=allPert[0, :], marker="r.-", 
                      label="M2 dz")
    __subsystemFigure(axM2CamDz, index=xticks, data=allPert[5, :], marker="b.-", 
                      label="Cam dz")

    # Label in figure
    if (dofRange is not None):
        title = "M2 %d/$\pm$%d$\mu$m; Cam %d/$\pm$%d$\mu$m" % (
                round(np.max(np.absolute(allPert[0, :]))), dofRange[0],
                round(np.max(np.absolute(allPert[5, :]))), dofRange[5])
    else:
        title = "M2 %d$\mu$m; Cam %d$\mu$m" % (round(np.max(np.absolute(allPert[0, :]))),
                                               round(np.max(np.absolute(allPert[5, :]))))

    __subsystemFigure(axM2CamDz, xlabel="iteration", ylabel="$\mu$m", 
                      title=title, grid=False)

    # 2: M2, cam dx,dy
    # Plot the data
    __subsystemFigure(axM2CamDxDy, index=xticks, data=allPert[1, :], marker="r.-", 
                      label="M2 dx")
    __subsystemFigure(axM2CamDxDy, index=xticks, data=allPert[2, :], marker="r*-", 
                      label="M2 dy")
    __subsystemFigure(axM2CamDxDy, index=xticks, data=allPert[6, :], marker="b.-", 
                      label="Cam dx")
    __subsystemFigure(axM2CamDxDy, index=xticks, data=allPert[7, :], marker="b*-", 
                      label="Cam dy")

    # Label in figure
    if (dofRange is not None):
        title = "M2 %d/$\pm$%d$\mu$m; Cam %d/$\pm$%d$\mu$m" % (
                round(np.max(np.absolute(allPert[1:3, :]))), dofRange[1],
                round(np.max(np.absolute(allPert[6:8, :]))), dofRange[6])
    else:
        title = "M2 %d$\mu$m; Cam %d$\mu$m" % (round(np.max(np.absolute(allPert[1:3, :]))),
                                               round(np.max(np.absolute(allPert[6:8, :]))))

    __subsystemFigure(axM2CamDxDy, xlabel="iteration", ylabel="$\mu$m", title=title, 
                      grid=False)

    # 3: M2, cam rx,ry
    # Plot the data
    __subsystemFigure(axM2CamRxRy, index=xticks, data=allPert[3, :], marker="r.-", 
                      label="M2 rx")
    __subsystemFigure(axM2CamRxRy, index=xticks, data=allPert[4, :], marker="r*-", 
                      label="M2 ry")
    __subsystemFigure(axM2CamRxRy, index=xticks, data=allPert[8, :], marker="b.-", 
                      label="Cam rx")
    __subsystemFigure(axM2CamRxRy, index=xticks, data=allPert[9, :], marker="b*-", 
                      label="Cam ry")

    # Label in figure
    if (dofRange is not None):
        title = "M2 %d/$\pm$%darcsec; Cam %d/$\pm$%darcsec" % (
                round(np.max(np.absolute(allPert[3:5, :]))), dofRange[3],
                round(np.max(np.absolute(allPert[8:10, :]))), dofRange[8])
    else:
        title = "M2 %darcsec; Cam %darcsec" % (round(np.max(np.absolute(allPert[3:5, :]))), 
                                               round(np.max(np.absolute(allPert[8:10, :]))))

    __subsystemFigure(axM2CamRxRy, xlabel="iteration", ylabel="arcsec", 
                      title=title, grid=False)

    # 4: M1M3 bending

    # Set the colors in figure 
    myColors = ("r", "b", "g", "c", "m", "y", "k")

    # Calculate the standard deviation of bending mode offset
    rms = np.std(allPert[10: 10+nB13Max, :], axis=1)
    idx = np.argsort(rms)

    # Plot the data
    for ii in range(interestedBend, nB13Max+1):
        __subsystemFigure(axM1M3B, index=xticks, data=allPert[idx[-ii]+10, :], 
                          marker=myColors[-1]+".-", grid=False)

    for ii in range(1, interestedBend+1):
       __subsystemFigure(axM1M3B, index=xticks, data=allPert[idx[-ii]+10, :], 
                         marker=myColors[ii-1]+".-", label="M1M3 b%d" % (idx[-ii]+1), 
                         grid=False)

    # Label in figure
    allF = M1M3ActForce[:, :nB13Max].dot(allPert[10:10+nB13Max, :])
    stdForce = np.std(allF, axis=0)
    maxForce = np.max(allF, axis=0)

    if (dofRange is not None):
        title = "Max %d/$\pm$%dN; RMS %dN" % (round(np.max(maxForce)), round(dofRange[0]/rhoM13),
                                              round(np.max(stdForce)))
    else:
        title = "Max %dN; RMS %dN" % (round(np.max(maxForce)), round(np.max(stdForce)))

    __subsystemFigure(axM1M3B, xlabel="iteration", ylabel="$\mu$m", 
                      title=title, grid=False)

    # 5: M2 bending

    # Calculate the standard deviation of bending mode offset
    rms = np.std(allPert[10+nB13Max: 10+nB13Max+nB2Max, :], axis=1)
    idx = np.argsort(rms)

    # Plot the data
    for ii in range(interestedBend, nB2Max+1):
        __subsystemFigure(axM2B, index=xticks, data=allPert[idx[-ii]+10+nB13Max, :], 
                          marker=myColors[-1]+".-", grid=False)

    for ii in range(1, interestedBend+1):
       __subsystemFigure(axM2B, index=xticks, data=allPert[idx[-ii]+10+nB13Max, :], 
                         marker=myColors[ii-1]+".-", label="M2 b%d" % (idx[-ii]+1), 
                         grid=False)

    # Label in figure
    allF = M2ActForce[:, :nB2Max].dot(allPert[10+nB13Max:10+nB13Max+nB2Max, :])
    stdForce = np.std(allF, axis=0)
    maxForce = np.max(allF, axis=0)

    if (dofRange is not None):
        title = "Max %d/$\pm$%dN; RMS %dN" % (round(np.max(maxForce)), round(dofRange[0]/rhoM2),
                                              round(np.max(stdForce)))
    else:
        title = "Max %dN; RMS %dN" % (round(np.max(maxForce)), round(np.max(stdForce)))

    __subsystemFigure(axM2B, xlabel="iteration", ylabel="$\mu$m", 
                      title=title, grid=False)

    # 6: PSSN

    # Plot the data
    for ii in range(nField):
        __subsystemFigure(axPSSN, index=xticks, data=1-allPSSN[ii, :], 
                          marker="b.-", logPlot=True)
    __subsystemFigure(axPSSN, index=xticks, data=1-allPSSN[-1, :], 
                      marker="r.-", label="GQ(1-PSSN)", logPlot=True)

    # Label in figure
    if (allPSSN.shape[1] > 1):
        title = "Last 2 PSSN: %5.3f, %5.3f" % (allPSSN[-1, -2], allPSSN[-1, -1])
    else:
        title = "Last PSSN: %5.3f" % allPSSN[-1, -1]
    __subsystemFigure(axPSSN, xlabel="iteration", title=title)

    # 7: FWHMeff

    # Plot the data
    # Plot all effective FWHM in arcsec 
    if (debugLevel >= 0):
        for ii in range(nField):
            __subsystemFigure(axFWHMeff, index=xticks, data=allFWHMeff[ii, :], marker="b.-")
    __subsystemFigure(axFWHMeff, index=xticks, data=allFWHMeff[-1, :], marker="r.-", 
                      label="GQ ($FWHM_{eff}$)")

    # Plot the seeing in arcsec
    __subsystemFigure(axFWHMeff, index=xticks, data=allSeeingVk, marker="g.-", 
                      label="seeing", grid=False)

    # Plot the error budget
    __subsystemFigure(axFWHMeff, index=xticks, data=iqBudget*np.ones(len(xticks)), 
                      marker="k-", label="Error Budget", grid=False)

    # Label in figure
    if (debugLevel == -1):
        title = "$FWHM_{eff}$"
    else:
        if (allFWHMeff.shape[1] > 1):
            title = "Last 2 $FWHM_{eff}$: %5.3f, %5.3f arcsec" % (allFWHMeff[-1, -2], allFWHMeff[-1, -1])
        else:
            title = "Last $FWHM_{eff}$: %5.3f arcsec" % (allFWHMeff[-1, -1])

    __subsystemFigure(axFWHMeff, xlabel="iteration", ylabel="arcsec", title=title)

    # 8: dm5

    # Check with Bo for the meaning of dm5 here.

    # Plot the data
    for ii in range(nField):
        __subsystemFigure(axDm5, index=xticks, data=allDm5[ii, :], marker="b.-")
    __subsystemFigure(axDm5, index=xticks, data=allDm5[-1, :], marker="r.-", label="GQ($\Delta$m5)")

    # Label in figure
    if (allDm5.shape[1] > 1):
        title = "Last 2 $\Delta$m5: %5.3f, %5.3f" % (allDm5[-1, -2], allDm5[-1, -1])
    else:
        title = "Last $\Delta$m5: %5.3f" % (allDm5[-1, -1])

    __subsystemFigure(axDm5, xlabel="iteration", title=title)

    # 9: ellipticity

    # Check with Bo for the meaning of ellipticity here.

    # Plot the data
    if (debugLevel >= 0):
        for ii in range(nField):
            __subsystemFigure(axElli, index=xticks, data=allElli[ii, :]*100, marker="b.-")
    __subsystemFigure(axElli, index=xticks, data=allElli[-1, :]*100, marker="r.-", 
                      label="GQ (ellipticity)")

    # Plot the error budget in percent
    if (eBudget is not None):
        __subsystemFigure(axElli, index=xticks, data=100*eBudget*np.ones(len(xticks)), 
                          marker="k-", label="SRD Spec (Median)", grid=False)
    
    # Label in figure
    if (debugLevel == -1):
        title = "Ellipticity"
    else:
        if (allElli.shape[1] > 1):
            title = "Last 2 e: %4.2f%%, %4.2f%%" % (allElli[-1, -2]*100, allElli[-1, -1]*100)
        else:
            title = "Last e: %4.2f%%" % (allElli[-1, -1]*100)
    
    __subsystemFigure(axElli, xlabel="iteration", ylabel="percent", title=title)

    # Save the image or not
    __saveFig(plt, dpi=dpi, saveFilePath=saveFilePath, doWrite=doWrite)

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
        doWrite {[bool]} -- Write the figure into the file or not. (default: {True})
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
    __subsystemFigure(axm2rig, xticksStart=1, index=range(0, 3), data=uk, marker="ro", 
                      title="M2 dz, dx, dy", ylabel="$\mu$m")
    __subsystemFigure(axm2rot, xticksStart=4, index=range(3, 5), data=uk, marker="ro", 
                      title="M2 rx, ry", ylabel="arcsec")
    __subsystemFigure(axcamrig, xticksStart=6, index=range(5, 8), data=uk, marker="ro", 
                      title="Cam dz, dx, dy", ylabel="$\mu$m")
    __subsystemFigure(axcamrot, xticksStart=9, index=range(8, 10), data=uk, marker="ro", 
                      title="Cam rx, ry", ylabel="arcsec")
    __subsystemFigure(axm13, xticksStart=1, index=range(10, 30), data=uk, marker="ro", 
                      title="M1M3 bending", ylabel="$\mu$m")
    __subsystemFigure(axm2, xticksStart=1, index=range(30, 50), data=uk, marker="ro", 
                      title="M2 bending", ylabel="$\mu$m")

    # Plot the wavefront error
    subPlotList = [axz1, axz2, axz3, axz4]
    titleList = ["Zernikes R44", "Zernikes R40", "Zernikes R00", "Zernikes R04"]

    # Plot the final wavefront error in the basis of Zk
    if ((yfinal is not None) and (termZk is not None)):
        label = None
        if (iterNum is not None):
            label = "iter %d" % (iterNum-1)
        __wavefrontFigure(subPlotList, titleList, yfinal, termZk, marker="*b-", 
                          xticksStart=4, label=label)

    # Plot the residue of wavefront error if full correction of wavefront error is applied
    # This is for the performance prediction only
    if ((yresi is not None) and (termZk is not None)):
        label = "if full correction applied"
        __wavefrontFigure(subPlotList, titleList, yresi, termZk, marker="*r-", 
                          xticksStart=4, label=label)

    # Save the image or not
    __saveFig(plt, saveFilePath=saveFilePath, doWrite=doWrite)

def __saveFig(plotFig, dpi=None, saveFilePath=None, doWrite=True):
    """
    
    Save the figure to specific path or just show the figure.
    
    Arguments:
        plotFig {[matplotlib.pyplot]} -- Pyplot figure object.
    
    Keyword Arguments:
        dpi {[int]} -- The resolution in dots per inch. (default: {None})
        saveFilePath {[str]} -- File path to save the figure. (default: {None})
        doWrite {[bool]} -- Write the figure into the file or not. (default: {True})
    """

    if (doWrite):
        if (saveFilePath):
            
            # Adjust the space between xlabel and title for neighboring sub-figures
            plotFig.tight_layout()

            # Save the figure to file
            plotFig.savefig(saveFilePath, bbox_inches="tight", dpi=dpi)

            # Close the figure
            plotFig.close()
    else:
        # Show the figure only
        plotFig.show()

def __wavefrontFigure(subPlotList, titleList, wavefront, termZk, marker="b", 
                      xticksStart=None, label=None):
    """
    
    Plot the wavefront error in the basis of annular Zk for each wavefront sensor (WFS).
    
    Arguments:
        subPlotList {[list]} -- The list of subplots of WFS.
        titleList {[list]} -- The title list of WFS. The idea is to use the name of WFS.
        wavefront {[ndarray]} -- Wavefront error in the basis of annular Zk.
        termZk {[int]} -- Number of terms of annular Zk.
    
    Keyword Arguments:
        marker {str} -- Maker of data point. (default: {"b"})
        xticksStart {[float]} -- x-axis start sticks. (default: {None})
        label {[str]} -- Label of data. (default: {None})
    
    Raises:
        RuntimeError -- The lengths of subPlotList and nameList do not match.
    """

    if (len(subPlotList) != len(titleList)):
        raise RuntimeError("The lengths of subPlotList and titleList do not match.")

    for ii in range(len(subPlotList)):
        __subsystemFigure(subPlotList[ii], xticksStart=xticksStart, 
                          index=range(ii*int(termZk), (ii+1)*int(termZk)), data=wavefront, 
                          title=titleList[ii], ylabel="um", marker=marker, label=label)

def __subsystemFigure(subPlot, xticksStart=None, index=None, data=None, marker="b", 
                      xlabel=None, ylabel=None, label=None, title=None, 
                      logPlot=False, grid=True):
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
        xlabel {[str]} -- Label in x-axis. (default: {None})
        ylabel {[str]} -- Label in y-axis. (default: {None})
        label {[str]} -- Label of plot. (default: {None})
        title {[str]} -- Title of plot. (default: {None})
        logPlot {[bool]} -- Plot in log scale. (default: {False})
        grid {[bool]} -- Show the grid or not. (default: {True})
    """

    # Get the data
    if ((index is not None) and (data is not None)):
        # Plot the figure in log scale or not
        if (logPlot):
            subPlot.semilogy(index, data[index], marker, ms=8, label=label)
        else:
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

    # Set the legend
    if (label is not None):
        subPlot.legend(loc="best", shadow=False, fancybox=True)

    # Set the title
    if (title is not None):
        subPlot.set_title(title)

    # Set the grid
    if (grid):
        subPlot.grid()

def isNumber(s):
    """
    
    Check the string is a number or not. Copy this function from:
    https://www.pythoncentral.io/how-to-check-if-a-string-is-a-number-in-python-including-unicode/.
    
    Arguments:
        s {[str]} -- Input string.
    
    Returns:
        [bool] -- True if the input string is a number and vice verse.
    """

    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False

def getLUTforce(zangle, LUTfile):
    """
    
    Get the actuator force of mirror based on the look-up table (LUT).
    
    Arguments:
        zangle {[float]} -- Zenith angle in degree.
        LUTfile {[str]} -- Path to the file of LUT.
    
    Returns:
        [ndarray] -- Actuator forces in specific zenith angle.
    
    Raises:
        ValueError -- Incorrect LUT degree order.
    """

    # Read the LUT file
    lut = np.loadtxt(LUTfile)

    # Get the step. The values of LUT are listed in every step size.
    # The degree range is 0 - 90 degree.
    # The file in the simulation is every 1 degree. The formal one should 
    # be every 5 degree.
    ruler = lut[0, :]
    stepList = np.diff(ruler)
    if np.any(stepList <= 0):
        raise ValueError("The degee order in LUT is incorrect.")

    # The specific zenith angle is larger than the listed angle range
    if (zangle >= ruler.max()):
        # Use the biggest listed zenith angle data instead
        lutForce = lut[1:, -1]

    # The specific zenith angle is smaller than the listed angle range
    elif (zangle <= ruler.min()):
        # Use the smallest listed zenith angle data instead
        lutForce = lut[1:, 0]

    # The specific zenith angle is in the listed angle range
    else:
        # Linear fit the data
        # Find the boundary indexes for the specific zenith angle
        p1 = np.where(ruler<=zangle)[0][-1]
        p2 = p1+1

        # Do the linear approximation
        w2 = (zangle-ruler[p1])/stepList[p1]
        w1 = 1-w2

        lutForce = w1*lut[1:, p1] + w2*lut[1:, p2]

    return lutForce

def hardLinkFile(targetFilePath, sourceNum, targetNum):
    """
    
    Hard link the past calculation result instead of repeated calculation.
    
    Arguments:
        targetFilePath {[str]} -- Path of file that is intended to do the hard link 
                                  with the previous result.
        sourceNum {[int]} -- Source simulation number.
        targetNum {[int]} -- Target simulation number.
    """

    if (not os.path.isfile(targetFilePath)):

        # Get the path of base run file by changing the simulation number
        sourceFilePath = targetFilePath.replace("sim%d" % targetNum, "sim%d" % sourceNum)

        # Construct a hard link
        os.link(sourceFilePath, targetFilePath)

def runProgram(command, binDir=None, argstring=None):
    """
    
    Run the program w/o arguments.
    
    Arguments:
        command {[string]} -- Command of application.
    
    Keyword Arguments:
        binDir {[str]} -- Directory of binary application. (default: {None})
        argstring {[str]} -- Arguments of program. (default: {None})
    
    Raises:
        RuntimeError -- There is the error in running the program.
    """

    # Directory of binary application
    if (binDir is not None):
        command = os.path.join(binDir, command)

    # Arguments for the program
    if (argstring is not None):
        command += (" " + argstring)

    # Call the program w/o arguments
    if (subprocess.call(command, shell=True) != 0):
        raise RuntimeError("Error running: %s" % command)

def writeToFile(filePath, content=None, sourceFile=None, mode="a"):
    """
    
    Write the file based on the content to put or the source file to copy with.
    
    Arguments:
        filePath {[str]} -- File path to write.
    
    Keyword Arguments:
        content {[str]} -- Content to write into the file. (default: {None})
        sourceFile {[str]} -- Source file to write its content into the file. (default: {None})
        mode {[str]} -- Overwrite ("w") or append ("a") the file. (default: {"a"})
    """

    if mode not in ("w", "a"):
        raise ValueError("Mode: %s is not supported." % mode)

    if (content is not None) or (sourceFile is not None):

        # Open the file. If the file path does not exist, the new file will be generated.
        # Use the append instead of 
        fid = open(filePath, mode)

        # Write the content into the file
        if (content is not None):
            fid.write(content)

        # Write the content of source file into the file
        if (sourceFile is not None):
            fSrc = open(sourceFile, "r")
            fid.write(fSrc.read())
            fSrc.close()

        # Close the file
        fid.close()

def fieldXY2ChipFocalPlane(focalPlanePath, fieldX, fieldY, debugLevel=0):
    """
    
    Transform the field x, y to pixel x, y on the chip belong to a certain focal plane.
    
    Arguments:
        focalPlanePath {[str]} -- Path of focal plane layout data.
        fieldX {[field]} -- Field x.
        fieldY {[field]} -- Field y.
    
    Keyword Arguments:
        debugLevel {int} -- Debug level. The higher value gives more information.
                            (default: {0})

    Returns:
        [str] -- Chip Name.
        [int] -- Pixel position x.
        [int] -- Pixel position y.
    """

    # Get the chip boundary
    ruler = getChipBoundary(focalPlanePath)

    # Show the condition of ruler
    if (debugLevel >= 3):
        print("ruler:\n")
        print(ruler)
        print(len(ruler))

    # Get the raft (r), chip (c), and pixel (p)
    # It is noticed that the dimension of CCD has been hard-coded here. Check with Bo.
    rx, cx, px = fieldAgainstRuler(ruler, fieldX, 4000)
    ry, cy, py = fieldAgainstRuler(ruler, fieldY, 4072)

    # Get the chip name
    chipName = "R%d%d_S%d%d" % (rx, ry, cx, cy)

    return chipName, px, py

def getChipBoundary(fplayoutFile):
    """
    
    Get the chip boundary along x direction in um.
    
    Arguments:
        fplayoutFile {[str]} -- Path of focal plane layout data.
    
    Returns:
        [ndarray] -- Chip boundary along x direction.
    """

    # Get the chip and its center coordinate x, y in the unit of um
    mydict = {}
    f = open(fplayoutFile)

    for line in f:
        line = line.strip()
        if (line.startswith("R")):
            mydict[line.split()[0]] = [float(line.split()[1]),
                                       float(line.split()[2])]

    f.close()
    
    # Get the ruler
    ruler = sorted(set([x[0] for x in mydict.values()]))

    # Change to numpy array
    ruler = np.array(ruler)

    return ruler

def fieldAgainstRuler(ruler, field, chipPixel):
    """
    
    Get the raft, chip, and pixel position along a certain axis based on the chip pixel.
    
    Arguments:
        ruler {[ndarray]} -- Chip boundary along x direction.
        field {[float]} -- Field position in degree.
        chipPixel {[int]} -- Length of pixel along a certain direction (x or y).
    
    Returns:
        [int] -- Raft.
        [int] -- Chip.
        [int] -- Pixel position.
    """

    # Change the unit from degree to micron    
    field = field*180000  # degree to micron

    # Find the chip for this field position
    p2 = (ruler >= field)

    # Too large to be in range
    if (np.count_nonzero(p2) == 0):
        # p starts from 0
        p = len(ruler) - 1

    # Too small to be in range
    elif (p2[0]):
        p = 0

    # Field position is in the range
    else:

        # Lower bound
        p1 = p2.argmax() - 1

        # Upper bound
        p2 = p2.argmax()

        # Check the field position is in the left (p1) or right (p2) chip
        if (ruler[p2] - field) < (field - ruler[p1]):
            p = p2
        else:
            p = p1

    # Change the unit from um to pixel
    # 1 pixel = 10 um
    pixel = (field - ruler[p])/10

    # This is because the chip boundary is recorded based on the center
    pixel += chipPixel/2

    # Raft 
    raft = np.floor(p / 3)

    # Chip
    chip = p % 3

    # Pixel
    pixelPos = int(pixel)

    return raft, chip, pixelPos


if __name__ == "__main__":

    uk = np.arange(50)
    yfinal = np.arange(19*4)
    iterNum = 5
    yresi = np.arange(19*4)*3

    # Initiate the mirror actuator force
    # M1M3dir = "../data/M1M3"
    # M1M3 = aosM1M3(M1M3dir)

    # M2dir = "../data/M2"
    # M2 = aosM2(M2dir)

    # dofRange = np.arange(0,51)*1e3

    # dataDir = "/Users/Wolf/Documents/aosOutput"
    # saveFilePath = "/Users/Wolf/Desktop/temp.png"
    # showSummaryPlots(dataDir, dofRange=dofRange, iSim=6, startIter=0, endIter=5, 
    #                  M1M3ActForce=M1M3.force, M2ActForce=M2.force, eBudget=0.04, 
    #                  saveFilePath=saveFilePath, doWrite=True, debugLevel=3)

    saveFilePath1 = "/Users/Wolf/Desktop/temp1.png"
    showControlPanel(uk=uk, yfinal=yfinal, yresi=yresi, iterNum=iterNum, saveFilePath=saveFilePath1, doWrite=True)