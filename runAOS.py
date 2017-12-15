#!/usr/bin/env python

# @author: Bo Xin
# @      Large Synoptic Survey Telescope

# main function

import os, sys

from argparse import ArgumentParser
from datetime import datetime

from aos.aosWFS import aosWFS
from aos.aosEstimator import aosEstimator
from aos.aosController import aosController
from aos.aosMetric import aosMetric
from aos.aosM1M3 import aosM1M3
from aos.aosM2 import aosM2
from aos.aosTeleState import aosTeleState, getTelEffWave
from aos.aosUtility import showControlPanel, showSummaryPlots

from astropy.time import TimeDelta

def main(phosimDir, cwfsDir, outputDir, aosDataDir, algoFile="exp", cwfsModel="offAxis"):
    """

    Run the AOS close loop code.

    Arguments:
        phosimDir {[str]} -- PhoSim directory.
        cwfsDir {[str]} -- cwfs directory.
        outputDir {[str]} -- Output files directory.
        aosDataDir {[str]} -- AOS data directory.

    Keyword Arguments:
        algoFile {[str]} -- Algorithm to solve the transport of intensity equation (TIE).
                          (default: {"exp"})
        cwfsModel {[str]} -- Optical model. (default: {"offAxis"})
    """

    global state, M1M3, M2, ctrl, args, wfs, metr

    # Instantiate the parser for command line to use
    parser = __setParseAugs()

    # Get the arguments passed by the command line
    args = parser.parse_args()

    # Analyze the arguments

    # Do the summary plot
    if (args.makesum):
        args.sensor = "pass"
        args.ctrloff = True
        args.opdoff = True
        args.psfoff = True
        args.pssnoff = True
        args.ellioff = True

    # Show the debug information on screen
    if (args.debugLevel >= 1):
        print(args)

    # Decide the wavelength in micron
    # 0.5 micron is a ref wavelength. And PhoSim will do the monochromatic light simulation.
    if (args.wavestr == "0.5"):
        band = "g"
        wavelength = float(args.wavestr)
    else:
        band = args.wavestr
        wavelength = 0

    # Get the effective wavelength
    effwave = getTelEffWave(wavelength, band)[0]

    # *****************************************
    # simulate the perturbations
    # *****************************************

    # Instantiate mirrors
    M1M3dir = os.path.join(aosDataDir, "M1M3")
    M1M3 = aosM1M3(M1M3dir, debugLevel=args.debugLevel)

    M2dir = os.path.join(aosDataDir, "M2")
    M2 = aosM2(M2dir, debugLevel=args.debugLevel)

    # *****************************************
    # run wavefront sensing algorithm
    # *****************************************

    # Instantiate the AOS wavefront sensor estimator
    wfs = aosWFS(cwfsDir, args.inst, algoFile, 128, band, effwave, aosDataDir, args.debugLevel)

    # *****************************************
    # state estimator
    # *****************************************

    # State is defined after esti, b/c, for example, ndof we use in state
    # depends on the estimator.
    simDirName = "sim%d" % args.iSim

    # Get the directory positions of image and perturbation
    pertDir = os.path.join(outputDir, "pert", simDirName)
    imageDir = os.path.join(outputDir, "image", simDirName)

    # Instantiate the AOS estimator
    esti = aosEstimator(aosDataDir, args.estimatorParam, args.inst, wfs, icomp=args.icomp,
                        izn3=args.izn3, debugLevel=args.debugLevel)

    # Instantiate the AOS telescope state
    state = aosTeleState(aosDataDir, args.inst, args.simuParam, args.iSim, esti.ndofA,
                         phosimDir, pertDir, imageDir, band, wavelength,
                         args.enditer, M1M3=M1M3, M2=M2, opSimHisDir=aosDataDir, 
                         debugLevel=args.debugLevel)

    # *****************************************
    # control algorithm
    # *****************************************

    # Instantiate the AOS metrology and controller
    metr = aosMetric(args.inst, state.opdSize, wfs.znwcs3, args.debugLevel)
    ctrl = aosController(aosDataDir, args.inst, args.controllerParam, esti, metr,
                         M1M3.force, M2.force, effwave, args.gain, covM=wfs.covM, 
                         debugLevel=args.debugLevel)

    sys.exit()

    # *****************************************
    # start the Loop
    # *****************************************

    # Do the close loop simulation
    for iIter in range(args.startiter, args.enditer + 1):

        # Show the iteration number of not
        if (args.debugLevel >= 3):
            print("iteration No. %d" % iIter)

        # Observation ID
        obsID = 9000000 + args.iSim*1000 + iIter*10

        # Iteration time in the format of MJD
        timeIter = state.time0 + iIter*TimeDelta(39, format="sec")

        # Set the telescope iteration number
        state.setIterNum(iIter)

        # Set the path of perturbation file
        state.setPerFilePath(metr, wfs=wfs)

        # Read the perturbation file in the previous iteration run
        state.readPrePerFile(metr=metr, wfs=wfs)

        if (not args.ctrloff):

            # Update the telescope status
            if (iIter > 0):
                yfinal, yresi = esti.estimate(state, wfs, ctrl.y2File, args.sensor, authority=ctrl.Authority, obsID=obsID)
                uk = ctrl.getMotions(esti, metr, nWFS=wfs.nWFS, state=state)

                # Draw the dof and Zk 
                # File name and path
                controlPanelFigName = "sim%d_iter%d_ctrl.png" % (state.iSim, state.iIter)
                controlPanelFigPath = os.path.join(pertDir, "iter%d" % state.iIter, controlPanelFigName)
                
                # Generate the figure
                showControlPanel(uk=uk, yfinal=yfinal, yresi=yresi, iterNum=state.iIter, 
                                 saveFilePath=controlPanelFigPath, doWrite=True)

                # Need to remake the pert file here.
                # It will be inserted into OPD.inst, PSF.inst later
                state.update(uk, ctrl.range, M1M3=M1M3, M2=M2)

            if (args.baserun > 0 and iIter == 0):
                # Read the telescope status from the resetted baserun iteration
                state.getPertFilefromBase(args.baserun)
            else:
                # Write data into perturbation file
                state.writePertFile(esti.ndofA, M1M3=M1M3, M2=M2)

        # Do the metrology calculation
        if (args.baserun > 0 and iIter == 0):

            # Consider the 35 field points (contains WFS) instead of 31 in "LSST". 
            # This is why to use metr.nFieldp4 instead of metr.nField
            # It is noticed that there is no difference between metr.nField and metr.nField4 in "ComCam".
            # Need to redesign this part.
            state.getOPDAllfromBase(args.baserun, metr.nFieldp4)
            
            state.getPSFAllfromBase(args.baserun, metr)
            metr.getPSSNandMorefromBase(args.baserun, state)
            metr.getEllipticityfromBase(args.baserun, state)

            if args.sensor not in ("ideal", "covM", "pass", "check"):
                wfs.getZ4CfromBase(args.baserun, state.iSim)

        else:
            # Calculate the optical path difference (OPD) by PhoSim
            state.getOPDAll(obsID, metr, args.numproc, wfs.znwcs, wfs.inst.obscuration, 
                            opdoff=args.opdoff, debugLevel=args.debugLevel)

            # Calculate the point spread function (PSF) by PhoSim
            state.getPSFAll(obsID, metr, args.numproc, psfoff=args.psfoff, debugLevel=args.debugLevel)

            metr.getPSSNandMore(args.pssnoff, state, args.numproc, args.debugLevel)
            metr.getEllipticity(args.ellioff, state, args.numproc, args.debugLevel)

            if args.sensor not in ("ideal", "covM", "pass"):

                if (args.sensor == "phosim"):
                    # Create donuts for last iter, so that picking up from there will be easy
                    state.getWFSAll(obsID, timeIter, wfs, metr, args.numproc, args.debugLevel)
                    wfs.preprocess(obsID, state, metr, args.debugLevel)

                if args.sensor in ("phosim", "cwfs"):
                    wfs.parallelCwfs(cwfsModel, args.numproc)

                if args.sensor in ("phosim", "cwfs", "check"):
                    wfs.checkZ4C(state, metr, args.debugLevel)

    # Give the summary file name and path
    sumFigName = "sim%d_iter%d-%d.png" % (state.iSim, args.startiter, args.enditer)
    sumFigPath = os.path.join(pertDir, sumFigName)

    # Draw the summary plot
    showSummaryPlots(dataDir=outputDir, dofRange=ctrl.range, iSim=args.iSim, ndofA=esti.ndofA, 
                     nField=metr.nField, startIter=args.startiter, endIter=args.enditer, 
                     nB13Max=esti.nB13Max, nB2Max=esti.nB2Max, rhoM13=ctrl.rhoM13, M1M3ActForce=M1M3.force, 
                     rhoM2=ctrl.rhoM2, M2ActForce=M2.force, wavelength=effwave, iqBudget=state.iqBudget, 
                     eBudget=state.eBudget, dpi=500, saveFilePath=sumFigPath, doWrite=True, 
                     debugLevel=args.debugLevel)

    # Show the finish of iteration
    print("Done the runnng iterations: %d to %d." % (args.startiter, args.enditer))

def __setParseAugs():
    """

    Set the parser to pass the arguments from the command line.

    Returns:
        [ArgumentParser] -- Parser for the command line to use.
    """

    # Instantiate the argument parser
    parser = ArgumentParser(description="-----LSST Integrated Model------")

    # Add the arguments of parser

    # Simulation number
    helpDescript = "simulation #"
    parser.add_argument("iSim", type=int, help=helpDescript)

    # Override the "icomp" in aosEstimator
    helpDescript = "override icomp in the estimator parameter file, default=no override"
    parser.add_argument("-icomp", type=int, help=helpDescript)

    # Override the "izn3" in aosEstimator
    helpDescript = "override izn3 in the estimator parameter file, default=no override"
    parser.add_argument("-izn3", type=int, help=helpDescript)

    # Iteration number to start with
    helpDescript = "iteration No. to start with, default=0"
    parser.add_argument("-start", dest="startiter", type=int, default=0, help=helpDescript)

    # Iteration number to end with
    helpDescript = "iteration No. to end with, default=5"
    parser.add_argument("-end", dest="enditer", type=int, default=5, help=helpDescript)

    # Sensor type
    # Ideal: Do not use PhoSim to generate wavefront image. Predict wavefront Zk  
    # from the iteration 0 in telescope state directly.
    # covM: Add the random number to calculated Zk from iteration 0 to simulate the 
    # environment change. It is one more step from "Ideal" sensor.
    # PhoSim: Use PhoSim to generate the WFS images to calculate the Zk.
    # Ref: https://confluence.lsstcorp.org/pages/viewpage.action?pageId=64699925
    sensorChoices = ("ideal", "covM", "phosim", "cwfs", "check", "pass")
    helpDescript = "ideal: use true wavefront in estimator;\
                    covM: use covariance matrix to estimate wavefront;\
                    phosim: run PhoSim to create WFS images;\
                    cwfs: start by running cwfs on existing images;\
                    check: check wavefront against truth;\
                    pass: do nothing"
    parser.add_argument("-sensor", choices=sensorChoices, help=helpDescript)

    # Use the control algorithm or not
    helpDescript = "w/o applying ctrl rules or regenrating pert files"
    parser.add_argument("-ctrloff", help=helpDescript, action="store_true")

    # Regenerate the OPD map or not
    helpDescript = "w/o regenerating OPD maps"
    parser.add_argument("-opdoff", help=helpDescript, action="store_true")

    # Regenerate the PSF image or not
    helpDescript = "w/o regenerating psf images"
    parser.add_argument("-psfoff", help=helpDescript, action="store_true")

    # Calculate the PSSN or not
    helpDescript = "w/o calculating PSSN"
    parser.add_argument("-pssnoff", help=helpDescript, action="store_true")

    # Calculate the ellipticity or not
    helpDescript = "w/o calculating ellipticity"
    parser.add_argument("-ellioff", help=helpDescript, action="store_true")

    # Make the summary plot or not. This assums all data are available already.
    helpDescript = "make summary plot, assuming all data available"
    parser.add_argument("-makesum", help=helpDescript, action="store_true")

    # Number of processor
    helpDescript = "Number of Processors Phosim uses"
    parser.add_argument("-p", dest="numproc", default=1, type=int, help=helpDescript)

    # Gain value
    helpDescript = "override gain in the controller parameter file, default=no override"
    parser.add_argument("-g", dest="gain", default=0.7, type=float, help=helpDescript)

    # Type of instrument ("lsst", "comcam10", "comcam15", "comcam20")
    helpDescript = "instrument name, default=lsst"
    parser.add_argument("-i", dest="inst", default="lsst", help=helpDescript)

    # Simuation parameter file
    helpDescript = "simulation parameter file in data/, default=single_dof"
    parser.add_argument("-s", dest="simuParam", default="single_dof", help=helpDescript)

    # AOS estimator parameters
    helpDescript = "estimator parameter file in data/, default=pinv"
    parser.add_argument("-e", dest="estimatorParam", default="pinv", help=helpDescript)

    # AOS controller parameters
    controlChoices = ("optiPSSN_x0", "optiPSSN_0", "optiPSSN_x0xcor", "optiPSSN_x00", "null")
    helpDescript = "controller parameter file in data/, default=optiPSSN"
    parser.add_argument("-c", dest="controllerParam", default=controlChoices[0],
                        choices=controlChoices, help=helpDescript)

    # Wavelength in micron
    wavelengthChoices = ("0.5", "u", "g", "r", "i", "z", "y")
    helpDescript = "wavelength in micron, default=0.5"
    parser.add_argument("-w", dest="wavestr", choices=wavelengthChoices,
                        default=wavelengthChoices[0], help=helpDescript)

    # Debug level
    debugChoices = (-1, 0, 1, 2, 3)
    helpDescript = "debug level: -1=quiet, 0=Zernikes, 1=operator, 2=expert, 3=everything, default=0"
    parser.add_argument("-d", dest="debugLevel", type=int, default=0, choices=debugChoices,
                        help=helpDescript)

    # Reset the base run as this run instead of iter0 or not
    helpDescript = "iter0 is same as this run, so skip iter0"
    parser.add_argument("-baserun", default=-1, type=int, help=helpDescript)

    # Return the parser
    return parser

if __name__ == "__main__":

    # Output data directory
    outputDir = "/Users/Wolf/Documents/aosOutput"

    # PhoSim directory
    phosimDir = "/Users/Wolf/Documents/bitbucket/phosim_syseng2"

    # cwfs directory
    cwfsDir = "/Users/Wolf/Documents/github/cwfs"

    # AOS data directory
    aosDataDir = "/Users/Wolf/Documents/stash/ts_lsst_aos_phosim/data"

    # Get the start time
    timeStart = datetime.now().replace(microsecond=0)

    # Do the AOS
    main(phosimDir, cwfsDir, outputDir, aosDataDir)

    # Get the finish time
    timeFinish = datetime.now().replace(microsecond=0)

    # Print the calculation time
    print("Calcuation time is: %s." % (timeFinish - timeStart))
