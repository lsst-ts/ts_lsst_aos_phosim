#!/usr/bin/env python

# @author: Bo Xin
# @      Large Synoptic Survey Telescope

# main function

from argparse import ArgumentParser
from datetime import datetime

from aosWFS import aosWFS
from aosEstimator import aosEstimator
from aosController import aosController
from aosMetric import aosMetric
from aosM1M3 import aosM1M3
from aosM2 import aosM2
from aosTeleState import aosTeleState

def main():

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
    if (args.wavestr == "0.5"):
        band = "g"
        wavelength = float(args.wavestr)
    else:
        band = args.wavestr
        wavelength = 0

    # *****************************************
    # simulate the perturbations
    # *****************************************
    M1M3 = aosM1M3(args.debugLevel)
    M2 = aosM2(args.debugLevel)
    phosimDir = "/Users/Wolf/Documents/bitbucket/phosim_syseng2"
    # znPert = 28  # znmax used in pert file to define surfaces


    # *****************************************
    # run wavefront sensing algorithm
    # *****************************************
    cwfsDir = "/Users/Wolf/Documents/github/cwfs"
    algoFile = 'exp'
    if wavelength == 0:
        effwave = aosTeleState.effwave[band]
    else:
        effwave = wavelength
    wfs = aosWFS(cwfsDir, args.inst, algoFile,
                 128, band, effwave, args.debugLevel)

    cwfsModel = 'offAxis'

    # *****************************************
    # state estimator
    # *****************************************
    esti = aosEstimator(args.inst, args.estimatorParam, wfs, args.icomp,
                        args.izn3, args.debugLevel)
    # state is defined after esti, b/c, for example, ndof we use in state
    # depends on the estimator.
    pertDir = 'pert/sim%d' % args.iSim
    imageDir = 'image/sim%d' % args.iSim
    state = aosTeleState(args.inst, args.simuParam, args.iSim,
                         esti.ndofA, phosimDir,
                         pertDir, imageDir, band, wavelength,
                         args.enditer,
                         args.debugLevel, M1M3=M1M3, M2=M2)
    # *****************************************
    # control algorithm
    # *****************************************
    metr = aosMetric(args.inst, state.opdSize, wfs.znwcs3, args.debugLevel)
    ctrl = aosController(args.inst, args.controllerParam, esti, metr, wfs,
                         M1M3, M2,
                         effwave, args.gain, args.debugLevel)

    # *****************************************
    # start the Loop
    # *****************************************
    for iIter in range(args.startiter, args.enditer + 1):
        if args.debugLevel >= 3:
            print('iteration No. %d' % iIter)

        state.setIterNo(metr, iIter, wfs=wfs)

        if not args.ctrloff:
            if iIter > 0:  # args.startiter:
                esti.estimate(state, wfs, ctrl, args.sensor)
                ctrl.getMotions(esti, metr, wfs, state)
                ctrl.drawControlPanel(esti, state)

                # need to remake the pert file here.
                # It will be inserted into OPD.inst, PSF.inst later
                state.update(ctrl, M1M3, M2)
            if args.baserun > 0 and iIter == 0:
                state.getPertFilefromBase(args.baserun)
            else:
                state.writePertFile(esti.ndofA, M1M3=M1M3, M2=M2)

        if args.baserun > 0 and iIter == 0:
            state.getOPDAllfromBase(args.baserun, metr)
            state.getPSFAllfromBase(args.baserun, metr)
            metr.getPSSNandMorefromBase(args.baserun, state)
            metr.getEllipticityfromBase(args.baserun, state)
            if (args.sensor == 'ideal' or args.sensor == 'covM' or
                    args.sensor == 'pass' or args.sensor == 'check'):
                pass
            else:
                wfs.getZ4CfromBase(args.baserun, state)
        else:
            state.getOPDAll(args.opdoff, metr, args.numproc,
                            wfs.znwcs, wfs.inst.obscuration, args.debugLevel)

            state.getPSFAll(args.psfoff, metr, args.numproc, args.debugLevel)

            metr.getPSSNandMore(args.pssnoff, state,
                                args.numproc, args.debugLevel)

            metr.getEllipticity(args.ellioff, state,
                                args.numproc, args.debugLevel)

            if (args.sensor == 'ideal' or args.sensor == 'covM' or
                    args.sensor == 'pass'):
                pass
            else:
                if args.sensor == 'phosim':
                    # create donuts for last iter,
                    # so that picking up from there will be easy
                    state.getWFSAll(wfs, metr, args.numproc, args.debugLevel)
                    wfs.preprocess(state, metr, args.debugLevel)
                if args.sensor == 'phosim' or args.sensor == 'cwfs':
                    wfs.parallelCwfs(cwfsModel, args.numproc, args.debugLevel)
                if args.sensor == 'phosim' or args.sensor == 'cwfs' \
                        or args.sensor == 'check':
                    wfs.checkZ4C(state, metr, args.debugLevel)

    ctrl.drawSummaryPlots(state, metr, esti, M1M3, M2,
                          args.startiter, args.enditer, args.debugLevel)

    print('Done runnng iterations: %d to %d' % (args.startiter, args.enditer))
    

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
    sensorChoices = ("ideal", "covM", "phosim", "cwfs", "check", "pass")
    helpDescript = "ideal: use true wavefront in estimator;\
                    covM: use covarance matrix to estimate wavefront;\
                    phosim: run Phosim to create WFS images;\
                    cwfs: start by running cwfs on existing images;\
                    check: check wavefront against truth; \
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

    # Type of instrument ("lsst" or "comcam")
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

    # Do the base run or not
    helpDescript = "iter0 is same as this run, so skip iter0"
    parser.add_argument("-baserun", default=-1, type=int, help=helpDescript)

    # Return the parser
    return parser

if __name__ == "__main__":

    # Get the start time
    timeStart = datetime.now().replace(microsecond=0)
    
    # Do the AOS
    main()

    # Get the finish time
    timeFinish = datetime.now().replace(microsecond=0)
    
    # Print the calculation time
    print("Calcuation time is: %s." % (timeFinish - timeStart))
