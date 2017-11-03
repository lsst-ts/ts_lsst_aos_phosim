#!/usr/bin/env python
##
# @authors: Bo Xin
# @       Large Synoptic Survey Telescope

import os
import numpy as np

from aos.aosCoTransform import M2CRS2ZCRS, ZCRS2M2CRS

class aosM2(object):

    # Outer radius of M2 mirror in m
    R = 1.710

    # Inner radius of M2 mirror in m
    Ri = 0.9

    def __init__(self, M2dir, debugLevel=0):
        """
        
        Initiate the AocM2 class.
        
        Arguments:
            M2dir {[str]} -- Directory to M2 data.
        
        Keyword Arguments:
            debugLevel {int} -- Debug level. The higher value gives more information. 
                                (default: {0})
        """

        # Get the bending mode information
        data = np.loadtxt(os.path.join(M2dir, "M2_1um_grid.DAT"))

        bx = data[:, 0]
        by = data[:, 1]
        bz = data[:, 2:]


        # Get the actuator force

        # !!! We are using M1M3 forces in place of M2 forces because no M2 force FEA data
        # PhoSim does not need this file. But this should be updated by the measurement in the future.
        # It is noted that if M1M3 and M2 FEA data are updated, the related data in PhoSim
        # is also needed to be updated. But the format is different because the grid between
        # FEA and ZEMAX is diffrent.
        data = np.loadtxt(os.path.join(M2dir, "M2_1um_force.DAT"))
        self.force = data[:, :]

        # Show the information
        if (debugLevel >= 3):
            print("-b2--  %f" % bx[33])
            print("-b2--  %f" % by[193])
            print("-b2--  %d" % bx.shape)
            print("-b2--  %e" % bz[332, 15])
            print("-b2--  %e" % bz[4332, 15])

        # Do the coordinate transformation from M2 coordinate to Zemax coordinate
        self.bx, self.by, self.bz = M2CRS2ZCRS(bx, by, bz)

        # Data needed to determine gravitational print through and thermal deformation
        data = np.loadtxt(os.path.join(M2dir, "M2_GT_FEA.txt"), skiprows=1)

        self.zdz = data[:, 2]
        self.hdz = data[:, 3]
        self.tzdz = data[:, 4]
        self.trdz = data[:, 5]

    def getPrintthz(self, zAngle, pre_comp_elev=0):
        """
        
        Get the mirror print along z direction in specific zenith angle. 
        
        Arguments:
            zAngle {[float]} -- Zenith angle.
        
        Keyword Arguments:
            pre_comp_elev {number} -- Pre-compensation elevation angle. (default: {0})
        
        Returns:
            [float] -- Corrected projection in z direction.
        """

        # Do the M2 gravitational correction.
        # Map the changes of dz on a plane for certain zenith angle 
        printthz = self.zdz * np.cos(zAngle) + self.hdz * np.sin(zAngle)
        
        # Do the pre-compensation elevation angle correction
        printthz -= self.zdz * np.cos(pre_comp_elev) + self.hdz * np.sin(pre_comp_elev)
        
        return printthz

if __name__ == "__main__":

    pass
    