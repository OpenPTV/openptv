/****************************************************************************

Routine:	       	rotation.c

Author/Copyright:      	Hans-Gerd Maas

Address:	       	Institute of Geodesy and Photogrammetry
		       	ETH - Hoenggerberg
		       	CH - 8093 Zurich

Creation Date:	       	21.4.88
	
Description:	       	computes the rotation matrix for given three 
		       	angles omega, phi, kappa of exterior orientation
		      	(see: Kraus)
	
Routines contained:		-

****************************************************************************/

#include <math.h>
#include "calibration.h"

void rotation_matrix (Ex, dm)

Exterior  Ex;
Dmatrix   dm;

{
    dm[0][0] = cos(Ex.phi) * cos(Ex.kappa);
    dm[0][1] = (-1) * cos(Ex.phi) * sin(Ex.kappa);
    dm[0][2] = sin(Ex.phi);
    dm[1][0] = cos(Ex.omega) * sin(Ex.kappa)
             + sin(Ex.omega) * sin(Ex.phi) * cos(Ex.kappa);
    dm[1][1] = cos(Ex.omega) * cos(Ex.kappa)
             - sin(Ex.omega) * sin(Ex.phi) * sin(Ex.kappa);
    dm[1][2] = (-1) * sin(Ex.omega) * cos(Ex.phi);
    dm[2][0] = sin(Ex.omega) * sin(Ex.kappa)
             - cos(Ex.omega) * sin(Ex.phi) * cos(Ex.kappa);
    dm[2][1] = sin(Ex.omega) * cos(Ex.kappa)
             + cos(Ex.omega) * sin(Ex.phi) * sin(Ex.kappa);
    dm[2][2] = cos(Ex.omega) * cos(Ex.phi);
}

