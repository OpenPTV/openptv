/****************************************************************************

Routine:				intersect.c

Author/Copyright:		Hans-Gerd Maas

Address:				Institute of Geodesy and Photogrammetry
						ETH - Hoenggerberg
						CH - 8093 Zurich

Creation Date:			6.4.88
	
Description:			2 * (point + direction cosines) ==> intersection
	
Routines contained: 	-

****************************************************************************/
#include "intersect.h"


/*-------------------------------------------------------------------------*/
/* 2 cameras */


void intersect_rt (double X1, double Y1, double Z1, double a1, double b1, double c1,
				   double X2, double Y2, double Z2, double a2, double b2, double c2,
				   double *X, double *Y, double *Z)
/* intersection, given two points with direction cosines */
/* only valid, if Z1 = Z2 = 0 , which is the case after ray tracing */
{
	// ad holten, test first if lines are parallel
	if (a1/c1 == a2/c2 && b1/c1 == b2/c2) {		// if parallel, return high values 
		*X = *Y = *Z = 1e6;						// ie, out of the measuring volume
		return;
	}

	if (fabs(b1-b2) > fabs(a1-a2)) *Z = (Y2-Y1) / ((b1/c1) - (b2/c2));
	else						   *Z = (X2-X1) / ((a1/c1) - (a2/c2));
	
	*X = (X1 + X2  +  *Z * (a1/c1 + a2/c2)) /2;
	*Y = (Y1 + Y2  +  *Z * (b1/c1 + b2/c2)) /2;
}




