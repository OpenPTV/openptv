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


void intersect_rt (double pos1[3], double vec1[3],
				   double pos2[3], double vec2[3],
				   double *X, double *Y, double *Z)
/* intersection, given two points with direction cosines */
/* only valid, if Z1 = Z2 = 0 , which is the case after ray tracing */
{

	// ad holten, test first if lines are parallel
	if (vec1[0]/vec1[2] == vec2[0]/vec2[2] && vec1[1]/vec1[2] == vec2[1]/vec2[2]) {		// if parallel, return high values 
		*X = *Y = *Z = 1e6;						// ie, out of the measuring volume
		return;
	}

	if (fabs(vec1[1]-vec2[1]) > fabs(vec1[0]-vec2[0])) *Z = (pos2[1]-pos1[1]) / ((vec1[1]/vec1[2]) - (vec2[1]/vec2[2]));
	else						   *Z = (pos2[0]-pos1[0]) / ((vec1[0]/vec1[2]) - (vec2[0]/vec2[2]));
	
	*X = (pos1[0] + pos2[0] +  *Z * (vec1[0]/vec1[2] + vec2[0]/vec2[2])) /2;
	*Y = (pos1[1] + pos2[1] +  *Z * (vec1[1]/vec1[2] + vec2[1]/vec2[2])) /2;
}


/* see the original intersect.c in the 3dptv repository for obsolete functions */

