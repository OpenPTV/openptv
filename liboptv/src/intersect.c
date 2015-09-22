#include "intersect.h"

/* Intersection of the imaging rays of 2 cameras 
 * Arguments:
 * pos1,pos2 - double vectors of 3 x 1, positions of the points on the two cameras 
 * vec1, vec2 - double vectors of 3 x 1, direction cosines of the imaging axis vectors
 * X - double vector of 3 x 1, the point of intersection in space.
 * Originally developed by Hans-Gerd Maas, ETH Zurich
 * 2 * (point + direction cosines) ==> intersection
 * only valid, if Z1 = Z2 = 0 , which is the case after ray tracing
*/

void intersect_rt (double pos1[3], double vec1[3],
				   double pos2[3], double vec2[3],
				   double X[3])

{

	/* ad holten, test first if lines are parallel 
	 * if parallel, return high values
	 * ie, out of the measuring volume 
	*/
	if (vec1[0]/vec1[2] == vec2[0]/vec2[2] && vec1[1]/vec1[2] == vec2[1]/vec2[2]) {
		X[0] = X[1] = X[2] = 1e6;
		return;
	}

	if (fabs(vec1[1]-vec2[1]) > fabs(vec1[0]-vec2[0])) X[2] = (pos2[1]-pos1[1]) / ((vec1[1]/vec1[2]) - (vec2[1]/vec2[2]));
	else						   X[2] = (pos2[0]-pos1[0]) / ((vec1[0]/vec1[2]) - (vec2[0]/vec2[2]));
	
	X[0] = (pos1[0] + pos2[0] +  X[2] * (vec1[0]/vec1[2] + vec2[0]/vec2[2])) /2;
	X[1] = (pos1[1] + pos2[1] +  X[2] * (vec1[1]/vec1[2] + vec2[1]/vec2[2])) /2;
	
}


/* see the original intersect.c in the 3dptv repository for obsolete functions */

