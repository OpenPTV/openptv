/****************************************************************************

Routine:				ray_tracing

Author/Copyright:		Hans-Gerd Maas

Address:		       	Institute of Geodesy and Photogrammetry
			       		ETH - Hoenggerberg
			       		CH - 8093 Zurich

Creation Date:			21.4.88
Modification date: 		August 4, 2013, Alex Liberzon
						ray_tracing_v2 is the only functions used
						
	
Description:	       	traces one ray, given by image coordinates,
			       	exterior and interior orientation 
			       	through dufferent media
			       	(see Hoehle, Manual of photogrammetry)
	
Routines contained:		-

****************************************************************************/


#include "ray_tracing.h"



/* removed point_line_line from ray_tracing - it is never used here */

/* Normalized cross product of two vectors
*
* Arguments:
*	double a[3], double b[3] - two vectors of 3 x 1 
* Returns:
* 	double *n1, *n2, *n3 - three components of the cross product vector. 
*  Note: the vector is normalized to unity length
*
* TODO: replace by standard vec_utils subroutines 
*/

void norm_cross(double a[3], double b[3], double *n1, double *n2, double *n3) {

//Beat Lüthi Nov 2008

	double  res[3], dummy, norm;

	res[0]=a[1]*b[2]-a[2]*b[1];
	res[1]=a[2]*b[0]-a[0]*b[2];
	res[2]=a[0]*b[1]-a[1]*b[0];
	
	// norm = sqrt( (res[0]*res[0]) + (res[1]*res[1]) + (res[2]*res[2]) );
	
	// printf("Res: %6.3f %6.3f %6.3f\n", res[0],res[1],res[2]);
	
	modu(res,&norm);
	
	// printf(" Norm : %6.3f \n",norm);
	
	// fixing the zero length norm bug:
	// Alex, Aug 3, 2013
	// Thanks for the test suite :)
	
	if (norm == 0.0){
		*n1 = res[0];
		*n2 = res[0];
		*n3 = res[0];
	} else {	
	*n1=res[0]/norm;
	*n2=res[1]/norm;
	*n3=res[2]/norm;
	}
}

/* Beat Lüthi Nov 2008
* Dot product of two vectors 
* TODO: use ready subroutines from vec_utils.h
*
*/

void dot(double a[3], double b[3], double *d) {

	*d = a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}


/* Modulus of a vector
* TODO: use ready subroutine called norm in vec_utils.h
*/
//Beat Lüthi Nov 2008
void modu(double a[3], double *m) {

	*m = sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
}


/* ray_tracing_v2 
*
*
* 
*/


void ray_tracing_v2 (double x, double y, Exterior Ex, Interior I, Glass G, mm_np mm,\
double *Xb2, double *Yb2, double *Zb2, double *a3, double *b3, double *c3) {
/* ray-tracing, see HOEHLE and Manual of Photogrammetry */

	double  a1, b1, c1, a2, b2, c2, Xb1, Yb1, Zb1, d1, d2, cosi1, cosi2,
			vect1[3], vect2[3], factor, s2;

	double a[3],b[3],base2[3],c,dummy,bn[3],bp[3],n,p;

	s2 = sqrt (x*x + y*y + I.cc*I.cc);
	
	/* direction cosines in image coordinate system */
	vect1[0] = x/s2;  vect1[1] = y/s2;	vect1[2] = -I.cc/s2;

	matmul (vect2, (double *) Ex.dm, vect1, 3,3,1);
 	
	/* direction cosines in space coordinate system , medium n1 */
	a1 = vect2[0];  b1 = vect2[1];  c1 = vect2[2];  
	
    //old d1 = -(Ex.z0 - mm.d[0]) / c1;
    //find dist to outer interface
	//...   from Jakob Mann vector3 XLinePlane(vector3 a, vector3 b, struct plane pl)
    //...   a + b*((pl.c - dot(pl.base[2],a))/dot(pl.base[2],b));
    
	/*Ex.x0=0.;
    Ex.y0=20.;
    Ex.z0=10.;
    Ex.omega=-0.7853981;
    Ex.phi=0.;
    Ex.kappa=0.;
    G.vec_x=0.;
    G.vec_y=10.;
    G.vec_z=0.;
	vect2[0]=0.;
	vect2[1]=-1./sqrt(2.);
	vect2[2]=-1./sqrt(2.);*/
   
	
	a[0] = Ex.x0; a[1] = Ex.y0; a[2] = Ex.z0;
	b[0] = vect2[0]; b[1] = vect2[1]; b[2] = vect2[2];
	c = sqrt(G.vec_x*G.vec_x + G.vec_y*G.vec_y + G.vec_z*G.vec_z);
	base2[0] = G.vec_x/c; base2[1] = G.vec_y/c; base2[2] = G.vec_z/c;

	c = c + mm.d[0];
	dummy = base2[0]*a[0]+base2[1]*a[1]+base2[2]*a[2];
	dummy = dummy-c;
	d1 = -dummy/(base2[0]*b[0]+base2[1]*b[1]+base2[2]*b[2]);
	

	/* point on the horizontal plane between n1,n2 */
	//old Xb1 = Ex.x0 + d1*a1;  Yb1 = Ex.y0 + d1*b1;  Zb1 = Ex.z0 + d1*c1;
	Xb1 = a[0] + b[0]*d1;
	Yb1 = a[1] + b[1]*d1;
	Zb1 = a[2] + b[2]*d1;
	
	//old cosi1 = c1;
	//cosi1=base2[0]*b[0]+base2[1]*b[1]+base2[2]*b[2];
	//factor = cosi1 * mm.n1/mm.n2[0]
	//	   + sqrt (1 - (mm.n1*mm.n1)/(mm.n2[0]*mm.n2[0])
	//	   			 + (cosi1*cosi1)*(mm.n1*mm.n1)/(mm.n2[0]*mm.n2[0]));

	/* direction cosines in space coordinate system , medium n2 */
	//old a2 = a1 * mm.n1/mm.n2[0];
	//old b2 = b1 * mm.n1/mm.n2[0];
	//old c2 = c1 * mm.n1/mm.n2[0] - factor;
	
	//old d2 = -mm.d[0]/c2;

    bn[0] = base2[0]; bn[1] = base2[1]; bn[2] = base2[2];
	n = (b[0]*bn[0]+b[1]*bn[1]+b[2]*bn[2]);
	bp[0] = b[0] - bn[0]*n; bp[1] = b[1] - bn[1]*n; bp[2] = b[2] - bn[2]*n;
	dummy = sqrt(bp[0]*bp[0] + bp[1]*bp[1] + bp[2]*bp[2]);
	bp[0] = bp[0]/dummy;bp[1]=bp[1]/dummy;bp[2]=bp[2]/dummy;

	p = sqrt(1 - n*n);
	p = p * mm.n1/mm.n2[0];//interface parallel
	//n = n * mm.n1/mm.n2[0] - factor;//interface normal
	n =-sqrt(1-p*p);
	a2 = p*bp[0]+n*bn[0];
	b2 = p*bp[1]+n*bn[1];
	c2 = p*bp[2]+n*bn[2];
    d2 = mm.d[0]/fabs((base2[0]*a2+base2[1]*b2+base2[2]*c2));
	

	/* point on the horizontal plane between n2,n3 */
	*Xb2 = Xb1 + d2*a2;  *Yb2 = Yb1 + d2*b2;  *Zb2 = Zb1 + d2*c2;
	
	//old cosi2 = c2;
	//cosi2=base2[0]*a2+base2[1]*b2+base2[2]*c2;
	//factor = cosi2 * mm.n2[0]/mm.n3 
	//	   + sqrt (1 - (mm.n2[0]*mm.n2[0])/(mm.n3*mm.n3)
	//	   			 + (cosi2*cosi2)*(mm.n2[0]*mm.n2[0])/(mm.n3*mm.n3));

	/* direction cosines in space coordinate system , medium mm.n3 */
	//old *a3 = a2 * mm.n2[0]/mm.n3;
	//old *b3 = b2 * mm.n2[0]/mm.n3;
	//old *c3 = c2 * mm.n2[0]/mm.n3 - factor;

	n=(a2*bn[0]+b2*bn[1]+c2*bn[2]);
	bp[0]=a2-bn[0]*n;bp[1]=b2-bn[1]*n;bp[2]=c2-bn[2]*n;
	dummy=sqrt(bp[0]*bp[0]+bp[1]*bp[1]+bp[2]*bp[2]);
	bp[0]=bp[0]/dummy;bp[1]=bp[1]/dummy;bp[2]=bp[2]/dummy;

	p=sqrt(1-n*n);
	p = p * mm.n2[0]/mm.n3;//interface parallel
	//n = n * mm.n2[0]/mm.n3 - factor;//interface normal
	n = -sqrt(1-p*p);
	*a3=p*bp[0]+n*bn[0];
	*b3=p*bp[1]+n*bn[1];
	*c3=p*bp[2]+n*bn[2];
}

