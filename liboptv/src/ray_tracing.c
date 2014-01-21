/***************************************************************
Routine:                ray_tracing

Author/Copyright:       Hans -  Gerd Maas

Address:                Institute of Geodesy and Photogrammetry
                        ETH  -   Hoenggerberg
                        CH  -   8093 Zurich

Creation Date:          21.4.88
    
Description:            traces one ray, given by image coordinates,
                    exterior and interior orientation 
                    through dufferent media
                    (see Hoehle, Manual of photogrammetry)
    
Routines contained:      -  
***************************************************************/

#include "ray_tracing.h"

/*  wraps previous ray_tracing, parameters are read directly from control_par* structure */
void ray_tracing (double x
				, double y
				, Calibration *cal
				, mm_np mm
				, double *X
				, double *a){

old_ray_tracing(x
			, y
			, cal->ext_par
			, cal->int_par
			, cal->glass_par
			, mm
			, &X[0]
			, &X[1]
			, &X[2]
			, &a[0]
			, &a[1]
			, &a[2]);
}



void old_ray_tracing (double x, double y, Exterior Ex, Interior I, Glass G,\
mm_np mm, double  *Xb2, double  *Yb2, double  *Zb2, \
double  *a3, double  *b3,double  *c3){

/*   ray -  tracing, see HOEHLE and Manual of Photogrammetry  */

    double a1, b1, c1, a2, b2, c2, Xb1, Yb1, Zb1, d1, d2, cosi1, cosi2;
    double vect1[3], vect2[3], factor, s2;
    double a[3],b[3],base2[3],c,dummy,bn[3],bp[3],n,p;

    s2  = sqrt (x * x + y * y + I.cc * I.cc);
    
    /*   direction cosines in image coordinate system  */
    vect1[0] = x/s2;
    vect1[1] = y/s2;
    vect1[2] =- I.cc/s2;

    matmul (vect2, (double *)Ex.dm, vect1, 3,3,1, 3,3);
    
    /*   direction cosines in space coordinate system , medium n1  */
    a1 = vect2[0];
    b1 = vect2[1];
    c1 = vect2[2];
    
    
    a[0] = Ex.x0; 
    a[1] = Ex.y0; 
    a[2] = Ex.z0;
    b[0] = vect2[0];
    b[1] = vect2[1];
    b[2] = vect2[2];
    c  =  sqrt(G.vec_x * G.vec_x + G.vec_y * G.vec_y + G.vec_z * G.vec_z);
    base2[0] = G.vec_x/c; 
    base2[1] = G.vec_y/c; 
    base2[2] = G.vec_z/c;

    c = c + mm.d[0];
    dummy = base2[0] * a[0] + base2[1] * a[1] + base2[2] * a[2];
    dummy = dummy - c;
    d1 =- dummy/(base2[0] * b[0] + base2[1] *  b[1] + base2[2] * b[2]);
    

    /*   point on the horizontal plane between n1,n2  */
    Xb1 = a[0] +  b[0] *  d1;
    Yb1 = a[1] +  b[1] *  d1;
    Zb1 = a[2] +  b[2] *  d1;
    
    bn[0] = base2[0];
    bn[1] = base2[1];
    bn[2] = base2[2];
    n = (b[0] *  bn[0] +  b[1] *  bn[1] +  b[2] *  bn[2]);
    bp[0] = b[0] -  bn[0] *  n;
    bp[1] = b[1] -  bn[1] *  n;
    bp[2] = b[2] -  bn[2] *  n;
    dummy = sqrt(bp[0] *  bp[0] +  bp[1] *  bp[1] +  bp[2] *  bp[2]);
    bp[0] = bp[0]/dummy;
    bp[1] = bp[1]/dummy;
    bp[2] = bp[2]/dummy;

    p = sqrt(1 -  n *  n);
    /*   interface parallel  */
    p  =  p  *   mm.n1/mm.n2[0];
    /*   interface normal  */
    n =  -sqrt(1 -  p * p);
    a2 = p * bp[0] + n * bn[0];
    b2 = p * bp[1] + n * bn[1];
    c2 = p * bp[2] + n * bn[2];
    d2 = mm.d[0]/fabs((base2[0] *  a2 + base2[1] * b2 + base2[2] * c2));
    

    /*   point on the horizontal plane between n2,n3  */
     *Xb2 = Xb1 + d2 * a2;   
     *Yb2 = Yb1 + d2 * b2;   
     *Zb2 = Zb1 + d2 * c2;
    
    n = (a2 *  bn[0] +  b2 *  bn[1] +  c2 *  bn[2]);
    bp[0] = a2 - bn[0] * n;
    bp[1] = b2 - bn[1] * n;
    bp[2] = c2 - bn[2] *  n;
    dummy = sqrt(bp[0] * bp[0] + bp[1] * bp[1] + bp[2] * bp[2]);
    bp[0] = bp[0] / dummy;
    bp[1] = bp[1] / dummy;
    bp[2] = bp[2] / dummy;

    p = sqrt(1 - n * n);
    p = p * mm.n2[0]/mm.n3;
    n = -sqrt(1 - p * p);
    *a3 = p * bp[0] + n * bn[0];
    *b3 = p * bp[1] + n * bn[1];
    *c3 = p * bp[2] + n * bn[2];
}

