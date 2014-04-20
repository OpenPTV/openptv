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

Updated for liboptv: Alex Liberzon
Modification Date:     April 20, 2014
***************************************************************/

#include <stdio.h>
#include "ray_tracing.h"


#define EPS 1e-5
void unit_vector();

/* returns a unit vector, normalized by the norm */
void unit_vector(double a[3]){
	double dummy; 
	
	dummy = sqrt(a[0] *  a[0] +  a[1] *  a[1] +  a[2] *  a[2]);
	
	/* if the vector is zero length we return zero vector back */
	if (dummy < EPS) dummy = 1.0;
	
    a[0] = a[0]/dummy;
    a[1] = a[1]/dummy;
    a[2] = a[2]/dummy;
}


/* tracers the optical ray through the multi-media interface of three layers, typically
air - glass - water, returns the two vectors: 
position of the ray crossing point and the vector normal to the interface 
*/

void ray_tracing (double x
				, double y
				, Calibration* cal
				, mm_np mm
				, double X[3]
				, double a[3]){

/*   ray -  tracing, see HOEHLE and Manual of Photogrammetry  */

    double a1, b1, c1, a2, b2, c2, Xb1, Yb1, Zb1, d1, d2;
    double vect1[3], vect2[3], factor, s2;
    double b[3],base2[3],c, dummy,bn[3],bp[3],n,p;
    

    s2  = sqrt (x * x + y * y + cal->int_par.cc * cal->int_par.cc);
        
    
    /*   direction cosines in image coordinate system  */
    vect1[0] = x/s2;
    vect1[1] = y/s2;
    vect1[2] =- cal->int_par.cc/s2;
    

    matmul (vect2, (double *)cal->ext_par.dm, vect1, 3,3,1, 3,3);
    
    
    /*   direction cosines in space coordinate system , medium n1  */
    a1 = vect2[0];
    b1 = vect2[1];
    c1 = vect2[2];
    
    
    a[0] = cal->ext_par.x0; 
    a[1] = cal->ext_par.y0; 
    a[2] = cal->ext_par.z0;
    
    b[0] = vect2[0];
    b[1] = vect2[1];
    b[2] = vect2[2];
    
    c  =  sqrt(
    cal->glass_par.vec_x * cal->glass_par.vec_x + \
    cal->glass_par.vec_y * cal->glass_par.vec_y + \
    cal->glass_par.vec_z * cal->glass_par.vec_z);
    
    base2[0] = cal->glass_par.vec_x/c; 
    base2[1] = cal->glass_par.vec_y/c; 
    base2[2] = cal->glass_par.vec_z/c;
    
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
    
     
    
     /*
    dummy = sqrt(bp[0] *  bp[0] +  bp[1] *  bp[1] +  bp[2] *  bp[2]);
    bp[0] = bp[0]/dummy;
    bp[1] = bp[1]/dummy;
    bp[2] = bp[2]/dummy;
    */ 
    unit_vector(bp);
    

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
     X[0] = Xb1 + d2 * a2;   
     X[1] = Yb1 + d2 * b2;   
     X[2] = Zb1 + d2 * c2;
         
    n = (a2 *  bn[0] +  b2 *  bn[1] +  c2 *  bn[2]);
    bp[0] = a2 - bn[0] * n;
    bp[1] = b2 - bn[1] * n;
    bp[2] = c2 - bn[2] * n;
    
        
    /*
    dummy = sqrt(bp[0] * bp[0] + bp[1] * bp[1] + bp[2] * bp[2]);
    bp[0] = bp[0] / dummy;
    bp[1] = bp[1] / dummy;
    bp[2] = bp[2] / dummy;
    */
    unit_vector(bp);

    p = sqrt(1 - n * n);
    p = p * mm.n2[0]/mm.n3;
    n = -sqrt(1 - p * p);
    
    
    a[0] = p * bp[0] + n * bn[0];
    a[1] = p * bp[1] + n * bn[1];
    a[2] = p * bp[2] + n * bn[2];

}




