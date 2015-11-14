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


/* ray_tracing () traces the optical ray through the multi-media interface of 
   (presently) three layers, typically air - glass - water, and returns the two vectors: 
    position of the ray crossing point and the vector normal to the interface 
    Arguments:
    x,y - doubles, position of a point in the image space 
    Calibration *cal structure of a specific camera
    mm_np mm structure of the multi-media information (thickness, index of refraction)
    Output:
    X - vector (vec3d) of the crossing point position
    out - vector pointing normal to the interface 
*/

void ray_tracing (double x
				, double y
				, Calibration* cal
				, mm_np mm
				, vec3d X
				, vec3d out){

/*   ray -  tracing, see HOEHLE and Manual of Photogrammetry  */

    double d1, d2, c, dummy, n, p;
    vec3d vect1, vect2, a, base2, bn, bp, tmp1, tmp2, Xb, a2;
    
    /*   direction cosines in image coordinate system  */
    vec_set(tmp1, x, y, -1*cal->int_par.cc);
    unit_vector(tmp1, vect1);

    matmul (vect2, (double *)cal->ext_par.dm, vect1, 3,3,1, 3,3);
    
    vec_set(a, cal->ext_par.x0, cal->ext_par.y0, cal->ext_par.z0);    

    /* base2 is the unit vector in glass direction */
    vec_set(tmp1, cal->glass_par.vec_x, cal->glass_par.vec_y, cal->glass_par.vec_z);
    
    unit_vector(tmp1, base2);
    c = vec_norm(tmp1) + mm.d[0];
    
    dummy = vec_dot(base2, a) - c;
    d1 = -1*dummy/vec_dot(base2, vect2); 
        

    /*   point on the horizontal plane between n1,n2  */
    vec_scalar_mul(vect2, d1, tmp1);
    vec_add(a, tmp1, Xb);

        
    vec_copy(bn, base2); 
    n = vec_dot(vect2, bn);
    
    vec_scalar_mul(bn, n, tmp1);
    vec_subt(vect2, tmp1, tmp2);
    unit_vector(tmp2, bp);
    

    p = sqrt(1 -  n *  n);
    /*   interface parallel  */
    p  =  p  *   mm.n1/mm.n2[0];
    /*   interface normal  */
    n =  -sqrt(1 -  p * p);
    
    
    vec_scalar_mul(bp, p, tmp1); 
    vec_scalar_mul(bn, n, tmp2); 
    vec_add(tmp1, tmp2, a2); 
    d2 = mm.d[0]/fabs(vec_dot(base2, a2));        
    

    /*   point on the horizontal plane between n2,n3  */
    vec_scalar_mul(a2, d2, tmp1);
    vec_add(Xb, tmp1, X);

    n = vec_dot(a2, bn);
    vec_scalar_mul(bn, n, tmp1);
    vec_subt(a2, tmp1, tmp2);      
    unit_vector(tmp2, bp);

    p = sqrt(1 - n * n);
    p = p * mm.n2[0]/mm.n3;
    n = -sqrt(1 - p * p);
    
    vec_scalar_mul(bp, p, tmp1);
    vec_scalar_mul(bn, n, tmp2);
    vec_add (tmp1, tmp2, out);
}
