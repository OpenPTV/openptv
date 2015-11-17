/***************************************************************
Routine:                ray_tracing
Author/Copyright:       Hans -  Gerd Maas
Updated for liboptv: Alex Liberzon
Modification Date:     April 20, 2014

References:
[1] HOEHLE
[2] Manual of Photogrammetry
***************************************************************/

#include <stdio.h>
#include "ray_tracing.h"

/*  ray_tracing () traces the optical ray through the multi-media interface of
    (presently) three layers, typically air - glass - water, and returns the
    position of the ray crossing point and the vector normal to the interface.
    See refs. [1,2]
    
    Arguments:
    double x, y - metric position of a point in the image space 
    Calibration *cal - parameters of a specific camera.
    mm_np mm - multi-media information (thickness, index of refraction)
    
    Output Arguments:
    vec3d X - crossing point position.
    vec3d out - vector pointing normal to the interface.
*/

void ray_tracing (double x, double y, Calibration* cal, mm_np mm, vec3d X,
    vec3d out)
{
    double d1, d2, c, dist_cam_glass, n, p;
    vec3d start_dir, primary_point, glass_dir, bp, tmp1, tmp2, Xb, a2;
    
    /* Initial ray direction in global coordinate system  */
    vec_set(tmp1, x, y, -1*cal->int_par.cc);
    unit_vector(tmp1, tmp1);
    matmul (start_dir, (double *)cal->ext_par.dm, tmp1, 3,3,1, 3,3);
    
    vec_set(primary_point, cal->ext_par.x0, cal->ext_par.y0, cal->ext_par.z0);    

    vec_set(tmp1, cal->glass_par.vec_x, cal->glass_par.vec_y, cal->glass_par.vec_z);
    unit_vector(tmp1, glass_dir);
    c = vec_norm(tmp1) + mm.d[0];
    
    /* Project start ray on glass vector to find n1/n2 interface. */
    dist_cam_glass = vec_dot(glass_dir, primary_point) - c;
    d1 = -dist_cam_glass/vec_dot(glass_dir, start_dir); 
    vec_scalar_mul(start_dir, d1, tmp1);
    vec_add(primary_point, tmp1, Xb);
    
    /* Break down ray into glass-normal and glass-parallel components. */
    n = vec_dot(start_dir, glass_dir);
    vec_scalar_mul(glass_dir, n, tmp1);
    
    vec_subt(start_dir, tmp1, tmp2);
    unit_vector(tmp2, bp);
    
    /* Transform to direction inside glass, using Snell's law */
    p = sqrt(1 -  n *  n) * mm.n1/mm.n2[0]; /* glass parallel */
    n =  -sqrt(1 - p * p); /* glass normal */
    
    /* Propagation length in glass parallel to glass vector */
    vec_scalar_mul(bp, p, tmp1); 
    vec_scalar_mul(glass_dir, n, tmp2); 
    vec_add(tmp1, tmp2, a2); 
    d2 = mm.d[0]/fabs(vec_dot(glass_dir, a2));        

    /*   point on the horizontal plane between n2,n3  */
    vec_scalar_mul(a2, d2, tmp1);
    vec_add(Xb, tmp1, X);

    /* Again, direction in next medium */
    n = vec_dot(a2, glass_dir);
    vec_subt(a2, tmp2, tmp2);
    unit_vector(tmp2, bp);

    p = sqrt(1 - n * n);
    p = p * mm.n2[0]/mm.n3;
    n = -sqrt(1 - p * p);
    
    vec_scalar_mul(bp, p, tmp1);
    vec_scalar_mul(glass_dir, n, tmp2);
    vec_add (tmp1, tmp2, out);
}

