
#include "orientation.h"
#include "calibration.h"
#include "ray_tracing.h"
#include "vec_utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define PT_UNUSED -999

/*  skew_midpoint() finds the middle of the minimal distance segment between 
    skew rays. Undefined for parallel rays.
    
    Reference for algorithm: 
    docs/skew_midpoint.pdf

    Arguments:
    vec3d vert1, direct1 - vertex and direction unit vector of one ray.
    vec3d vert2, direct2 - vertex and direction unit vector of other ray.
    vec3d res - output buffer for midpoint result.
    
    Returns:
    The minimal distance between the skew lines.
*/
double skew_midpoint(vec3d vert1, vec3d direct1, vec3d vert2, vec3d direct2, 
    vec3d res) 
{
    vec3d perp_both, sp_diff, on1, on2, temp;
    double scale;
    
    /* vector between starting points */
    vec_subt(vert2, vert1, sp_diff);
    
    /* The shortest distance is on a line perpendicular to both rays. */
    vec_cross(direct1, direct2, perp_both);
    scale = vec_dot(perp_both, perp_both);
    
    /* position along each ray */
    vec_cross(sp_diff, direct2, temp);
    vec_scalar_mul(direct1, vec_dot(perp_both, temp)/scale, temp);
    vec_add(vert1, temp, on1);
    
    vec_cross(sp_diff, direct1, temp);
    vec_scalar_mul(direct2, vec_dot(perp_both, temp)/scale, temp);
    vec_add(vert2, temp, on2);
 
    /* Distance: */
    vec_subt(on1, on2, res);
    scale = vec_norm(res);
       
    /* Average */
    vec_add(on1, on2, res);
    vec_scalar_mul(res, 0.5, res);
    
    return scale;
}

/*  point_position() calculates an average 3D position implied by the rays
    sent toward it from cameras through the image projections of the point.
    
    Arguments:
    vec2d targets[] - for each camera, the 2D metric coordinates of the 
        identified point projection.
    int num_cams - number of cameras ( = number of elements in ``targets``).
    mm_np *multimed_pars - multimedia parameters struct for ray tracing through
        several layers.
    Calibration* cals[] - each camera's calibration object.
    vec3d res - the result average point.
    
    Returns:
    The ray convergence measure, an average of skew ray distance across all
    ray pairs.
*/
double point_position(vec2d targets[], int num_cams, mm_np *multimed_pars,
    Calibration* cals[], vec3d res)
{
    int cam, pair; /* loop counters */
    int num_used_pairs = 0; /* averaging accumulators */
    double dtot = 0;
    vec3d point_tot = {0., 0., 0.};
    
    vec2d current;
    vec3d* vertices = (vec3d *) calloc(num_cams, sizeof(vec3d));
    vec3d* directs = (vec3d *) calloc(num_cams, sizeof(vec3d));
    vec3d point;
    
    /* Shoot rays from all cameras. */
    for (cam = 0; cam < num_cams; cam++) {
        if (targets[cam][0] != PT_UNUSED) {
            current[0] = targets[cam][0] - cals[cam]->int_par.xh;
            current[1] = targets[cam][1] - cals[cam]->int_par.yh;
            
            ray_tracing(current[0], current[1], cals[cam], *multimed_pars,
                vertices[cam], directs[cam]);
        }
    }
    
    /* Check intersection distance for each pair of rays and find position */
    for (cam = 0; cam < num_cams; cam++) {
        if (targets[cam][0] == PT_UNUSED) continue;
        
        for (pair = cam + 1; pair < num_cams; pair++) {
            if (targets[pair][0] == PT_UNUSED) continue;
            
            num_used_pairs++;
            dtot += skew_midpoint(vertices[cam], directs[cam],
                vertices[pair], directs[pair], point);
            vec_add(point_tot, point, point_tot);
        }
    }
    
    free(vertices);
    free(directs);
    
    vec_scalar_mul(point_tot, 1./num_used_pairs, res);
    return (dtot / num_used_pairs);
}

/*  epipolar_convergence() Finds how closely epipolar lines converge to find
    true 3D positions of known points. 
    
    Arguments:
    (vec2d **) targets - 2D array of targets, so order 3 tensor of shape
        (num_targs,num_cams,2). Each target is the 2D metric coordinates of 
        one identified point.
    int num_targs - the number of known targets, assumed to be the same in all
        cameras.
    int num_cams - number of cameras.
    mm_np *multimed_pars - multimedia parameters struct for ray tracing through
        several layers.
    Calibration* cals[] - each camera's calibration object.
*/
double epipolar_convergence(vec2d** targets, int num_targs, int num_cams,
    mm_np *multimed_pars, Calibration* cals[]) 
{
    int pt;
    double dtot = 0;
    vec3d res;
    
    for (pt = 0; pt < num_targs; pt++) {
        dtot += point_position(targets[pt], num_cams, multimed_pars, cals, res);
    } /* end of per-point iteration */
    
    return (dtot / num_targs);
}

