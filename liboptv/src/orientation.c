
#include "orientation.h"
#include "calibration.h"
#include "ray_tracing.h"
#include "vec_utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define PT_UNUSED -999

/*  ray_distance() calculates the minimal distance between skew rays.
    Reference for algorithm: 
    http://mathworld.wolfram.com/Line-LineDistance.html
    
    Arguments:
    vec3d vert1, direct1 - vertex and direction unit vector of one ray.
    vec3d vert2, direct2 - vertex and direction unit vector of other ray.
*/
double ray_distance(vec3d vert1, vec3d direct1, vec3d vert2, vec3d direct2) {
    vec3d perp_both, sp_diff, proj;
    double scale;
    
    /* vector between starting points */
    vec_subt(vert2, vert1, sp_diff);
    
    /* Check for parallel rays. Then use planar geometry. */
    if (vec_cmp(direct1, direct2)) {
        scale = vec_dot(sp_diff, direct1);
        vec_scalar_mul(direct1, scale, proj);
        vec_subt(sp_diff, proj, perp_both);
        return vec_norm(perp_both);
    }
    
    /* The shortest distance is on a line perpendicular to both rays. */
    vec_cross(direct1, direct2, perp_both);
    
    return fabs(vec_dot(sp_diff, perp_both))/vec_norm(perp_both);
}


/*  epipolar_convergence() Finds how closely epipolar lines converge to find
    true 3D positions of known points. 
    
    Arguments:
    (vec2d *) targets - each element points a 2D array for one camera, each 
        row within is the 2D metric coordinates of one identified point.
    int num_targs - the number of known targets, assumed to be the same in all
        cameras.
    int num_cams - number of cameras ( = number of elements in ``targets``).
    mm_np *multimed_pars - multimedia parameters struct for ray tracing through
        several layers.
    Calibration* cals[] - each camera's calibration object.
*/
double epipolar_convergence(vec2d* targets[], int num_targs, int num_cams,
    mm_np *multimed_pars, Calibration* cals[]) 
{
    int pt, cam, pair; /* loop counters */
    int num_used_pairs = 0; /* averaging accumulators */
    double dtot = 0;
    
    vec2d current;
    vec3d* vertices = (vec3d *) calloc(num_cams, sizeof(vec3d));
    vec3d* directs = (vec3d *) calloc(num_cams, sizeof(vec3d));
    
    for (pt = 0; pt < num_targs; pt++) {
        /* Shoot rays from all cameras. */
        for (cam = 0; cam < num_cams; cam++) {
            if (targets[cam][pt][0] != PT_UNUSED) {
                current[0] = targets[cam][pt][0] - cals[cam]->int_par.xh;
                current[1] = targets[cam][pt][1] - cals[cam]->int_par.yh;
                
                ray_tracing(current[0], current[1], cals[cam], *multimed_pars,
                    vertices[cam], directs[cam]);
            }
        }
        
        /* Check intersection distance for each pair of rays */
        for (cam = 0; cam < num_cams; cam++) {
            if (targets[cam][pt][0] == PT_UNUSED) continue;
            
            for (pair = cam + 1; pair < num_cams; pair++) {
                if (targets[pair][pt][0] == PT_UNUSED) continue;
                
                num_used_pairs++;
                dtot += ray_distance(vertices[cam], directs[cam],
                    vertices[pair], directs[pair]);
            }
        }
    } /* end of per-point iteration */
    
    free(vertices);
    free(directs);
    
    return (dtot / num_used_pairs);
}

