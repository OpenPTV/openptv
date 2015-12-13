
#include "orientation.h"
#include "calibration.h"
#include "ray_tracing.h"
#include "vec_utils.h"
#include <stdlib.h>

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
    
    return vec_dot(sp_diff, perp_both)/vec_norm(perp_both);
}

