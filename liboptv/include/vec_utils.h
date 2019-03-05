/* Utilities for handling simple 3d vectors implemented as an array of 3
doubles.
*/

#ifndef VEC_UTILS_H
#define VEC_UTILS_H

#include <math.h>

#ifdef NAN
    #define EMPTY_CELL NAN
#else
    #if _MSC_VER <= 1500 // Visual C 2008 - for Python 2.7, or earlier
        #define MSVC_NAN_REQUIRED
        double return_nan(void);
        #define EMPTY_CELL return_nan()
    #else // More modern compilers or non Visual Studio
        #define EMPTY_CELL 0.0/0.0
    #endif
#endif

#define is_empty(x) isnan(x)
#define norm(x,y,z) sqrt((x)*(x) + (y)*(y) + (z)*(z))

typedef double vec3d[3];

void vec_init(vec3d init);
void vec_set(vec3d dest, double x, double y, double z);
void vec_copy(vec3d dest, vec3d src);
void vec_subt(vec3d from, vec3d sub, vec3d output);
void vec_add(vec3d vec1, vec3d vec2, vec3d output);
void vec_scalar_mul(vec3d vec, double scalar, vec3d output);
double vec_norm(vec3d vec);
double vec_diff_norm(vec3d vec1, vec3d vec2);
double vec_dot(vec3d vec1, vec3d vec2);
void vec_cross(vec3d vec1, vec3d vec2, vec3d out);
int vec_cmp(vec3d vec1, vec3d vec2);
int vec_approx_cmp(vec3d vec1, vec3d vec2, double eps);
void unit_vector(vec3d vec, vec3d out);


#endif

