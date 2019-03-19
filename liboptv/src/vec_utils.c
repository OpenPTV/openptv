/* implemnentation of vec_utils.h 

Implementation detail: yes, we use loops. In this day and age, compilers
can optimize this away at the cost of code size, so it is up to the build 
system to decide whether to invest in loop peeling etc. Here we write
the logical structure, and allow optimizing for size as well.
*/

#include "vec_utils.h"
#include <math.h>

#ifdef MSVC_NAN_REQUIRED

/* Returns a NAN, which is surprisingly  non-trivial on Visual C for Python 2.7 */

static const unsigned long _explicit_dNAN[2] = {0x00000000, 0x7ff80000};
double return_nan(void) {
    return *( double* )_explicit_dNAN;
}

#endif

/*  vec_init() initializes all components of a 3D vector to NaN.
    
    Arguments:
    vec3d init - the vector to initialize
*/
void vec_init(vec3d init) {
    int ix;
    for (ix = 0; ix < 3; ix ++) init[ix] = EMPTY_CELL;
}

/*  vec_set() sets the components of a  3D vector from separate doubles.

    Arguments:
    vec3d dest - the destination of setting, the vector to overwrite.
    double x, y, z - new components for the vector.
*/
void vec_set(vec3d dest, double x, double y, double z) {
    dest[0] = x;
    dest[1] = y;
    dest[2] = z;
}

/*  vec_copy() copies one 3D vector into another.

    Arguments:
    vec3d dest - the destination of copy, the vector to overwrite.
    vec3d src - the vector to copy.
*/
void vec_copy(vec3d dest, vec3d src) {
    int ix;
    for (ix = 0; ix < 3; ix ++) dest[ix] = src[ix];
}

/*  vec_subt() subtracts two 3D vectors.
    
    Arguments:
    vec3d from, sub, output - result is output[i] = from[i] - sub[i]
*/
void vec_subt(vec3d from, vec3d sub, vec3d output) {
    int ix;
    for (ix = 0; ix < 3; ix ++) output[ix] = from[ix] - sub[ix];
}

/*  vec_add() adds two 3D vectors.
    
    Arguments:
    vec3d vec1, vec2, output - result is output[i] = vec1[i] + vec2[i]
*/
void vec_add(vec3d vec1, vec3d vec2, vec3d output) {
    int ix;
    for (ix = 0; ix < 3; ix ++) output[ix] = vec1[ix] + vec2[ix];
}

/*  vec_scalar_mul() multiplies a vector by a scalar.
    
    Arguments:
    vec3d vec - the vector multiplicand
    double scalar - the scalar multiplier
    vec3d output - result buffer.
*/
void vec_scalar_mul(vec3d vec, double scalar, vec3d output) {
    int ix;
    for (ix = 0; ix < 3; ix ++) output[ix] = scalar * vec[ix];
}

/* Implements the common operation of finding the norm of a difference between
   two vectors. This happens a lot, so we have an optimized function.
   
   Arguments:
   vec3d vec1, vec2 - we need the difference between these vectors.
   
   Returns:
   double, the norm of the difference, i.e. ||vec1 - vec2||
*/
double vec_diff_norm(vec3d vec1, vec3d vec2) {
    return norm(vec1[0] - vec2[0], vec1[1] - vec2[1], vec1[2] - vec2[2]);
}

/* vec_norm() calculates the norm of a vector.
   
   Arguments:
   vec3d vec - the vector to norm.
*/
double vec_norm(vec3d vec) {
    /* Just plug into the macro */
    return norm(vec[0], vec[1], vec[2]);
}

/*  vec_dot() gives the dot product of two vectors. 
    
    Arguments:
    vec3d vec1, vec2 - the vectors whose dot product is sought.
    
    Returns:
    double, the dot of vec1 and vec2, i.e. <vec1, vec2>
*/
double vec_dot(vec3d vec1, vec3d vec2) {
    int ix;
    double sum = 0;
    
    for (ix = 0; ix < 3; ix ++) sum += vec1[ix]*vec2[ix];
    return sum;
}

/*  vec_cross() calculates the cross product of two vectors.
    
    Arguments:
    vec3d vec1, vec2 - the vectors whose cross product is sought.
    vec3d out - output buffer for the result vector.
*/
void vec_cross(vec3d vec1, vec3d vec2, vec3d out) {
    out[0] = vec1[1]*vec2[2] - vec1[2]*vec2[1];
    out[1] = vec1[2]*vec2[0] - vec1[0]*vec2[2];
    out[2] = vec1[0]*vec2[1] - vec1[1]*vec2[0];
}

/*  vec_cmp() checks whether two vectors are equal.
    
    Arguments:
    vec3d vec1, vec2 - the vectors to compare.
    
    Returns:
    int - true if equal, 0 otherwise.
*/
int vec_cmp(vec3d vec1, vec3d vec2) {
    int ix;
    
    for (ix = 0; ix < 3; ix ++) 
        if (vec1[ix] != vec2[ix])
            return 0;
    return 1;
}

/*  vec_approx_cmp() checks whether two vectors are approximately equal.
    
    Arguments:
    vec3d vec1, vec2 - the vectors to compare.
    double eps - tolerance, the maximal difference between each two components
        that still counts as equal (typ. 1e-10 or less).
    
    Returns:
    int - true if equal, 0 otherwise.
*/
int vec_approx_cmp(vec3d vec1, vec3d vec2, double eps) {
    int ix;
    
    for (ix = 0; ix < 3; ix ++) 
        if (fabs(vec1[ix] - vec2[ix]) > eps)
            return 0;
    return 1;
}

/*  unit_vector() divides a vector by its norm. In the special case of zero
    vector, returns the original vector.
    
    Arguments:
    vec3d vec - the original vector (3 x 1 doubles) 
    vec3d output - result, normalised, unit length vector.
*/
void unit_vector(vec3d vec, vec3d out){
	double normed; 
	
	normed = vec_norm(vec);
	if (normed == 0)
        normed = 1.0;
    
	vec_scalar_mul(vec, 1./normed, out);
}

