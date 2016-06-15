/****************************************************************************
* Calculation of image coordinates from 3D positions, with or without 
* A distortion model.
* 
* References:
* [1] https://en.wikipedia.org/wiki/Distortion_(optics)
****************************************************************************/

#include "imgcoord.h"
#include "multimed.h"
#include "trafo.h"
#include <math.h>

/* flat_image_coord() calculates projection from coordinates in
    world space to metric coordinates in image space without 
    distortions
    
    Arguments:
    vec3d orig_pos - a vector of position in 3D (X,Y,Z real space)
    Calibration *cal - parameters of the camera on which to project.
    mm_np *mm - layer thickness and refractive index parameters.
    
    Output:
    double x,y - metric coordinates of projection in the image space.
 */
void flat_image_coord (vec3d orig_pos, Calibration *cal, mm_np *mm, 
    double *x, double *y)
{
    double deno;
    Calibration cal_t;
    double X_t, Y_t, cross_p[3], cross_c[3];
    vec3d pos_t, pos;
  
    cal_t.mmlut = cal->mmlut;

    /* This block calculate 3D position in an imaginary air-filled space, 
       i.e. where the point will have been seen in the absence of refractive
       layers between it and the camera.
    */
    trans_Cam_Point(cal->ext_par, *mm, cal->glass_par, orig_pos, \
         &(cal_t.ext_par), pos_t, cross_p, cross_c);
    multimed_nlay (&cal_t, mm, pos_t, &X_t,&Y_t);
    vec_set(pos_t,X_t,Y_t,pos_t[2]);
    back_trans_Point(pos_t, *mm, cal->glass_par, cross_p, cross_c, pos);

    deno = cal->ext_par.dm[0][2] * (pos[0]-cal->ext_par.x0)
    + cal->ext_par.dm[1][2] * (pos[1]-cal->ext_par.y0)
    + cal->ext_par.dm[2][2] * (pos[2]-cal->ext_par.z0);

    *x = - cal->int_par.cc *  (cal->ext_par.dm[0][0] * (pos[0]-cal->ext_par.x0)
          + cal->ext_par.dm[1][0] * (pos[1]-cal->ext_par.y0)
          + cal->ext_par.dm[2][0] * (pos[2]-cal->ext_par.z0)) / deno;

    *y = - cal->int_par.cc *  (cal->ext_par.dm[0][1] * (pos[0]-cal->ext_par.x0)
          + cal->ext_par.dm[1][1] * (pos[1]-cal->ext_par.y0)
          + cal->ext_par.dm[2][1] * (pos[2]-cal->ext_par.z0)) / deno;
}

/*  img_coord() uses flat_image_coord() to estimate metric coordinates in image space
    from the 3D position in the world and distorts it using the Brown 
    distortion model [1]
    
    Arguments:
    vec3d pos - a vector of position in 3D (X,Y,Z real space)
    Calibration *cal - parameters of the camera on which to project.
    mm_np *mm - layer thickness and refractive index parameters.
    
    Output:
    double x,y - metric distorted coordinates of projection in the image space.
*/
void img_coord (vec3d pos, Calibration *cal, mm_np *mm, double *x, double *y) {
    flat_image_coord (pos, cal, mm, x, y);
    flat_to_dist(*x, *y, cal, x, y);
}

