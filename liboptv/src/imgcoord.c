/****************************************************************************

Routine:	       	imgcoord.c

Author/Copyright:      	Hans-Gerd Maas

Address:	       	Institute of Geodesy and Photogrammetry
                        ETH - Hoenggerberg
		       	CH - 8093 Zurich

Creation Date:	       	22.4.88

Description:	       	computes x', y' from given Point and orientation
	       		(see: Kraus)

Routines contained:

****************************************************************************/

#include "imgcoord.h"


/*  img_coord() calculates projection from coordinates in
    world space to pixel coordinates in image space
    Arguments:
    vec3d pos is a vector of position in 3D (X,Y,Z real space)
    Calibration *cal parameters pointer of a specific camera
    multimedia *mm parameters pointer
    int i_cam - camera number (from 0 to cpar->num_cams)
    multimedia look-up table array mmLUT pointer
    Output:
    double x,y in pixel coordinates in the image space
 */
void img_coord (vec3d pos, Calibration *cal, mm_np *mm, double *x, double *y){
    
    double deno, r, dx, dy;
    Exterior Ex_t;
    double X,Y,Z, X_t,Y_t,Z_t,cross_p[3],cross_c[3];
    vec3d pos_t;
	
    /* calculate tilted positions and copy them to X_t, Y_t and Z_t */

	trans_Cam_Point(cal->ext_par, *mm, cal->glass_par, pos, \
          &Ex_t, pos_t, cross_p, cross_c);
    
    multimed_nlay (cal, mm, pos_t, &X_t,&Y_t);
    
    vec_set(pos_t,X_t,Y_t,pos_t[2]);
    
    back_trans_Point(pos_t, *mm, cal->glass_par, cross_p, cross_c, pos);
	
    pos[0] -= cal->ext_par.x0;  pos[1] -= cal->ext_par.y0;  pos[2] -= cal->ext_par.z0;

    deno = cal->ext_par.dm[0][2] * pos[0] + cal->ext_par.dm[1][2] * pos[1] + 
        cal->ext_par.dm[2][2] * pos[2];
        
    *x = cal->int_par.xh - cal->int_par.cc * (cal->ext_par.dm[0][0]*pos[0] + 
        cal->ext_par.dm[1][0]*pos[1] + cal->ext_par.dm[2][0]*pos[2]) / deno;
        
    *y = cal->int_par.yh - cal->int_par.cc * (cal->ext_par.dm[0][1]*pos[0] + 
        cal->ext_par.dm[1][1]*pos[1] + cal->ext_par.dm[2][1]*pos[2]) / deno;

    r = sqrt (*x * *x + *y * *y);
	
    dx = (*x) * (cal->added_par.k1*r*r + cal->added_par.k2*r*r*r*r + 
        cal->added_par.k3*r*r*r*r*r*r) + cal->added_par.p1 * (r*r + 2*(*x)*(*x)) + 
        2*cal->added_par.p2*(*x)*(*y);
        
    dy = (*y) * (cal->added_par.k1*r*r + cal->added_par.k2*r*r*r*r + 
        cal->added_par.k3*r*r*r*r*r*r) + cal->added_par.p2 * (r*r + 2*(*y)*(*y)) 
        + 2*cal->added_par.p1*(*x)*(*y);

    *x += dx;
    *y += dy;

    *x = cal->added_par.scx * (*x) - sin(cal->added_par.she) * (*y);
    *y = cos(cal->added_par.she) * (*y);

}


/* img_xy_geo() calculates projection from coordinates in
    world space to pixel coordinates in image space without 
    distortions
    Arguments:
    doubles X,Y,Z in real space
    Calibration *cal parameters pointer
    multimedia *mm parameters pointer
    int i_cam - camera number (from 0 to cpar->num_cams)
    multimedia look-up table array mmLUT pointer
    Output:
    double x,y in pixel coordinates in the image space
 */
 
void img_xy_mm_geo (vec3d pos, Calibration *cal, mm_np *mm, double *x, double *y){

  double deno;
  Exterior Ex_t;
  double X_t,Y_t,Z_t,cross_p[3],cross_c[3];
  vec3d pos_t;

  /* calculate tilted positions and copy them to X_t, Y_t and Z_t */
  
	trans_Cam_Point(cal->ext_par, *mm, cal->glass_par, pos, \
          &Ex_t, pos_t, cross_p, cross_c);
    
    multimed_nlay (cal, mm, pos_t, &X_t,&Y_t);
    
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

