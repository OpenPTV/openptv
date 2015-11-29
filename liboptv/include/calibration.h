/*  Objects related to calibration: exterior and interior parameters, glass
    parameters. A Calibration object to rule them all and find them. 

    References:
    [1] Th. Dracos (editor). Three-Dimensional Velocity and Vorticity Measuring
        and Image Analysis Techniques. Kluwer Academic Publishers. 1996. 
        pp.165-168
*/

#ifndef CALIBRATION_H
#define CALIBRATION_H

#include "vec_utils.h"

typedef	double	Dmatrix[3][3];	/* 3 x 3 rotation matrix */

typedef struct
{
  double  x0, y0, z0;
  double  omega, phi, kappa;
  Dmatrix dm;
}
Exterior;

typedef struct
{
  double xh, yh;
  double cc;
}
Interior;

typedef struct
{
  double vec_x,vec_y,vec_z;
}
Glass;

typedef struct
{
  double k1,k2,k3,p1,p2,scx,she;
}
ap_52;

/* mmLUT structure */
typedef struct {
    vec3d origin;
    int    nr, nz, rw;
    double *data; 
} mmlut;

typedef struct {
    Exterior ext_par;
    Interior int_par;
    Glass glass_par;
    ap_52 added_par;
    mmlut mmlut;
} Calibration;



int write_ori(Exterior Ex, Interior I, Glass G, ap_52 ap, char *filename, 
    char *add_file);
int read_ori (Exterior Ex[], Interior I[], Glass G[], char *ori_file, 
    ap_52 addp[], char *add_file, char *add_fallback);
int compare_exterior(Exterior *e1, Exterior *e2);
int compare_interior(Interior *i1, Interior *i2);
int compare_glass(Glass *g1, Glass *g2);
int compare_addpar(ap_52 *a1, ap_52 *a2);
int compare_calib(Calibration *c1, Calibration *c2);

Calibration *read_calibration(char *ori_file, char *add_file,
    char *fallback_file);
int write_calibration(Calibration *cal, char *ori_file, char *add_file);

void rotation_matrix(Exterior *Ex);

#endif

