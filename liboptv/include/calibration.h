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

#ifdef __cplusplus
extern "C" {
#endif

typedef double Dmatrix[3][3];

/* Define Glass structure first */
typedef struct Glass_t {
    double vec_x, vec_y, vec_z;
    double n1, n2, n3;
    double d;
} Glass;

/* Then the rest of the structs */
typedef struct Exterior_t {
    double x0, y0, z0;
    double omega, phi, kappa;
    Dmatrix dm;
} Exterior;

typedef struct Interior_t {
    double xh, yh;
    double cc;
} Interior;


typedef struct {
    double k1, k2, k3;
    double p1, p2;
    double scx, she;
    int field;
} ap_52;

typedef struct {
    vec3d origin;
    int nr, nz, rw;
    double *data;
} mmlut;

typedef struct {
    Exterior ext_par;
    Interior int_par;
    Glass glass_par;
    ap_52 added_par;
    mmlut mmlut;
} Calibration;

/* Function declarations - now all structs are defined */
int write_ori(Exterior Ex, Interior interior, Glass glass, ap_52 ap, 
    char *filename, char *add_file);
int read_ori(Exterior Ex[], Interior interior[], Glass glass[], char *ori_file,
    ap_52 addp[], char *add_file, char *add_fallback);
void rotation_matrix(Exterior *ex);

/* Compare two calibration objects for equality */
int compare_calib(Calibration *c1, Calibration *c2);

/* Compare individual calibration components */
int compare_exterior(Exterior *e1, Exterior *e2);
int compare_interior(Interior *i1, Interior *i2);
int compare_glass(Glass *g1, Glass *g2);
int compare_addpar(ap_52 *a1, ap_52 *a2);

Calibration* read_calibration(char *ori_file, char *add_file, char *fallback_file);
int write_calibration(Calibration *cal, char *filename, char *add_file);

#ifdef __cplusplus
}
#endif

#endif
