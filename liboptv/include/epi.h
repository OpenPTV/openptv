#ifndef EPI_H
#define EPI_H

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define MAXCAND 200 /* see typedefs.h for the reference */


#include "calibration.h"
#include "tracking_frame_buf.h"
#include "parameters.h"
#include "lsqadj.h"
#include "ray_tracing.h"
#include "multimed.h"
#include "vec_utils.h"
#include "imgcoord.h"


typedef struct {
  int  	pnr;
  double  tol, corr;
} candidate;

typedef struct
{
  int pnr;
  double x, y;
}
coord_2d;
	

void epi_mm (double xl, double yl, Calibration *cal1,
    Calibration *cal2, mm_np *mmp, volume_par *vpar,
    double *xmin, double *ymin, double *xmax, double *ymax);
    
void  epi_mm_2D (double xl, double yl, Calibration *cal1,
    mm_np *mmp, volume_par *vpar, vec3d out);
    
int find_candidate(coord_2d *crd, target *pix, int num,
    double xa, double ya, double xb, double yb,
    int n, int nx, int ny, int sumg, candidate cand[],
    volume_par *vpar, control_par *cpar, Calibration *cal);
    
#endif
