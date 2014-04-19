#ifndef EPI_H
#define EPI_H

#include "calibration.h"
#include <optv/tracking_frame_buf.h>
#include "parameters.h"
#include "lsqadj.h"
#include "ray_tracing.h"


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
	

void  epi_mm(double xl, double yl, Calibration *cal1,
    Calibration *cal2, mm_np mmp, volume_par *vpar,
    double *xmin, double *ymin, double *xmax, double *ymax);
    
int  epi_mm_2D(double xl, double yl, Calibration *cal1,
    mm_np mmp, volume_par *vpar, double *xout, double *yout, double *zout);
    
void find_candidate_plus_msg(coord_2d crd[], target pix[], int num,
    double xa, double ya, double xb, double yb,
    int n, int nx, int ny, int sumg, candidate cand[], int *count, int i12,
    volume_par *vpar);
    
void find_candidate_plus(coord_2d crd[], target pix[], int num,
    double xa, double ya, double xb, double yb,
    int n, int nx, int ny, int sumg, candidate cand[], int *count, int nr,
    volume_par *vpar);

#endif
