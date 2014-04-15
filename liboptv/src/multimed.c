/****************************************************************************

Routine:            multimed.c

Author/Copyright:       Hans-Gerd Maas

Address:            Institute of Geodesy and Photogrammetry
                    ETH - Hoenggerberg
                    CH - 8093 Zurich

Creation Date:          20.4.88
    
Description:            
1 point, exterior orientation, multimedia parameters    ==> X,Y=shifts
          (special case 3 media, medium 2 = plane parallel)
                            
Routines contained:     -

****************************************************************************/

#include "multimed.h"


/* using radial shift from the multimedia model 
 * creates the Xq,Yq points for each X,Y point in the image space
 */
void  multimed_nlay (Exterior *ex
                   , mm_np *mm
                   , double X
                   , double Y
                   , double Z
                   , double *Xq
                   , double *Yq
                   , int i_cam
                   , mmlut *mmLUT){
  
  double    radial_shift; 

    radial_shift = multimed_r_nlay (ex, mm, X, Y, Z, i_cam, mmLUT); 

     /* 
      * if radial_shift == 1.0, then
      * Xq = X; Yq = Y;
      */
     *Xq = ex->x0 + (X - ex->x0) * radial_shift;
     *Yq = ex->y0 + (Y - ex->y0) * radial_shift;
  
}


/* calculates and returns the radial shift
 * Arguments:
 * pointer to Exterior 
 * pointer to mm_np 
 * 3D position in terms of X,Y,Z
 * pointer to the multimedia look-up table 
 * Outputs:
 * double radial_shift: default is 1.0 if no solution is found
 */
double multimed_r_nlay (Exterior *ex
                      , mm_np *mm
                      , double X
                      , double Y
                      , double Z
                      , int i_cam
                      , mmlut *mmLUT){
 
 
 
  int   i, it = 0;
  double beta1, beta2[32], beta3, r, rbeta, rdiff, rq, mmf;
  /* ocf  over compensation factor for faster convergence
   * is removed as it is never been used 
   double ocf = 1.0;  
   */

  int n_iter = 40;
 

  /* 1-medium case */
  if (mm->n1 == 1 && mm->nlay == 1 && mm->n2[0]== 1 && mm->n3 == 1) return (1.0);
  
  
  /* interpolation in mmLUT, if selected (requires some global variables) */
  if (mm->lut) {
    mmf = get_mmf_from_mmLUT (i_cam, X,Y,Z, (mmlut *)mmLUT);
    if (mmf > 0) return (mmf);
  }
  
  /* iterative procedure */
  r = sqrt ((X - ex->x0) * (X - ex->x0) + (Y - ex->y0) * (Y - ex->y0));
  rq = r;
  
  do
    {
      beta1 = atan (rq/(ex->z0 - Z));
      for (i=0; i < mm->nlay; i++) beta2[i] = asin (sin(beta1) * mm->n1/mm->n2[i]);
      beta3 = asin (sin(beta1) * mm->n1/mm->n3);
      
      rbeta = (ex->z0 - mm->d[0]) * tan(beta1) - Z * tan(beta3);
      for (i = 0; i < mm->nlay; i++) rbeta += (mm->d[i] * tan(beta2[i]));
      rdiff = r - rbeta;
      /* rdiff *= ocf; */ 
      rq += rdiff;
      it++;
    }
  while (((rdiff > 0.001) || (rdiff < -0.001))  &&  it < n_iter);
  

  
  if (it >= n_iter) {
      printf ("multimed_r_nlay stopped after %d iterations\n", n_iter);  
      return (1.0);
    }
  
    if (r != 0){   
      return (rq/r);
    }  
    else {
    	return (1.0);
    }
}



/* Using Exterior and Interior parameters and the Glass vector of the variable
 * window position creates the shifted points X_t,Y_t,Z_t for each position X,Y,Z
 * and the two vectors that point to the crossing point
 */ 
void trans_Cam_Point(Exterior ex
                   , mm_np mm
                   , Glass gl
                   , double X
                   , double Y
                   , double Z
                   , Exterior *ex_t
                   , double *X_t
                   , double *Y_t
                   , double *Z_t
                   , double cross_p[3]
                   , double cross_c[3]){

  /* --Beat Luethi June 07: I change the stuff to a system perpendicular to the interface */
  double dist_cam_glas,dist_point_glas,dist_o_glas; //glas inside at water 
  
  dist_o_glas = sqrt( gl.vec_x * gl.vec_x + gl.vec_y * gl.vec_y + gl.vec_z * gl.vec_z);
  
  dist_cam_glas = ex.x0 * gl.vec_x / dist_o_glas + ex.y0 * gl.vec_y / dist_o_glas + \
  ex.z0 * gl.vec_z / dist_o_glas - dist_o_glas - mm.d[0];
  
  dist_point_glas = X * gl.vec_x / dist_o_glas + \
                    Y * gl.vec_y / dist_o_glas + \
                    Z * gl.vec_z / dist_o_glas - dist_o_glas; 

  cross_c[0] = ex.x0 - dist_cam_glas * gl.vec_x / dist_o_glas;
  cross_c[1] = ex.y0 - dist_cam_glas * gl.vec_y / dist_o_glas;
  cross_c[2] = ex.z0 - dist_cam_glas * gl.vec_z / dist_o_glas;
  
  cross_p[0] = X - dist_point_glas * gl.vec_x / dist_o_glas;
  cross_p[1] = Y - dist_point_glas * gl.vec_y / dist_o_glas;
  cross_p[2] = Z - dist_point_glas * gl.vec_z / dist_o_glas;

  ex_t->x0 = 0.;
  ex_t->y0 = 0.;
  ex_t->z0 = dist_cam_glas + mm.d[0];

  *X_t=sqrt( pow(cross_p[0] - (cross_c[0] - mm.d[0] * gl.vec_x / dist_o_glas ) ,2.)
            +pow(cross_p[1] - (cross_c[1] - mm.d[0] * gl.vec_y / dist_o_glas ), 2.)
            +pow(cross_p[2] - (cross_c[2] - mm.d[0] * gl.vec_z / dist_o_glas ), 2.));
  *Y_t = 0;
  *Z_t = dist_point_glas;
      
}

/* the opposite direction transfer from X_t,Y_t,Z_t to the X,Y,Z in 3D space */

void back_trans_Point(double X_t, double Y_t, double Z_t, mm_np mm, Glass G, \
double cross_p[], double cross_c[], double *X, double *Y, double *Z){
    
    double nVe, nGl;
    
    nGl = sqrt( pow ( G.vec_x, 2. ) + pow( G.vec_y, 2.) + pow( G.vec_z ,2.) );
    
    nVe = sqrt( pow ( cross_p[0] - (cross_c[0] - mm.d[0] * G.vec_x / nGl ), 2.)
              + pow ( cross_p[1] - (cross_c[1] - mm.d[0] * G.vec_y / nGl ), 2.)
              + pow ( cross_p[2] - (cross_c[2] - mm.d[0] * G.vec_z / nGl ), 2.));

    *X = cross_c[0] - mm.d[0] * G.vec_x / nGl + Z_t * G.vec_x / nGl;
    *Y = cross_c[1] - mm.d[0] * G.vec_y / nGl + Z_t * G.vec_y / nGl;
    *Z = cross_c[2] - mm.d[0] * G.vec_z / nGl + Z_t * G.vec_z / nGl;

    if (nVe > 0) {  
        /* We need this for when the cam-point line is exactly perp. to glass.
         The degenerate case will have nVe == 0 and produce NaNs on the
         following calculations.
        */
        *X += X_t * (cross_p[0] - (cross_c[0] - mm.d[0] * G.vec_x / nGl)) / nVe;
        *Y += X_t * (cross_p[1] - (cross_c[1] - mm.d[0] * G.vec_y / nGl)) / nVe;
        *Z += X_t * (cross_p[2] - (cross_c[2] - mm.d[0] * G.vec_z / nGl)) / nVe;
    }
}



/* init_mmLUT prepares the multimedia Look-Up Table
Arguments: 
	Pointer to volume parameters *vpar
	pointer to the control parameters *cpar
	pointer to the calibraiton parameters *cal
Output:
    pointer to the multi-media look-up table mmLUT structure

*/ 
void init_mmLUT (volume_par *vpar
               , control_par *cpar
               , Calibration *cal
               , mmlut *mmLUT){

  register int  i,j, nr, nz;
  int           i_cam;
  double        X,Y,Z, R, Zmin, Rmax=0, Zmax;
  double        pos[3], a[3]; 
  double        x,y, *Ri,*Zi;
  double        rw = 2.0; 
  Exterior      Ex_t[4];
  double        X_t,Y_t,Z_t, Zmin_t,Zmax_t;
  double        cross_p[3],cross_c[3]; 
  FILE          *fpp;
  double        xc[2], yc[2];  /* image corners */
  
  
  
  /* image corners */
  xc[0] = 0.0;
  xc[1] = (double) cpar->imx;
  yc[0] = 0.0;
  yc[1] = (double) cpar->imy;

 


  for (i_cam = 0; i_cam < cpar->num_cams; i_cam++){
  
      /* find extrema of imaged object volume */
      
      Zmin = vpar->Zmin_lay[0];
      Zmax = vpar->Zmax_lay[0];
   
      Zmin -= fmod (Zmin, rw);
  	  Zmax += (rw - fmod (Zmax, rw));
      Zmin_t=Zmin;
      Zmax_t=Zmax;
      
  
      /* intersect with image vertices rays */
      
      for (i = 0; i < 2; i ++) {
         for (j = 0; j < 2; j++) {
          
          pixel_to_metric (&x, &y, xc[i], yc[j], cpar);
          
        
          /* x = x - I[i_cam].xh;
             y = y - I[i_cam].yh;
          */
          x = x - cal[i_cam].int_par.xh;
          y = y - cal[i_cam].int_par.yh;
          
  
          correct_brown_affin (x, y, cal[i_cam].added_par, &x,&y);
          
      
          /* ray_tracing(x,y, Ex[i_cam], I[i_cam], G[i_cam], mmp, &X1, &Y1, &Z1, &a, &b, &c); */
          ray_tracing(x,y, &cal[i_cam], *(cpar->mm), pos, a);
          
                     
          
          /* Z = Zmin;   X = X1 + (Z-Z1) * a/c;   Y = Y1 + (Z-Z1) * b/c; */
          Z = Zmin;   
          X = pos[0] + (Z - pos[2]) * a[0]/a[2];   
          Y = pos[1] + (Z - pos[2]) * a[1]/a[2];
          
          
          /* trans */
          /* trans_Cam_Point(Ex[i_cam],mmp,G[i_cam],X,Y,Z,&Ex_t[i_cam],&X_t,&Y_t,&Z_t,&cross_p,&cross_c); */
          trans_Cam_Point(cal[i_cam].ext_par, *(cpar->mm), cal[i_cam].glass_par, X, Y, Z, \
          &Ex_t[i_cam], &X_t, &Y_t, &Z_t, (double *)cross_p, (double *)cross_c);

                
          if( Z_t < Zmin_t ) Zmin_t = Z_t;
          if( Z_t > Zmax_t ) Zmax_t = Z_t;
                
      
          R = sqrt (( X_t - Ex_t[i_cam].x0 ) * ( X_t - Ex_t[i_cam].x0 )
                  + ( Y_t - Ex_t[i_cam].y0 ) * ( Y_t - Ex_t[i_cam].y0 )); 
                  
        
  
          if (R > Rmax) Rmax = R;
          
          /* printf ("radial shift is %f and max shift is %f \n", R, Rmax); */
                
          /* Z = Zmax;   X = X1 + (Z-Z1) * a/c;   Y = Y1 + (Z-Z1) * b/c; */
          Z = Zmax;   
          X = pos[0] + (Z - pos[2]) * a[0]/a[2];   
          Y = pos[1] + (Z - pos[2]) * a[1]/a[2];
      
          /* trans */
          /* trans_Cam_Point(Ex[i_cam],mmp,G[i_cam],X,Y,Z,&Ex_t[i_cam],&X_t,&Y_t,&Z_t,&cross_p,&cross_c); */
          trans_Cam_Point(cal[i_cam].ext_par, *(cpar->mm), cal[i_cam].glass_par, X, Y, Z,\
          &Ex_t[i_cam], &X_t, &Y_t, &Z_t, (double *)cross_p, (double *)cross_c);
          
         /* printf("adjusted 3d point for Zmax %f %f %f \n", X_t,Y_t,Z_t); */
      
          if( Z_t < Zmin_t ) Zmin_t = Z_t;
          if( Z_t > Zmax_t ) Zmax_t = Z_t;
      
      
          R = sqrt (( X_t - Ex_t[i_cam].x0 ) * ( X_t - Ex_t[i_cam].x0 )
                  + ( Y_t - Ex_t[i_cam].y0 ) * ( Y_t - Ex_t[i_cam].y0 )); 
  
      
          if (R > Rmax) Rmax = R;
          
          /* printf ("radial shift is %f and max shift is %f \n", R, Rmax); */
        }
      }
      

  
      /* round values (-> enlarge) */
      Rmax += (rw - fmod (Rmax, rw));
    
      /* get # of rasterlines in r,z */
      nr = Rmax/rw + 1;
      nz = (Zmax_t - Zmin_t)/rw + 1;
      
 
      /* create two dimensional mmLUT structure */
      
      mmLUT[i_cam].origin.x = Ex_t[i_cam].x0;
      mmLUT[i_cam].origin.y = Ex_t[i_cam].y0;
      mmLUT[i_cam].origin.z = Zmin_t;
      mmLUT[i_cam].nr = nr;
      mmLUT[i_cam].nz = nz;
      mmLUT[i_cam].rw = rw;
      mmLUT[i_cam].data = (double *) malloc (nr*nz * sizeof (double));
      

   
      /* fill mmLUT structure */
      /* ==================== */
  
      Ri = (double *) malloc (nr * sizeof (double));
      for (i=0; i<nr; i++)  Ri[i] = i*rw;
      Zi = (double *) malloc (nz * sizeof (double));
      for (i=0; i<nz; i++)  Zi[i] = Zmin_t + i*rw;
      

  
      for (i=0; i<nr; i++) {
        for (j=0; j<nz; j++) {
        /* old mmLUT[i_cam].data[i*nz + j]= multimed_r_nlay (Ex[i_cam], mmp, 
                                                          Ri[i]+Ex[i_cam].x0, Ex[i_cam].y0, Zi[j]);
        */

          
        mmLUT[i_cam].data[i*nz + j] = multimed_r_nlay (&Ex_t[i_cam], cpar->mm, \
                              Ri[i] + Ex_t[i_cam].x0, Ex_t[i_cam].y0, Zi[j], i_cam, mmLUT);
                
                              
          } /* nz */
        } /* nr */
    } /* for n_cams */
}



double get_mmf_from_mmLUT (int i_cam
                         , double X
                         , double Y
                         , double Z
                         , mmlut *mmLUT){

  int       i, ir,iz, nr,nz, rw, v4[4];
  double    R, sr, sz, mmf = 1.0;
  
  rw =  mmLUT[i_cam].rw;
  
  if (X == 1.0 && Y == 1.0 && Z == 1.0){
  /* 
    printf("entered mmlut with zeros and %d \n", i_cam);
    printf("origin.z = %f \n", mmLUT[i_cam].origin.z);
    printf("and rw is %d \n", mmLUT[i_cam].rw);
  */
    Z -= mmLUT[i_cam].origin.z; 
    sz = Z/rw; 
    iz = (int) sz; 
    sz -= iz;
    
    /* printf("sz, iz %f, %d \n", sz, iz); */
    X -= mmLUT[i_cam].origin.x;
    Y -= mmLUT[i_cam].origin.y;
    /* printf("X, Y, %f, %f \n", X, Y); */
    R = sqrt (X*X + Y*Y); 
    sr = R/rw; 
    ir = (int) sr; 
    sr -= ir;
    /*
  	printf(" sr,ir %f, %d \n", sr, ir);
  	printf("rw ,mmLUT[i_cam].rw: %d, %d \n", rw, mmLUT[i_cam].rw);
  	printf("New position: %f, %f, %f \n", X,Y,Z);
  	*/
  	
  	nz =  mmLUT[i_cam].nz;
    nr =  mmLUT[i_cam].nr;
    
    /* printf("nz, nr, %d, %d \n", nz, nr); */
    
  /* check whether point is inside camera's object volume */
  if (ir > nr)              return (0.0);
  if (iz < 0  ||  iz > nz)  return (0.0);
  
  /* bilinear interpolation in r/z box */
  /* ================================= */
  
  /* get vertices of box */
  v4[0] = ir*nz + iz;
  v4[1] = ir*nz + (iz+1);
  v4[2] = (ir+1)*nz + iz;
  v4[3] = (ir+1)*nz + (iz+1);
  
  /* printf(" vertices are: %d, %d, %d, %d, %d \n", v4[0],v4[1],v4[2],v4[3], nr*nz); */
  
  /* 2. check wther point is inside camera's object volume */
  /* important for epipolar line computation */
  for (i=0; i<4; i++)
    if (v4[i] < 0  ||  v4[i] > nr*nz)   return (0);
  

   /* 
    printf(" %f, %f, %f, %f \n", mmLUT[i_cam].data[v4[0]], mmLUT[i_cam].data[v4[1]], \
    mmLUT[i_cam].data[v4[2]], mmLUT[i_cam].data[v4[3]]); 
    */
  /* interpolate */
  mmf = mmLUT[i_cam].data[v4[0]] * (1-sr)*(1-sz)
    + mmLUT[i_cam].data[v4[1]] * (1-sr)*sz
    + mmLUT[i_cam].data[v4[2]] * sr*(1-sz)
    + mmLUT[i_cam].data[v4[3]] * sr*sz;
  
  /* printf(" mmf after all estimates is %f \n", mmf); */
  return (mmf);  	
  }
  else {
  Z -= mmLUT[i_cam].origin.z; sz = Z/rw; iz = (int) sz; sz -= iz;
  X -= mmLUT[i_cam].origin.x;
  Y -= mmLUT[i_cam].origin.y;
  R = sqrt (X*X + Y*Y); sr = R/rw; ir = (int) sr; sr -= ir;
  
  
  nz =  mmLUT[i_cam].nz;
  nr =  mmLUT[i_cam].nr;
    
  /* check whether point is inside camera's object volume */
  if (ir > nr)              return (0);
  if (iz < 0  ||  iz > nz)  return (0);
  
  /* bilinear interpolation in r/z box */
  /* ================================= */
  
  /* get vertices of box */
  v4[0] = ir*nz + iz;
  v4[1] = ir*nz + (iz+1);
  v4[2] = (ir+1)*nz + iz;
  v4[3] = (ir+1)*nz + (iz+1);
  
  /* 2. check wther point is inside camera's object volume */
  /* important for epipolar line computation */
  for (i=0; i<4; i++)
    if (v4[i] < 0  ||  v4[i] > nr*nz)   return (0);
  
  /* interpolate */
  mmf = mmLUT[i_cam].data[v4[0]] * (1-sr)*(1-sz)
    + mmLUT[i_cam].data[v4[1]] * (1-sr)*sz
    + mmLUT[i_cam].data[v4[2]] * sr*(1-sz)
    + mmLUT[i_cam].data[v4[3]] * sr*sz;
  
  return (mmf);
  
  	
  }  
}



void volumedimension (double *xmax
                    , double *xmin
                    , double *ymax
                    , double *ymin
                    , double *zmax
                    , double *zmin
                    , volume_par *vpar
                    , control_par *cpar
                    , Calibration *cal){

  int   i_cam, i, j;
  double X, Y, Z, R, Rmax=0, Zmin, Zmax;
  double pos[3], a[3];
  double x,y;  
  double xc[2], yc[2];  /* image corners */
  Exterior Ex_t[4];
  double X_t, Y_t, Z_t, Zmin_t,Zmax_t;
  double        cross_p[3],cross_c[3];
  
  xc[0] = 0.0;
  xc[1] = (double) cpar->imx;
  yc[0] = 0.0;
  yc[1] = (double) cpar->imy;
  
  
  
       Zmin = vpar->Zmin_lay[0];
       Zmax = vpar->Zmax_lay[0];

       *zmin = Zmin;
       *zmax = Zmax;
 

    
  /* find extrema of imaged object volume */
  /* ==================================== */
  

  for (i_cam = 0; i_cam < cpar->num_cams; i_cam++){
      for (i = 0; i < 2; i ++) for (j = 0; j < 2; j++) {
        
//        *xmax = vpar->X_lay[1];
//        *xmin = vpar->X_lay[0];
 
  
      /* intersect with image vertices rays */
/*
          pixel_to_metric (0.0, 0.0, imx,imy, pix_x,pix_y, &x,&y, chfield);
          x = x - I[i_cam].xh;
          y = y - I[i_cam].yh;
          correct_brown_affin (x, y, ap[i_cam], &x,&y);
          ray_tracing(x,y, Ex[i_cam], I[i_cam], G[i_cam], mmp, &X1, &Y1, &Z1, &a, &b, &c);
          Z = Zmin;   X = X1 + (Z-Z1) * a/c;   Y = Y1 + (Z-Z1) * b/c;
          R = sqrt (  (X-Ex[i_cam].x0)*(X-Ex[i_cam].x0)
              + (Y-Ex[i_cam].y0)*(Y-Ex[i_cam].y0)); 

*/ 
          pixel_to_metric (&x, &y, xc[i], yc[j], cpar);
          x = x - cal[i_cam].int_par.xh;
          y = y - cal[i_cam].int_par.yh;
          correct_brown_affin (x, y, cal[i_cam].added_par, &x, &y);
          
          
          ray_tracing(x, y, &cal[i_cam], *(cpar->mm), pos, a);
          
          
          Z = Zmin;   
          X = pos[0] + (Zmin - pos[2]) * a[0]/a[2];   
          Y = pos[1] + (Zmin - pos[2]) * a[1]/a[2];
          
                     
          if ( X > *xmax) *xmax = X;
          if ( X < *xmin) *xmin = X;
          if ( Y > *ymax) *ymax = Y;
          if ( Y < *ymin) *ymin = Y;
          
          
        /* old version, not clear why Beat didn't introduce trans_Cam into volumedimension 
        R = sqrt (( X - cal[i_cam].ext_par.x0 ) * ( X - cal[i_cam].ext_par.x0 )
                  + ( Y - cal[i_cam].ext_par.y0 ) * ( Y - cal[i_cam].ext_par.y0 )); 
        */ 
        trans_Cam_Point(cal[i_cam].ext_par, *(cpar->mm), cal[i_cam].glass_par, X, Y, Z,\
          &Ex_t[i_cam], &X_t, &Y_t, &Z_t, (double *)cross_p, (double *)cross_c);
          
          R = sqrt (( X_t - Ex_t[i_cam].x0 ) * ( X_t - Ex_t[i_cam].x0 )
                  + ( Y_t - Ex_t[i_cam].y0 ) * ( Y_t - Ex_t[i_cam].y0 )); 

          if (R > Rmax) Rmax = R;
              
/* 
      
      Z = Zmax;   X = X1 + (Z-Z1) * a/c;   Y = Y1 + (Z-Z1) * b/c;
      R = sqrt (  (X-Ex[i_cam].x0)*(X-Ex[i_cam].x0)
          + (Y-Ex[i_cam].y0)*(Y-Ex[i_cam].y0));
*/
        
        Z = Zmax;
        X = pos[0] + (Z - pos[2]) * a[0]/a[2];   
        Y = pos[1] + (Z - pos[2]) * a[1]/a[2];
        
        
       if ( X > *xmax) *xmax = X;
       if ( X < *xmin) *xmin = X;
       if ( Y > *ymax) *ymax = Y;
       if ( Y < *ymin) *ymin = Y;
        
        
        /* old version, not clear why Beat didn't introduce trans_Cam into volumedimension 
        R = sqrt (( X - cal[i_cam].ext_par.x0 ) * ( X - cal[i_cam].ext_par.x0 )
                  + ( Y - cal[i_cam].ext_par.y0 ) * ( Y - cal[i_cam].ext_par.y0 )); 
        */ 
        trans_Cam_Point(cal[i_cam].ext_par, *(cpar->mm), cal[i_cam].glass_par, X, Y, Z,\
          &Ex_t[i_cam], &X_t, &Y_t, &Z_t, (double *)cross_p, (double *)cross_c);
          
          R = sqrt (( X_t - Ex_t[i_cam].x0 ) * ( X_t - Ex_t[i_cam].x0 )
                  + ( Y_t - Ex_t[i_cam].y0 ) * ( Y_t - Ex_t[i_cam].y0 )); 

      
      if (R > Rmax) Rmax = R;
      
      }
   }
}
