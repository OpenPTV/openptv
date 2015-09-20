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


/* multimed_nlay() creates the Xq,Yq points for each X,Y point in the image space
    using radial shift from the multimedia model. 
    Arguments:
        mmLUT pointer to the multimedia look-up table  
        Calibration *cal pointer to calibration parameters
        vec3d pos (was X,Y,Z) - three doubles of the particle position in the 3D space
        Xq,Yq - two dimensional position (double pointer) on the glass surface, separating
        the water from air
        i_cam - integer number of the camera (0 - num_cams)
*/
void  multimed_nlay (mmlut *mmLUT, Exterior *ex, mm_np *mm, vec3d pos, double *Xq, 
    double *Yq, int i_cam){
  
    double  radial_shift; 
      
    radial_shift = multimed_r_nlay (mmLUT, ex, mm, pos, i_cam); 

     /* 
      * if radial_shift == 1.0, then
      * Xq = X; Yq = Y;
      */
     *Xq = ex->x0 + (pos[0] - ex->x0) * radial_shift;
     *Yq = ex->y0 + (pos[1] - ex->y0) * radial_shift;
  
}


/* multimedia_r_nlay() calculates and returns the radial shift
    Arguments:
    pointer to the multimedia look-up table 
    pointer to Exterior 
    pointer to mm_np 
    vec3d 3D position in terms of X,Y,Z
    number of the camera i_cam
    Outputs:
    double radial_shift: default is 1.0 if no solution is found
 */

double multimed_r_nlay (mmlut *mmLUT, Exterior *ex, mm_np *mm, vec3d pos, int i_cam){
 
    int   i, it = 0;
    double beta1, beta2[32], beta3, r, rbeta, rdiff, rq, mmf=1.0;
    double X,Y,Z;
    
    X = pos[0]; Y = pos[1]; Z = pos[2];
    
    /* ocf  over compensation factor for faster convergence
    is removed as it is never been used 
    double ocf = 1.0;  
    */

    int n_iter = 40;
 

    /* 1-medium case */
    if (mm->n1 == 1 && mm->nlay == 1 && mm->n2[0]== 1 && mm->n3 == 1) return (1.0);
  
  
    /* interpolation using the existing mmLUT */
    if (mm->lut) {
        mmf = get_mmf_from_mmLUT (mmLUT, i_cam, pos);
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



/* trans_Cam_Point() creates the shifted points X_t,Y_t,Z_t for each position X,Y,Z 
    Using Exterior and Interior parameters and the Glass vector of the variable
    window position and the two vectors that point to the crossing point
    Arguments: 
    
 */ 
void trans_Cam_Point(Exterior ex
                   , mm_np mm
                   , Glass gl
                   , vec3d pos
                   , Exterior *ex_t
                   , vec3d pos_t
                   , double cross_p[3]
                   , double cross_c[3]){

/* Beat Luethi June 07: I change the stuff to a system perpendicular to the interface */
    double dist_cam_glas,dist_point_glas,dist_o_glas; //glas inside at water 
    int row, col;
    double X,Y,Z;
    
    X = pos[0]; Y = pos[1]; Z = pos[2];
    

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


    for (row = 0; row < 3; row++)
        for (col = 0; col < 3; col++)
             ex_t->dm[row][col] = ex.dm[row][col];


    ex_t->omega = ex.omega;
    ex_t->phi   = ex.phi;
    ex_t->kappa = ex.kappa;

    ex_t->x0 = 0.;
    ex_t->y0 = 0.;
    ex_t->z0 = dist_cam_glas + mm.d[0];

    pos_t[0]=sqrt( pow(cross_p[0] - (cross_c[0] - mm.d[0] * gl.vec_x / dist_o_glas ) ,2.)
            +pow(cross_p[1] - (cross_c[1] - mm.d[0] * gl.vec_y / dist_o_glas ), 2.)
            +pow(cross_p[2] - (cross_c[2] - mm.d[0] * gl.vec_z / dist_o_glas ), 2.));
    pos_t[1] = 0;
    pos_t[2] = dist_point_glas;
          
}

/* the opposite direction transfer from X_t,Y_t,Z_t to the X,Y,Z in 3D space */

void back_trans_Point(vec3d pos_t, mm_np mm, Glass G, double cross_p[], 
    double cross_c[], vec3d pos){
    
    double nVe, nGl;
    double X_t,Y_t,Z_t;
    X_t = pos_t[0]; Y_t = pos_t[1]; Z_t = pos_t[2];

    
    nGl = sqrt( pow ( G.vec_x, 2. ) + pow( G.vec_y, 2.) + pow( G.vec_z ,2.) );
    
    nVe = sqrt( pow ( cross_p[0] - (cross_c[0] - mm.d[0] * G.vec_x / nGl ), 2.)
              + pow ( cross_p[1] - (cross_c[1] - mm.d[0] * G.vec_y / nGl ), 2.)
              + pow ( cross_p[2] - (cross_c[2] - mm.d[0] * G.vec_z / nGl ), 2.));

    pos[0] = cross_c[0] - mm.d[0] * G.vec_x / nGl + Z_t * G.vec_x / nGl;
    pos[1] = cross_c[1] - mm.d[0] * G.vec_y / nGl + Z_t * G.vec_y / nGl;
    pos[2] = cross_c[2] - mm.d[0] * G.vec_z / nGl + Z_t * G.vec_z / nGl;

    if (nVe > 0) {  
        /* We need this for when the cam-point line is exactly perp. to glass.
         The degenerate case will have nVe == 0 and produce NaNs on the
         following calculations.
        */
        pos[0] += X_t * (cross_p[0] - (cross_c[0] - mm.d[0] * G.vec_x / nGl)) / nVe;
        pos[1] += X_t * (cross_p[1] - (cross_c[1] - mm.d[0] * G.vec_y / nGl)) / nVe;
        pos[2] += X_t * (cross_p[2] - (cross_c[2] - mm.d[0] * G.vec_z / nGl)) / nVe;
    }
}



/* init_mmLUT() prepares the multimedia Look-Up Table
    Arguments: 
    Pointer to volume parameters *vpar
    pointer to the control parameters *cpar
    pointer to the calibraiton parameters *cal
    Output:
    pointer to the multi-media look-up table mmLUT structure
*/ 
void init_mmLUT (mmlut *mmLUT, volume_par *vpar, control_par *cpar, Calibration *cal){

  register int  i,j, nr, nz;
  int           i_cam;
  double        X,Y,Z, R, Zmin, Rmax=0, Zmax;
  vec3d         pos, a, xyz, xyz_t; 
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
          
        
          x = x - cal[i_cam].int_par.xh;
          y = y - cal[i_cam].int_par.yh;
          
          correct_brown_affin (x, y, cal[i_cam].added_par, &x,&y);
                    
      
          /* ray_tracing(x,y, Ex[i_cam], I[i_cam], G[i_cam], mmp, &X1, &Y1, &Z1, &a, &b, &c); */
          ray_tracing(x,y, &cal[i_cam], *(cpar->mm), pos, a);
          
                     
          
          /* Z = Zmin;   X = X1 + (Z-Z1) * a/c;   Y = Y1 + (Z-Z1) * b/c; */
          Z = Zmin;   
          X = pos[0] + (Z - pos[2]) * a[0]/a[2];   
          Y = pos[1] + (Z - pos[2]) * a[1]/a[2];
                    
          xyz[0] = X; xyz[1] = Y; xyz[2] = Z;
          
          /* trans_Cam_Point(Ex[i_cam],mmp,G[i_cam],X,Y,Z,&Ex_t[i_cam],&X_t,&Y_t,&Z_t,&cross_p,&cross_c); */
          trans_Cam_Point(cal[i_cam].ext_par, *(cpar->mm), cal[i_cam].glass_par, xyz, \
          &Ex_t[i_cam], xyz_t, (double *)cross_p, (double *)cross_c);

          X_t = xyz_t[0]; Y_t = xyz_t[1]; Z_t = xyz_t[2];      
                
          if( Z_t < Zmin_t ) Zmin_t = Z_t;
          if( Z_t > Zmax_t ) Zmax_t = Z_t;
                
      
          R = sqrt (( X_t - Ex_t[i_cam].x0 ) * ( X_t - Ex_t[i_cam].x0 )
                  + ( Y_t - Ex_t[i_cam].y0 ) * ( Y_t - Ex_t[i_cam].y0 )); 
                  
        
  
          if (R > Rmax) Rmax = R;
                          
          /* Z = Zmax;   X = X1 + (Z-Z1) * a/c;   Y = Y1 + (Z-Z1) * b/c; */
          Z = Zmax;   
          X = pos[0] + (Z - pos[2]) * a[0]/a[2];   
          Y = pos[1] + (Z - pos[2]) * a[1]/a[2];
          
          xyz[0] = X; xyz[1] = Y; xyz[2] = Z;
      
          /* trans_Cam_Point(Ex[i_cam],mmp,G[i_cam],X,Y,Z,&Ex_t[i_cam],&X_t,&Y_t,&Z_t,&cross_p,&cross_c); */
          trans_Cam_Point(cal[i_cam].ext_par, *(cpar->mm), cal[i_cam].glass_par, xyz,\
          &Ex_t[i_cam], xyz_t, (double *)cross_p, (double *)cross_c);
          
          X_t = xyz_t[0]; Y_t = xyz_t[1]; Z_t = xyz_t[2];
          
      
          if( Z_t < Zmin_t ) Zmin_t = Z_t;
          if( Z_t > Zmax_t ) Zmax_t = Z_t;
      
      
          R = sqrt (( X_t - Ex_t[i_cam].x0 ) * ( X_t - Ex_t[i_cam].x0 )
                  + ( Y_t - Ex_t[i_cam].y0 ) * ( Y_t - Ex_t[i_cam].y0 )); 
  
      
          if (R > Rmax) Rmax = R;
          
        }
      }
      

  
      /* round values (-> enlarge) */
      Rmax += (rw - fmod (Rmax, rw));
      
    
      /* get # of rasterlines in r,z */
      // BUG: fixed by Ad Holten 
      // nr = Rmax/rw + 1;
      // nz = (Zmax_t - Zmin_t)/rw + 1;
      
      
      /* get # of rasterlines in r,z */
	  nr = (int)(Rmax/rw + 1);
	  nz = (int)((Zmax_t-Zmin_t)/rw + 1);
      
      
      /* create two dimensional mmLUT structure */
      xyz[0] = X; xyz[1] = Y; xyz[2] = Z;
      
      trans_Cam_Point(cal[i_cam].ext_par, *(cpar->mm), cal[i_cam].glass_par, xyz,\
          &Ex_t[i_cam], xyz_t, (double *)cross_p, (double *)cross_c);
          
      X_t = xyz_t[0]; Y_t = xyz_t[1]; Z_t = xyz_t[2];
       
      mmLUT[i_cam].origin.x = Ex_t[i_cam].x0;
      mmLUT[i_cam].origin.y = Ex_t[i_cam].y0;
      mmLUT[i_cam].origin.z = Zmin_t;
      mmLUT[i_cam].nr = nr;
      mmLUT[i_cam].nz = nz;
      mmLUT[i_cam].rw = rw;
      // if (mmLUT[i_cam].data != NULL)			// preventing memory leaks, ad holten, 04-2013
	  //	free (mmLUT[i_cam].data);
      mmLUT[i_cam].data = (double *) malloc (nr*nz * sizeof (double));
      
      
      

   
      /* fill mmLUT structure */  
      Ri = (double *) malloc (nr * sizeof (double));
      for (i=0; i<nr; i++)  Ri[i] = i*rw;
      
      Zi = (double *) malloc (nz * sizeof (double));
      for (i=0; i<nz; i++)  Zi[i] = Zmin_t + i*rw;
      

  
      for (i=0; i<nr; i++) for (j=0; j<nz; j++) {
        
        /* old mmLUT[i_cam].data[i*nz + j]= multimed_r_nlay (Ex[i_cam], mmp, 
                                                          Ri[i]+Ex[i_cam].x0, Ex[i_cam].y0, Zi[j]);
        */

        /* there is no reason for multiple trans_Cam_Point inside the loop, Alex.
          trans_Cam_Point(cal[i_cam].ext_par, *(cpar->mm), cal[i_cam].glass_par, X, Y, Z,\
          &Ex_t[i_cam], &X_t, &Y_t, &Z_t, (double *)cross_p, (double *)cross_c);
        */ 
        
        xyz[0] = Ri[i] + Ex_t[i_cam].x0; xyz[1] = Ex_t[i_cam].y0; xyz[2] = Zi[j];
            
        mmLUT[i_cam].data[i*nz + j] = multimed_r_nlay (mmLUT, &Ex_t[i_cam], cpar->mm, 
            xyz, i_cam);                              
        } /* nr */
    free (Ri);	// preventing memory leaks, Ad Holten, 04-2013
	free (Zi);
    } /* for n_cams */
  
  /* when finished initalization, change the setting of the LUT flag */
  cpar->mm->lut = 1;

}


/* get_mmf_from_mmLUT() returns the value of mmf (double) for the space point X,Y,Z
    using the multimedia look up table
    Arguments:
    integer number of the cameras i_cam
    double position in 3D space, X,Y,Z
    multimedia look-up table mmLUT
*/
double get_mmf_from_mmLUT (mmlut *mmLUT, int i_cam, vec3d pos){

  int       i, ir,iz, nr,nz, rw, v4[4];
  double    R, sr, sz, mmf = 1.0;
  double    X,Y,Z;
  
  X = pos[0]; Y = pos[1]; Z = pos[2];
  
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
    
    X -= mmLUT[i_cam].origin.x;
    Y -= mmLUT[i_cam].origin.y;
    R = sqrt (X*X + Y*Y); 
    sr = R/rw; 
    ir = (int) sr; 
    sr -= ir;
        
    
    nz =  mmLUT[i_cam].nz;
    nr =  mmLUT[i_cam].nr;
    
    
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


/*  volumedimension() finds the limits of the measurement volume in 3D space
    Arguments:
    double pointers to the limits of the volume in x (xmin,xmax), y (ymin, ymax) and
    z (zmin, zmax) directions
    pointer to the volume parameters vpar
    pointer to the control parameters cpar
    pointer to the Calibration parameters cal
*/  
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
  vec3d pos, a, xyz, xyz_t;
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
        
        xyz[0] = X; xyz[1] = Y; xyz[2] = Z;

        trans_Cam_Point(cal[i_cam].ext_par, *(cpar->mm), cal[i_cam].glass_par, xyz,\
          &Ex_t[i_cam], xyz_t, (double *)cross_p, (double *)cross_c);
  
        X_t = xyz_t[0]; Y_t = xyz_t[1]; Z_t = xyz_t[2];
      
          
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
        
        
        xyz[0] = X; xyz[1] = Y; xyz[2] = Z;

        trans_Cam_Point(cal[i_cam].ext_par, *(cpar->mm), cal[i_cam].glass_par, xyz,\
            &Ex_t[i_cam], xyz_t, (double *)cross_p, (double *)cross_c);

        X_t = xyz_t[0]; Y_t = xyz_t[1]; Z_t = xyz_t[2];
        
          
          R = sqrt (( X_t - Ex_t[i_cam].x0 ) * ( X_t - Ex_t[i_cam].x0 )
                  + ( Y_t - Ex_t[i_cam].y0 ) * ( Y_t - Ex_t[i_cam].y0 )); 

      
      if (R > Rmax) Rmax = R;
      
      }
   }
}
