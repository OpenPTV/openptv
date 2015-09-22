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

/*  multimed_nlay() creates the Xq,Yq points for each X,Y point in the image space
    using radial shift from the multimedia model.
    
    Arguments:
    Calibration *cal - calibration parameters.
    mm_np *mm - multimedia info struct (e.g. number of layers, thickness of 
        layers)
    vec3d pos - three doubles of the particle position in the 3D space.
    double *Xq, *Yq - returned values; two dimensional position on the glass 
        surface separating the water from air.
*/
void  multimed_nlay (Calibration *cal, mm_np *mm, vec3d pos, 
    double *Xq, double *Yq)
{
    double  radial_shift;
      
    radial_shift = multimed_r_nlay (cal, mm, pos); 

    /* if radial_shift == 1.0, this degenerates to Xq = X, Yq = Y  */
    *Xq = cal->ext_par.x0 + (pos[0] - cal->ext_par.x0) * radial_shift;
    *Yq = cal->ext_par.y0 + (pos[1] - cal->ext_par.y0) * radial_shift;
}


/*  multimedia_r_nlay() calculates and returns the radial shift
    
    Arguments:
    Calibration *cal - calibration parameters.
    mm_np *mm - multimedia info struct (e.g. number of layers, thickness of 
        layers)
    vec3d pos - three doubles of the particle position in the 3D space.
    
    Returns:
    double radial_shift: default is 1.0 if no solution is found
*/
double multimed_r_nlay (Calibration *cal, mm_np *mm, vec3d pos) {
    int i, it = 0;
    int n_iter = 40;
    double beta1, beta2[32], beta3, r, rbeta, rdiff, rq, mmf=1.0;
    double X,Y,Z;
    
    /* 1-medium case */
    if (mm->n1 == 1 && mm->nlay == 1 && mm->n2[0]== 1 && mm->n3 == 1)
        return (1.0);
    
    X = pos[0]; Y = pos[1]; Z = pos[2];
    
    /* interpolation using the existing mmlut */
    if (mm->lut) {
        mmf = get_mmf_from_mmlut(cal, pos);
        if (mmf > 0) return (mmf);
    }
    
    /* iterative procedure */
    r = norm((X - cal->ext_par.x0), (Y - cal->ext_par.y0), 0);
    rq = r;
  
    do
    {
        beta1 = atan (rq/(cal->ext_par.z0 - Z));
        for (i = 0; i < mm->nlay; i++)
            beta2[i] = asin (sin(beta1) * mm->n1/mm->n2[i]);
        beta3 = asin (sin(beta1) * mm->n1/mm->n3);

        rbeta = (cal->ext_par.z0 - mm->d[0]) * tan(beta1) - Z * tan(beta3);
        for (i = 0; i < mm->nlay; i++)
            rbeta += (mm->d[i] * tan(beta2[i]));
        
        rdiff = r - rbeta;
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


/*  trans_Cam_Point() creates the shifted points for each position X,Y,Z 
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

/*  move_along_ray() calculates the position of a point in a global Z value
    along a ray whose vertex and direction are given.
    
    Arguments:
    double glob_Z - the Z value of the result point in the global
        coordinate system.
    vec3d vertex - the ray vertex.
    vec3d direct - the ray direction, a unit vector.
    vec3d out - result buffer.
*/
void move_along_ray(double glob_Z, vec3d vertex, vec3d direct, vec3d out) {
    out[0] = vertex[0] + (glob_Z - vertex[2]) * direct[0]/direct[2];   
    out[1] = vertex[1] + (glob_Z - vertex[2]) * direct[1]/direct[2];
    out[2] = glob_Z;
}

/*  init_mmlut() prepares the multimedia Look-Up Table for a single camera
    Arguments: 
    
    volume_par *vpar - struct holding the observed volume size.
    control_par *cpar - struct holding general control parameters such as 
        image size.
    Calibration *cal - current single-camera positioning and other camera-
        specific data.
    
    Output:
    initializes data in cal->mmlut->data
*/ 
void init_mmlut (volume_par *vpar, control_par *cpar, Calibration *cal) {
  register int  i,j, nr, nz;
  int  i_cam;
  double X,Y,Z, R, Zmin, Rmax=0, Zmax;
  vec3d pos, a, xyz, xyz_t; 
  double x,y, *Ri,*Zi;
  double rw = 2.0; 
  Exterior Ex_t; /* A frame representing a point outside tank, middle of glass*/
  double X_t,Y_t,Z_t, Zmin_t,Zmax_t;
  double cross_p[3],cross_c[3]; 
  double xc[2], yc[2];  /* image corners */
  
  /* image corners */
  xc[0] = 0.0;
  xc[1] = (double) cpar->imx;
  yc[0] = 0.0;
  yc[1] = (double) cpar->imy;

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

          x = x - cal->int_par.xh;
          y = y - cal->int_par.yh;
  
          correct_brown_affin (x, y, cal->added_par, &x,&y);  
          ray_tracing(x,y, cal, *(cpar->mm), pos, a);
          
          move_along_ray(Zmin, pos, a, xyz);
          trans_Cam_Point(cal->ext_par, *(cpar->mm), cal->glass_par, xyz, \
            &Ex_t, xyz_t, (double *)cross_p, (double *)cross_c);

          if( xyz_t[2] < Zmin_t ) Zmin_t = xyz_t[2];
          if( xyz_t[2] > Zmax_t ) Zmax_t = xyz_t[2];

          R = norm((xyz_t[0] - Ex_t.x0), (xyz_t[1] - Ex_t.y0), 0);
          if (R > Rmax)
              Rmax = R;
           
          move_along_ray(Zmax, pos, a, xyz);
          trans_Cam_Point(cal->ext_par, *(cpar->mm), cal->glass_par, xyz,\
              &Ex_t, xyz_t, (double *)cross_p, (double *)cross_c);
  
          if( xyz_t[2] < Zmin_t ) Zmin_t = xyz_t[2];
          if( xyz_t[2] > Zmax_t ) Zmax_t = xyz_t[2];

          R = norm((xyz_t[0] - Ex_t.x0), (xyz_t[1] - Ex_t.y0), 0);
          if (R > Rmax)
              Rmax = R;
      }
  }

  /* round values (-> enlarge) */
  Rmax += (rw - fmod (Rmax, rw));

  /* get # of rasterlines in r,z */
  nr = (int)(Rmax/rw + 1);
  nz = (int)((Zmax_t-Zmin_t)/rw + 1);

  /* create two dimensional mmlut structure */
  cal->mmlut.origin.x = Ex_t.x0;
  cal->mmlut.origin.y = Ex_t.y0;
  cal->mmlut.origin.z = Zmin_t;
  
  cal->mmlut.nr = nr;
  cal->mmlut.nz = nz;
  cal->mmlut.rw = rw;
  
  // if (cal.mmlut.data != NULL)			// preventing memory leaks, ad holten, 04-2013
  //	free (cal.mmlut.data);
  cal->mmlut.data = (double *) malloc (nr*nz * sizeof (double));
  
  /* fill mmlut structure */
  Ri = (double *) malloc (nr * sizeof (double));
  for (i = 0; i < nr; i++)
    Ri[i] = i*rw;

  Zi = (double *) malloc (nz * sizeof (double));
  for (i = 0; i < nz; i++)
    Zi[i] = Zmin_t + i*rw;

  for (i = 0; i < nr; i++) {
    for (j = 0; j < nz; j++) {
        xyz[0] = Ri[i] + Ex_t.x0;
        xyz[1] = Ex_t.y0;
        xyz[2] = Zi[j];
    
        cal->mmlut.data[i*nz + j] = multimed_r_nlay(cal, cpar->mm, xyz);
    } /* nr */
  } /* nz */
    
  free (Ri);	// preventing memory leaks, Ad Holten, 04-2013
  free (Zi);

  /* when finished initalization, change the setting of the LUT flag */
  cpar->mm->lut = 1;
}


/*  get_mmf_from_mmlut() returns the value of mmf (double) for a 3D point
    using the multimedia look up table.
    
    Arguments:
    double position in 3D space, X,Y,Z
    multimedia look-up table mmlut
*/
double get_mmf_from_mmlut (Calibration *cal, vec3d pos){
  int i, ir,iz, nr,nz, rw, v4[4];
  double R, sr, sz, mmf = 1.0;
  double X,Y,Z;
  
  X = pos[0]; Y = pos[1]; Z = pos[2];
  rw =  cal->mmlut.rw;
  
  if (X == 1.0 && Y == 1.0 && Z == 1.0){
    Z -= cal->mmlut.origin.z; 
    sz = Z/rw; 
    iz = (int) sz; 
    sz -= iz;
    
    X -= cal->mmlut.origin.x;
    Y -= cal->mmlut.origin.y;
    R = sqrt (X*X + Y*Y); 
    sr = R/rw; 
    ir = (int) sr; 
    sr -= ir;
    
    nz =  cal->mmlut.nz;
    nr =  cal->mmlut.nr;
    
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
  
    /* 2. check whether point is inside camera's object volume */
    /* important for epipolar line computation */
    for (i = 0; i < 4; i++)
        if (v4[i] < 0 || v4[i] > nr*nz)
            return (0);
  
    /* interpolate */
    mmf = cal->mmlut.data[v4[0]] * (1-sr)*(1-sz)
        + cal->mmlut.data[v4[1]] * (1-sr)*sz
        + cal->mmlut.data[v4[2]] * sr*(1-sz)
        + cal->mmlut.data[v4[3]] * sr*sz;
  
    return (mmf);     
  } else {
    Z -= cal->mmlut.origin.z; 
    sz = Z/rw;
    iz = (int) sz;
    sz -= iz;
    
    X -= cal->mmlut.origin.x;
    Y -= cal->mmlut.origin.y;
    R = norm(X, Y, 0);
    
    sr = R/rw;
    ir = (int) sr;
    sr -= ir;
  
    nz = cal->mmlut.nz;
    nr = cal->mmlut.nr;
    
    /* check whether point is inside camera's object volume */
    if (ir > nr)
        return (0);
    if (iz < 0  ||  iz > nz)
        return (0);
  
    /* bilinear interpolation in r/z box */
    /* ================================= */
  
    /* get vertices of box */
    v4[0] = ir*nz + iz;
    v4[1] = ir*nz + (iz+1);
    v4[2] = (ir+1)*nz + iz;
    v4[3] = (ir+1)*nz + (iz+1);
  
    /* 2. check wther point is inside camera's object volume */
    /* important for epipolar line computation */
    for (i = 0; i < 4; i++)
        if (v4[i] < 0  ||  v4[i] > nr*nz)
            return (0);
    
    /* interpolate */
    mmf = cal->mmlut.data[v4[0]] * (1-sr)*(1-sz)
        + cal->mmlut.data[v4[1]] * (1-sr)*sz
        + cal->mmlut.data[v4[2]] * sr*(1-sz)
        + cal->mmlut.data[v4[3]] * sr*sz;
  
    return (mmf);
  }
}


/*  volumedimension() finds the limits of the measurement volume in 3D space.
    
    Arguments:
    double pointers to the limits of the volume in x (xmin,xmax), y (ymin, ymax) and
    z (zmin, zmax) directions
    volume_par *vpar - struct holding the observed volume size.
    control_par *cpar - struct holding general control parameters such as 
        image size.
    Calibration *cal - current single-camera positioning and other camera-
        specific data.
*/  
void volumedimension (double *xmax, double *xmin, 
    double *ymax, double *ymin,
    double *zmax, double *zmin,
    volume_par *vpar, control_par *cpar, Calibration *cal)
{
  int i_cam, i, j;
  double X, Y, Z, R, Rmax=0, Zmin, Zmax;
  vec3d pos, a, xyz, xyz_t;
  double x,y;  
  double xc[2], yc[2];  /* image corners */
  Exterior Ex_t;
  double X_t, Y_t, Z_t, Zmin_t,Zmax_t;
  double cross_p[3],cross_c[3];
  
  xc[0] = 0.0;
  xc[1] = (double) cpar->imx;
  yc[0] = 0.0;
  yc[1] = (double) cpar->imy;
  
  Zmin = vpar->Zmin_lay[0];
  Zmax = vpar->Zmax_lay[0];
  
  /* there is a bug, see the original multimed.c  
  https://github.com/3dptv/3dptv/src_c/multimed.c#L899
  obviously we have left and right side that could be smaller or larger in Z
  */
  if (vpar->Zmin_lay[1] < Zmin) Zmin = vpar->Zmin_lay[1];
  if (vpar->Zmax_lay[1] > Zmax) Zmax = vpar->Zmax_lay[1];

  *zmin = Zmin;
  *zmax = Zmax;
    
  /* find extrema of imaged object volume */
  for (i_cam = 0; i_cam < cpar->num_cams; i_cam++) {
      for (i = 0; i < 2; i ++) {
          for (j = 0; j < 2; j++) {
              pixel_to_metric (&x, &y, xc[i], yc[j], cpar);
              x = x - cal[i_cam].int_par.xh;
              y = y - cal[i_cam].int_par.yh;
              correct_brown_affin (x, y, cal[i_cam].added_par, &x, &y);
          
              ray_tracing(x, y, &cal[i_cam], *(cpar->mm), pos, a);
          
              X = pos[0] + (Zmin - pos[2]) * a[0]/a[2];   
              Y = pos[1] + (Zmin - pos[2]) * a[1]/a[2];
              
                     
              if ( X > *xmax) *xmax = X;
              if ( X < *xmin) *xmin = X;
              if ( Y > *ymax) *ymax = Y;
              if ( Y < *ymin) *ymin = Y;

              X = pos[0] + (Zmax - pos[2]) * a[0]/a[2];   
              Y = pos[1] + (Zmax - pos[2]) * a[1]/a[2];
        
              if ( X > *xmax) *xmax = X;
              if ( X < *xmin) *xmin = X;
              if ( Y > *ymax) *ymax = Y;
              if ( Y < *ymin) *ymin = Y;
        
          }
      }
  }
}

