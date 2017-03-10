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
    double zout;
    
    /* 1-medium case */
    if (mm->n1 == 1 && mm->nlay == 1 && mm->n2[0]== 1 && mm->n3 == 1)
        return (1.0);
    
    /* interpolation using the existing mmlut */
	if (cal->mmlut.data != NULL) {
        mmf = get_mmf_from_mmlut(cal, pos);
        if (mmf > 0) return (mmf);
    }
    
    /* iterative procedure */
    X = pos[0];
    Y = pos[1];
    Z = pos[2];
    
    /* Extra layers protrude into water side: */
    zout = Z;
    for (i = 1; i < mm->nlay; i++)
        zout += mm->d[i];
    
    r = norm((X - cal->ext_par.x0), (Y - cal->ext_par.y0), 0);
    rq = r;
  
    do
    {
        beta1 = atan (rq/(cal->ext_par.z0 - Z));
        for (i = 0; i < mm->nlay; i++)
            beta2[i] = asin (sin(beta1) * mm->n1/mm->n2[i]);
        beta3 = asin (sin(beta1) * mm->n1/mm->n3);

        rbeta = (cal->ext_par.z0 - mm->d[0]) * tan(beta1) - zout * tan(beta3);
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

/*  trans_Cam_Point() projects global-coordinate points of the camera and an
    image points on the glass surface separating those points.
    
    Input arguments:
    Exterior ex - holding global-coordinates camera position.
    mm_np mm - holding glass thickness.
    Glass gl - holding the glass position in global-coordinates as a 
        glass-normal vector.
    vec3d pos - the global coordinates of the observed point.
    
    Output arguments:
    Exterior ex_t - only the primary point fields are used, and are set 
        to the glass/Z axis intersection.
    double cross_p[3] - the observed point projection coordinates in global
        coordinates.
    double cross_c[3] - same for the camera position. Since the camera point
        is projected on the other side of the glass, it'll have a small 
        difference in Z value from ``cross_p``.
*/
void trans_Cam_Point(Exterior ex, mm_np mm, Glass gl, vec3d pos, 
    Exterior *ex_t, vec3d pos_t, double cross_p[3], double cross_c[3])
{
    double dist_cam_glas,dist_point_glas,dist_o_glas; //glas inside at water 
    vec3d glass_dir, primary_pt, renorm_glass, temp;
    
    vec_set(glass_dir, gl.vec_x, gl.vec_y, gl.vec_z);
    vec_set(primary_pt, ex.x0, ex.y0, ex.z0);

    dist_o_glas = vec_norm(glass_dir);
    dist_cam_glas = vec_dot(primary_pt, glass_dir) / dist_o_glas \
        - dist_o_glas - mm.d[0];
    dist_point_glas = vec_dot(pos, glass_dir) / dist_o_glas - dist_o_glas;

    vec_scalar_mul(glass_dir, dist_cam_glas/dist_o_glas, renorm_glass);
    vec_subt(primary_pt, renorm_glass, cross_c);

    vec_scalar_mul(glass_dir, dist_point_glas/dist_o_glas, renorm_glass);
    vec_subt(pos, renorm_glass, cross_p);

    ex_t->x0 = 0.;
    ex_t->y0 = 0.;
    ex_t->z0 = dist_cam_glas + mm.d[0];

    vec_scalar_mul(glass_dir, mm.d[0]/dist_o_glas, renorm_glass);
    vec_subt(cross_c, renorm_glass, temp);
    vec_subt(cross_p, temp, temp);
    
    vec_set(pos_t, vec_norm(temp), 0, dist_point_glas);
}

/* the opposite direction transfer from X_t,Y_t,Z_t to the X,Y,Z in 3D space */

void back_trans_Point(vec3d pos_t, mm_np mm, Glass G, double cross_p[3], 
    double cross_c[3], vec3d pos)
{  
    double nVe, nGl;
    vec3d glass_dir, renorm_glass, after_glass, temp;
    
    vec_set(glass_dir, G.vec_x, G.vec_y, G.vec_z);
    nGl = vec_norm(glass_dir);
    
    vec_scalar_mul(glass_dir, mm.d[0]/nGl, renorm_glass);
    vec_subt(cross_c, renorm_glass, after_glass);
    vec_subt(cross_p, after_glass, temp);
    
    nVe = vec_norm(temp);
    
    vec_scalar_mul(glass_dir, -pos_t[2]/nGl, renorm_glass);
    vec_subt(after_glass, renorm_glass, pos);
    
    if (nVe > 0) {  
        /* We need this for when the cam-point line is exactly perp. to glass.
         The degenerate case will have nVe == 0 and produce NaNs on the
         following calculations.
        */
        vec_scalar_mul(temp, -pos_t[0]/nVe, renorm_glass);
        vec_subt(pos, renorm_glass, pos);
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
  double R, Zmin, Rmax=0, Zmax;
  vec3d pos, a, xyz, xyz_t; 
  double x,y, *Ri,*Zi, *data;
  double rw = 2.0; 

  /* A frame representing a point outside tank, middle of glass*/
  Calibration cal_t;

  double Zmin_t, Zmax_t;
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
  
  if (vpar->Zmin_lay[1] < Zmin) Zmin = vpar->Zmin_lay[1];
  if (vpar->Zmax_lay[1] > Zmax) Zmax = vpar->Zmax_lay[1];
 
  Zmin -= fmod (Zmin, rw);
  Zmax += (rw - fmod (Zmax, rw));
  Zmin_t=Zmin;
  Zmax_t=Zmax;

  /* intersect with image vertices rays */
  cal_t = *cal;

  for (i = 0; i < 2; i ++) {
      for (j = 0; j < 2; j++) {
          pixel_to_metric (&x, &y, xc[i], yc[j], cpar);

          x = x - cal->int_par.xh;
          y = y - cal->int_par.yh;
  
          correct_brown_affin (x, y, cal->added_par, &x,&y);  
          ray_tracing(x, y, cal, *(cpar->mm), pos, a);
          
          move_along_ray(Zmin, pos, a, xyz);
          trans_Cam_Point(cal->ext_par, *(cpar->mm), cal->glass_par, xyz, \
            &(cal_t.ext_par), xyz_t, (double *)cross_p, (double *)cross_c);

          if( xyz_t[2] < Zmin_t ) Zmin_t = xyz_t[2];
          if( xyz_t[2] > Zmax_t ) Zmax_t = xyz_t[2];

          R = norm((xyz_t[0] - cal_t.ext_par.x0), (xyz_t[1] - cal_t.ext_par.y0), 0);
          if (R > Rmax)
              Rmax = R;
           
          move_along_ray(Zmax, pos, a, xyz);
          trans_Cam_Point(cal->ext_par, *(cpar->mm), cal->glass_par, xyz,\
              &(cal_t.ext_par), xyz_t, (double *)cross_p, (double *)cross_c);
  
          if( xyz_t[2] < Zmin_t ) Zmin_t = xyz_t[2];
          if( xyz_t[2] > Zmax_t ) Zmax_t = xyz_t[2];

          R = norm((xyz_t[0] - cal_t.ext_par.x0), (xyz_t[1] - cal_t.ext_par.y0), 0);
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
  vec_set(cal->mmlut.origin, cal_t.ext_par.x0, cal_t.ext_par.y0, Zmin_t);
  
  cal->mmlut.nr = nr;
  cal->mmlut.nz = nz;
  cal->mmlut.rw = rw;
  
  if (cal->mmlut.data == NULL) {
      data = (double *) malloc (nr*nz * sizeof (double));
  
      /* fill mmlut structure */
      Ri = (double *) malloc (nr * sizeof (double));
      for (i = 0; i < nr; i++)
        Ri[i] = i*rw;

      Zi = (double *) malloc (nz * sizeof (double));
      for (i = 0; i < nz; i++)
        Zi[i] = Zmin_t + i*rw;

      for (i = 0; i < nr; i++) {
        for (j = 0; j < nz; j++) {
            vec_set(xyz, Ri[i] + cal_t.ext_par.x0, cal_t.ext_par.y0, Zi[j]);
            data[i*nz + j] = multimed_r_nlay(&cal_t, cpar->mm, xyz);
        } /* nr */
      } /* nz */
    
      free (Ri);
      free (Zi);
      cal->mmlut.data = data;
    }
}


/*  get_mmf_from_mmlut() returns the value of mmf (double) for a 3D point
    using the multimedia look up table.
    
    Arguments:
    pos - vector of 3 doubles, position in 3D space 
   Calibration parameters pointer, *cal
*/
double get_mmf_from_mmlut (Calibration *cal, vec3d pos){
    int i, ir,iz, nr,nz, rw, v4[4];
    double R, sr, sz, mmf = 1.0;
    vec3d temp;
    
    rw = cal->mmlut.rw;
  
    vec_subt(pos, cal->mmlut.origin, temp);
    sz = temp[2]/rw;
    iz = (int) sz;
    sz -= iz;
    
    R = norm(temp[0], temp[1], 0);
    
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


/*  volumedimension() finds the limits of the measurement volume in 3D space.
    
    Arguments:
    double pointers to the limits of the volume in x (xmin,xmax), y (ymin, ymax) and
    z (zmin, zmax) directions
    volume_par *vpar - struct holding the observed volume size.
    control_par *cpar - struct holding general control parameters such as 
        image size.
    Calibration *cal - scene parameters.
*/  
void volumedimension (double *xmax, double *xmin, 
    double *ymax, double *ymin,
    double *zmax, double *zmin,
    volume_par *vpar, control_par *cpar, Calibration **cal)
{
  int i_cam, i, j;
  double X, Y;
  vec3d pos, a;
  double x,y;  
  double xc[2], yc[2];  /* image corners */
  double Zmin, Zmax;
  
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
              x = x - cal[i_cam]->int_par.xh;
              y = y - cal[i_cam]->int_par.yh;
              correct_brown_affin (x, y, cal[i_cam]->added_par, &x, &y);
          
              ray_tracing(x, y, cal[i_cam], *(cpar->mm), pos, a);
          
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

