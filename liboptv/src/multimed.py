def multimed_nlay(cal, mm, pos, Xq, Yq):
    radial_shift = multimed_r_nlay(cal, mm, pos)
    if radial_shift == 1.0:
        Xq[0] = pos[0]
        Yq[0] = pos[1]
    else:
        Xq[0] = cal.ext_par.x0 + (pos[0] - cal.ext_par.x0) * radial_shift
        Yq[0] = cal.ext_par.y0 + (pos[1] - cal.ext_par.y0) * radial_shift

import math

def multimed_r_nlay(cal, mm, pos):
    i, it = 0, 0
    n_iter = 40
    beta1, beta2, beta3, r, rbeta, rdiff, rq, mmf = 1.0, [0]*32, 0.0, 0.0, 0.0, 0.0, 1.0
    X, Y, Z = 0.0, 0.0, 0.0
    zout = 0.0
    
    # 1-medium case
    if mm.n1 == 1 and mm.nlay == 1 and mm.n2[0] == 1 and mm.n3 == 1:
        return 1.0
    
    # interpolation using the existing mmlut
    if cal.mmlut.data != None:
        mmf = get_mmf_from_mmlut(cal, pos)
        if mmf > 0:
            return mmf
    
    # iterative procedure
    X = pos[0]
    Y = pos[1]
    Z = pos[2]

    # Extra layers protrude into water side:
    zout = Z
    for i in range(1, mm.nlay):
        zout += mm.d[i]

    r = math.sqrt(math.pow((X - cal.ext_par.x0), 2) + math.pow((Y - cal.ext_par.y0), 2))
    rq = r
    
    while True:
        beta1 = math.atan(rq/(cal.ext_par.z0 - Z))
        for i in range(0, mm.nlay):
            beta2[i] = math.asin(math.sin(beta1) * mm.n1/mm.n2[i])
        beta3 = math.asin(math.sin(beta1) * mm.n1/mm.n3)
        rbeta = (cal.ext_par.z0 - mm.d[0]) * math.tan(beta1) - zout * math.tan(beta3)
        for i in range(0, mm.nlay):
            rbeta += (mm.d[i] * math.tan(beta2[i]))
        rdiff = r - rbeta
        rq += rdiff
        it += 1
        if (rdiff > 0.001 or rdiff < -0.001) and it < n_iter:
            break
    
    if it >= n_iter:
        print(f"multimed_r_nlay stopped after {n_iter} iterations\n")
        return 1.0
    
    if r != 0:
        return rq/r
    else:
        return 1.0
    

def trans_Cam_Point(ex, mm, gl, pos):
    dist_cam_glas = 0
    dist_point_glas = 0
    dist_o_glas = 0
    # glas inside at water 
    glass_dir = []
    primary_pt = []
    renorm_glass = []
    temp = []

    vec_set(glass_dir, gl.vec_x, gl.vec_y, gl.vec_z)
    vec_set(primary_pt, ex.x0, ex.y0, ex.z0)

    dist_o_glas = vec_norm(glass_dir)
    dist_cam_glas = vec_dot(primary_pt, glass_dir) / dist_o_glas \
                    - dist_o_glas - mm.d[0]
    dist_point_glas = vec_dot(pos, glass_dir) / dist_o_glas - dist_o_glas

    vec_scalar_mul(glass_dir, dist_cam_glas/dist_o_glas, renorm_glass)
    vec_subt(primary_pt, renorm_glass, cross_c)

    vec_scalar_mul(glass_dir, dist_point_glas/dist_o_glas, renorm_glass)
    vec_subt(pos, renorm_glass, cross_p)

    ex_t.x0 = 0
    ex_t.y0 = 0
    ex_t.z0 = dist_cam_glas + mm.d[0]

    vec_scalar_mul(glass_dir, mm.d[0]/dist_o_glas, renorm_glass)
    vec_subt(cross_c, renorm_glass, temp)
    vec_subt(cross_p, temp, temp)

    vec_set(pos_t, vec_norm(temp), 0, dist_point_glas)

"""
Plan: 

1. Calculate the norm of the glass vector
2. Renormalize the glass vector with mm.d[0] and store it in renorm_glass
3. Calculate the after_glass vector by subtracting renorm_glass from cross_c
4. Calculate the temp vector by subtracting after_glass from cross_p
5. Calculate the norm of the temp vector
6. Renormalize the glass vector with -pos_t[2] and store it in renorm_glass
7. Calculate the pos vector by subtracting renorm_glass from after_glass
8. If nVe is greater than 0, then:
   a. Renormalize the temp vector with -pos_t[0] and store it in renorm_glass
   b. Calculate the pos vector by subtracting renorm_glass from pos

"""

def back_trans_Point(pos_t, mm, G, cross_p, cross_c, pos):
    nGl = vec_norm([G.vec_x, G.vec_y, G.vec_z])
    glass_dir = [G.vec_x, G.vec_y, G.vec_z]
    renorm_glass = [0, 0, 0]
    after_glass = [0, 0, 0]
    temp = [0, 0, 0]

    vec_scalar_mul(glass_dir, mm.d[0]/nGl, renorm_glass)
    vec_subt(cross_c, renorm_glass, after_glass)
    vec_subt(cross_p, after_glass, temp)

    nVe = vec_norm(temp)

    vec_scalar_mul(glass_dir, -pos_t[2]/nGl, renorm_glass)
    vec_subt(after_glass, renorm_glass, pos)

    if (nVe > 0):
        vec_scalar_mul(temp, -pos_t[0]/nVe, renorm_glass)
        vec_subt(pos, renorm_glass, pos)
        

def move_along_ray(glob_Z, vertex, direct, out):
    out[0] = vertex[0] + (glob_Z - vertex[2]) * direct[0]/direct[2]   
    out[1] = vertex[1] + (glob_Z - vertex[2]) * direct[1]/direct[2]
    out[2] = glob_Z
    



def init_mmlut(vpar, cpar, cal):
    import numpy as np
    i, j = 0, 0
    Rmax = 0
    rw = 2.0
    Zmin, Zmax = vpar['Zmin_lay'][0], vpar['Zmax_lay'][0]
    Zmin_t, Zmax_t = Zmin, Zmax
    
    xc, yc = np.zeros(2), np.zeros(2)
    xc[1], yc[1] = cpar['imx'], cpar['imy']
  
    for i in range(2):
        for j in range(2):
            x, y = xc[i], yc[j]
            x -= cal['int_par']['xh']
            y -= cal['int_par']['yh']
            correct_brown_affin(x, y, cal['added_par'], x, y)  
            ray_tracing(x, y, cal, cpar['mm'][0], pos, a)
            move_along_ray(Zmin, pos, a, xyz)
            trans_Cam_Point(cal['ext_par'], cpar['mm'][0], cal['glass_par'], xyz,
                cal['ext_par'], xyz_t, cross_p, cross_c)

            if xyz_t[2] < Zmin_t:
                Zmin_t = xyz_t[2]
            if xyz_t[2] > Zmax_t:
                Zmax_t = xyz_t[2]

            R = norm((xyz_t[0] - cal['ext_par']['x0']), (xyz_t[1] - cal['ext_par']['y0']), 0)
            if R > Rmax:
                Rmax = R

            move_along_ray(Zmax, pos, a, xyz)
            trans_Cam_Point(cal['ext_par'], cpar['mm'][0], cal['glass_par'], xyz,\
                cal_t['ext_par'], xyz_t, cross_p, cross_c)

            if xyz_t[2] < Zmin_t:
                Zmin_t = xyz_t[2]
            if xyz_t[2] > Zmax_t:
                Zmax_t = xyz_t[2]

            R = norm((xyz_t[0] - cal_t['ext_par']['x0']), (xyz_t[1] - cal_t['ext_par']['y0']), 0)
            if R > Rmax:
                Rmax = R

    Rmax += (rw - Rmax % rw)
    nr = int(Rmax / rw + 1)
    nz = int((Zmax_t - Zmin_t) / rw + 1)

    # create two dimensional mmlut structure
    cal['mmlut']['origin'] = np.array([cal_t['ext_par']['x0'], cal_t['ext_par']['y0'], Zmin_t])
    cal['mmlut']['nr'] = nr
    cal['mmlut']['nz'] = nz
    cal['mmlut']['rw'] = rw
  
    if cal['mmlut']['data'] is None:
        data = np.zeros(nr*nz)
  
        # fill mmlut structure
        Ri = np.arange(nr) * rw
        Zi = np.arange(nz) * rw + Zmin_t
  
        for i in range(nr):
            for j in range(nz):
                xyz = np.array([Ri[i] + cal_t['ext_par']['x0'], cal_t['ext_par']['y0'], Zi[j]])
                data[i*nz + j] = multimed_r_nlay(cal_t, cpar['mm'], xyz)
    
        cal['mmlut']['data'] = data
        
        
        
def get_mmf_from_mmlut(cal, pos):
    i, ir, iz, nr, nz, rw, v4 = 0, 0, 0, 0, 0, 0, [0,0,0,0]
    mmf = 1.0
    temp = [0.0, 0.0, 0.0]

    rw = cal.mmlut.rw

    vec_subt(pos, cal.mmlut.origin, temp)
    sz = temp[2]/rw
    iz = int(sz)
    sz -= iz

    R = norm(temp[0], temp[1], 0)

    sr = R/rw
    ir = int(sr)
    sr -= ir

    nz = cal.mmlut.nz
    nr = cal.mmlut.nr

    # check whether point is inside camera's object volume
    if ir > nr:
        return 0
    if iz < 0 or iz > nz:
        return 0

    # bilinear interpolation in r/z box
    # =================================

    # get vertices of box
    v4[0] = ir * nz + iz
    v4[1] = ir * nz + (iz+1)
    v4[2] = (ir+1) * nz + iz
    v4[3] = (ir+1) * nz + (iz+1)

    # 2. check whether point is inside camera's object volume
    # important for epipolar line computation
    for i in range(4):
        if v4[i] < 0 or v4[i] > nr * nz:
            return 0

    # interpolate
    mmf = cal.mmlut.data[v4[0]] * (1-sr)*(1-sz) + \
        cal.mmlut.data[v4[1]] * (1-sr)*sz + \
        cal.mmlut.data[v4[2]] * sr*(1-sz) + \
        cal.mmlut.data[v4[3]] * sr*sz

    return mmf


def volumedimension(xmax, xmin, ymax, ymin, zmax, zmin, vpar, cpar, cal):
    
    xc = [0.0, cpar['imx']]
    yc = [0.0, cpar['imy']]
    
    Zmin = vpar['Zmin_lay'][0]
    Zmax = vpar['Zmax_lay'][0]
    
    if (vpar['Zmin_lay'][1] < Zmin): 
        Zmin = vpar['Zmin_lay'][1]
    if (vpar['Zmax_lay'][1] > Zmax): 
        Zmax = vpar['Zmax_lay'][1]
    
    zmin = Zmin
    zmax = Zmax
    
    for i_cam in range(cpar['num_cams']):
        for i in range(2):
            for j in range(2):                
                x, y = pixel_to_metric(xc[i], yc[j], cpar)
                
                x -= cal[i_cam]['int_par']['xh']
                y -= cal[i_cam]['int_par']['yh']
                
                x, y = correct_brown_affin(x, y, cal[i_cam]['added_par'])
                
                pos, a = ray_tracing(x, y, cal[i_cam], cpar['mm'])
                
                X = pos[0] + (Zmin - pos[2]) * a[0] / a[2]
                Y = pos[1] + (Zmin - pos[2]) * a[1] / a[2]
                
                if (X > xmax): 
                    xmax = X
                if (X < xmin):
                    xmin = X
                if (Y > ymax): 
                    ymax = Y
                if (Y < ymin): 
                    ymin = Y

                X = pos[0] + (Zmax - pos[2]) * a[0] / a[2]
                Y = pos[1] + (Zmax - pos[2]) * a[1] / a[2]
                
                if (X > xmax): 
                    xmax = X
                if (X < xmin): 
                    xmin = X
                if (Y > ymax): 
                    ymax = Y
                if (Y < ymin): 
                    ymin = Y
                    
    return (xmax, xmin, ymax, ymin, zmax, zmin)
