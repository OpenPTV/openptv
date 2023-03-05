import numpy as np

def ray_tracing(x, y, cal, mm):
    d1, d2, c, dist_cam_glass, n, p = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    start_dir = np.zeros(3)
    primary_point = np.array([cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0])
    glass_dir = np.zeros(3)
    bp = np.zeros(3)
    tmp1 = np.zeros(3)
    tmp2 = np.zeros(3)
    Xb = np.zeros(3)
    a2 = np.zeros(3)
    X = np.zeros(3)
    out = np.zeros(3)
    
    # Initial ray direction in global coordinate system
    tmp1 = np.array([x, y, -1 * cal.int_par.cc])
    tmp1 = tmp1 / np.linalg.norm(tmp1)
    start_dir = np.dot(cal.ext_par.dm, tmp1)
    
    # Project start ray on glass vector to find n1/n2 interface.
    tmp1 = np.array([cal.glass_par.vec_x, cal.glass_par.vec_y, cal.glass_par.vec_z])
    tmp1 = tmp1 / np.linalg.norm(tmp1)
    glass_dir = tmp1.copy()
    c = np.linalg.norm(tmp1) + mm.d[0]
    dist_cam_glass = np.dot(glass_dir, primary_point) - c
    d1 = -dist_cam_glass / np.dot(glass_dir, start_dir)
    tmp1 = start_dir * d1
    Xb = primary_point + tmp1
    
    # Break down ray into glass-normal and glass-parallel components.
    n = np.dot(start_dir, glass_dir)
    tmp1 = glass_dir * n
    bp = start_dir - tmp1
    bp = bp / np.linalg.norm(bp)
    
    # Transform to direction inside glass, using Snell's law.
    p = np.sqrt(1 - n * n) * mm.n1 / mm.n2[0]  # glass parallel
    n = -np.sqrt(1 - p * p)  # glass normal
    
    # Propagation length in glass parallel to glass vector.
    tmp1 = bp * p
    tmp2 = glass_dir * n
    a2 = tmp1 + tmp2
    d2 = mm.d[0] / abs(np.dot(glass_dir, a2))
    
    # Point on the horizontal plane between n2,n3.
    tmp1 = a2 * d2
    X = Xb + tmp1
    
    # Again, direction in next medium.
    n = np.dot(a2, glass_dir)
    tmp2 = a2 - glass_dir * n
    bp = tmp2 / np.linalg.norm(tmp2)
    
    p = np.sqrt(1 - n * n)
    p = p * mm.n2[0] / mm.n3
    n = -np.sqrt(1 - p * p)
    
    tmp1 = bp * p
    tmp2 = glass_dir * n
    out = tmp1 + tmp2
    
    return X, out
