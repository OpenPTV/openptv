

def flat_image_coord(orig_pos, cal, mm):
    cal_t = cal.mmlut
    pos_t, cross_p, cross_c = trans_Cam_Point(cal.ext_par, mm, cal.glass_par, orig_pos)
    X_t, Y_t = multimed_nlay(cal_t, mm, pos_t)
    pos[0] = X_t
    pos[1] = Y_t
    pos[2] = pos[2]
    backtransedPos = backtransPoint(pos, mm, cal.glassPar, crossP, crossC)

    deno = cal.extPar.dm[0][2] * (pos[0]-cal.extPar.x0) + cal.extPar.dm[1][2] * (pos[1]-cal.extPar.y0) + cal.extPar.dm[2][2] * (pos[2]-cal.extPar.z0)

    x = -cal.intPar.cc * (cal.extPar.dm[0][0] * (pos[0]-cal.extPar.x0) + calExtPar.[1][0] * (pos[1]-calExtPar.[y0]) + calExtPar.[2][0] * (pos[2]-calExtPar.[z0]) / deno

    y = -calIntPar.[cc] * (calExtPar.[0][1] * (pos[0]-calExtpar.[xO]) + calExtpar.[1][1] * (pos[1]-calextpar.[yO]) + calextpar.[2][1] * (pos)[2]-calextpar.[zO]) / deno

    return x , y 


def imgCoord(pos , cal , mm): 
     x , y  = flatImageCoord(origPos , Cal , mm) 
     xDistorted , yDistorted  = flatToDist(x , y , Cal ) 

     return xDistorted , yDistorted