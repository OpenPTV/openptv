def write_ori(Ex, I, G, ap, filename, add_file):
    """Write exterior and interior orientation, and - if available, parameters for
    distortion corrections.
    
    Arguments:
    Exterior Ex - exterior orientation.
    Interior I - interior orientation.
    Glass G - glass parameters.
    ap_52 addp - optional additional (distortion) parameters. NULL is fine if
       add_file is NULL.
    char *filename - path of file to contain interior, exterior and glass
       orientation data.
    char *add_file - path of file to contain added (distortions) parameters.
    """
    success = 0
    try:
        with open(filename, 'w') as fp:
            fp.write("{:11.8f} {:11.8f} {:11.8f}\n    {:10.8f}  {:10.8f}  {:10.8f}\n\n".format(
                Ex.x0, Ex.y0, Ex.z0, Ex.omega, Ex.phi, Ex.kappa))
            for i in range(3):
                fp.write("    {:10.7f} {:10.7f} {:10.7f}\n".format(
                    Ex.dm[i][0], Ex.dm[i][1], Ex.dm[i][2]))
            fp.write("\n    {:8.4f} {:8.4f}\n    {:8.4f}\n".format(I.xh, I.yh, I.cc))
            fp.write("\n    {:20.15f} {:20.15f}  {:20.15f}\n".format(G.vec_x, G.vec_y, G.vec_z))
    except IOError:
        print("Can't open ascii file: {}".format(filename))
        return success
    
    if add_file is None:
        return success
    
    try:
        with open(add_file, 'w') as fp:
            fp.write("{:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f}".format(
                ap.k1, ap.k2, ap.k3, ap.p1, ap.p2, ap.scx, ap.she))
            success = 1
    except IOError:
        print("Can't open ascii file: {}".format(add_file))
        return success
    
    return success


def read_ori (Ex, I, G, ori_file, addp, add_file, add_fallback):
    """
    Reads the orientation file and the additional parameters file.
    """
    fp = open (ori_file, "r")
    if not fp:
        print("Can't open ORI file: %s\n", ori_file)
        return 0
    
    # Exterior
    scan_res = fscanf (fp, "%lf %lf %lf %lf %lf %lf",
	  &(Ex->x0), &(Ex->y0), &(Ex->z0),
	  &(Ex->omega), &(Ex->phi), &(Ex->kappa))
    if (scan_res != 6):
        return 0
    
    # Exterior rotation matrix
    for i in range(3):
        scan_res = fscanf (fp, " %lf %lf %lf",
            &(Ex->dm[i][0]), &(Ex->dm[i][1]), &(Ex->dm[i][2]))
        if (scan_res != 3):
            return 0
    
    # Interior
    scan_res = fscanf (fp, "%lf %lf %lf", &(I->xh), &(I->yh), &(I->cc))
    if (scan_res != 3):
        return 0
    
    # Glass
    scan_res = fscanf (fp, "%lf %lf %lf", &(G->vec_x), &(G->vec_y), &(G->vec_z))
    if (scan_res != 3):
        return 0
    
    fp.close()
    
    # Additional:
    fp = open(add_file, "r")
    if ((fp == NULL) and add_fallback):
        fp = open (add_fallback, "r")
    
    if fp:
        scan_res = fscanf (fp, "%lf %lf %lf %lf %lf %lf %lf",
            &(addp->k1), &(addp->k2), &(addp->k3), &(addp->p1), &(addp->p2),
            &(addp->scx), &(addp->she))
        fp.close()
    else:
        print("no addpar fallback used\n") # Waits for proper logging.
        addp->k1 = addp->k2 = addp->k3 = addp->p1 = addp->p2 = addp->she = 0.0
        addp->scx=1.0
    
    return 1

