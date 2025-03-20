from libc.stdio cimport FILE

cdef extern from "optv/calibration.h":
    ctypedef double Dmatrix[3][3]
    
    ctypedef struct Exterior:
        double x0, y0, z0
        double omega, phi, kappa
        Dmatrix dm

    ctypedef struct Interior:
        double xh, yh
        double cc

    ctypedef struct Glass:
        double vec_x, vec_y, vec_z
        double n1, n2, n3
        double d

    ctypedef struct ap_52:
        double k1, k2, k3
        double p1, p2
        double scx, she
        int field

    ctypedef struct mmlut:
        double origin[3]
        int nr, nz, rw
        double *data

    ctypedef struct Calibration:
        Exterior ext_par
        Interior int_par
        Glass glass_par
        ap_52 added_par
        mmlut mmlut

    int write_ori(Exterior Ex, Interior I, Glass G, ap_52 ap, char *filename,
        char *add_file)
    int read_ori(Exterior Ex[], Interior I[], Glass G[], char *ori_file,
        ap_52 addp[], char *add_file, char *add_fallback)
    void rotation_matrix(Exterior *ex)
    Calibration* read_calibration(char *ori_file, char *add_file, char *fallback_file)
    int write_calibration(Calibration *cal, char *filename, char *add_file)
