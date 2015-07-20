/* Forward declarations for matrix operations defined in lsqadj.c */

#ifndef LSQADJ_H
#define LSQADJ_H

void ata(double *a, double *ata, int m, int n, int n_large);
void atl(double *u, double *a, double *l, int m, int n, int n_large);
void matinv (double *a, int n, int n_large);
void matmul(double *a, double *b, double *c, int m,int n,int k,\
int m_large, int n_large);
void norm_cross(double a[3], double b[3], double n[3]);

#endif
