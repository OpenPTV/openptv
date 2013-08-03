/* Forward declarations for matrix operations defined in lsqadj.c */

#ifndef LSQADJ_H
#define LSQADJ_H

#include <stdlib.h>
#include <stdio.h>


void ata(double *a, double *ata, int m, int n);
void ata_v2(double *a, double *ata, int m, int n, int n_large);
void atl(double *a, double *u, double *l, int m, int n);
void atl_v2 (double *u, double *a, double *l, int m, int n, int n_large);


void matinv(double *a, int n);
void matinv_v2 (double *a, int n, int n_large);
void matmul(double *a, double *b, double *c, int m, int n, int k);
void matmul_v2 (double *a, double *b, double *c, int m,int n,int k,\
int m_large, int n_large);
void transp (double a[], int m, int n);
void mat_transpose(double *mat1, double *mat2, int m, int n);

#endif
