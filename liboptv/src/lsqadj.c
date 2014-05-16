#include "lsqadj.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define EPS 1e-5


/* parts of code of adjlib.c from Horst Beyer, Hannes Kirndorfer */


/* Multiply transpose of a matrix A by matrix A itself, creating symmetric matrix
*  with the option of working with the sub-matrix only 
*
*   Arguments:
*   a - matrix of doubles of the size (m x n_large).
*   ata  - matrix of the result multiply(a.T,a) of size (n x n)
*   m - number of rows in matrix a
*   n - number of rows and columns in the output ata - the size of the sub-matrix
*   n_large - number of columns in matrix a
*/

void ata (double *a, double *ata, int m, int n, int n_large ) {
/* matrix a and result matrix ata = at a 
               a is m * n, ata is n * n  */

  register int      i, j, k;
  
  for (i = 0; i < n; i++)
    {
      for (j = 0; j < n; j++)
    {
      *(ata+i*n_large+j) = 0.0;
      for (k = 0; k < m; k++)
        *(ata+i*n_large+j) +=  *(a+k*n_large+i)  * *(a+k*n_large+j);
    }
    }
}



/* Multiply transpose of a matrix A by vector l , creating vector u
*  with the option of working with the sub-vector only, when n < n_large 
*
*   Arguments:
*   u - vector of doubles of the size (n x 1)
*   a - matrix of doubles of the size (m x n_large).
*   l  - vector of doubles (m x 1)
*   m - number of rows in matrix a
*   n - length of the output u - the size of the sub-matrix
*   n_large - number of columns in matrix a
*/

void atl (double *u, double *a, double *l, int m, int n, int n_large) {
/* matrix a , vector l and 
             result vector u = at l ,  a(m,n)  */

  int      i, k;
  
  for (i = 0; i < n; i++)
    {
      *(u + i) = 0.0;
      for (k = 0; k < m; k++)
    *(u + i) += *(a + k * n_large + i) * *(l + k);
    }  
}


/* Calculate inverse of a matrix A,
*  with the option of working with the sub-vector only, when n < n_large 
*
*   Arguments:
*   a - matrix of doubles of the size (n_large x n_large).
*   n - size of the output - size of the sub-matrix, number of observations
*   n_large - number of rows and columns in matrix a
*/

void matinv (double *a, int n, int n_large) {
  int      ipiv, irow, icol;
  double   pivot;   /* pivot element = 1.0 / aii */
  double    npivot; /*  negative of pivot */
  
  
  for (ipiv = 0; ipiv < n; ipiv++)
    {
      pivot = 1.0 / *(a + ipiv * n_large + ipiv);
      npivot = - pivot;
      for (irow = 0; irow < n; irow++)
    {
      for (icol = 0; icol < n; icol++)
        {
          if (irow != ipiv && icol != ipiv)
        {
          *(a + irow * n_large + icol) -= *(a + ipiv * n_large + icol) * 
            *(a + irow * n_large + ipiv) * pivot;
        }
        }
    }
      for (icol = 0; icol < n; icol++)
    {
      if (ipiv != icol) 
        *(a + ipiv * n_large + icol) *= npivot;
    }
      for (irow = 0; irow < n; irow++)
    {
      if (ipiv != irow)
        *(a + irow * n_large + ipiv) *= pivot;
    }
      *(a + ipiv * n_large + ipiv) = pivot;
    }
}   /* end matinv */



/* Calculate dot product of a matrix 'b' of the size (m_large x n_large) with 
*  a vector 'c' of the size (n_large x 1) to get vector 'a' of the size 
*  (m x 1), when m < m_large and n < n_large
*  i.e. one can get dot product of the submatrix of b with sub-vector of c
*   when n_large > n and m_large > m
*   Arguments:
*   a - output vector of doubles of the size (m x 1).
*   b - matrix of doubles of the size (m x n)
*   c - vector of doubles of the size (n x 1)
*   m - integer, number of rows of a
*   n - integer, number of columns in a
*   k - integer, size of the vector output 'a', typically k = 1
*/

void matmul (double *a, double *b, double *c, int m,int n,int k,\
int m_large, int n_large) { 

int    i,j,l;
double  x,*pa,*pb,*pc;

for (i=0; i<k; i++) {  
  pb = b;
  pa = a++;
  for (j=0; j<m; j++) {  
    pc = c;
    x = 0.0;
    for (l=0; l<n; l++) {  
      x = x + *pb++ * *pc;
      pc += k;
      }
    for (l=0;l<n_large-n;l++) {
      pb++;
      pc += k;
      }
    *pa = x;
    pa += k;
    }
  for (j=0;j<m_large-m;j++) {
    pa += k;
    }
  c++;
  }
}


/* Unit vector which is normal to two vectors, calculated as
  a unit vector in the direction of the cross product of two vectors
* Arguments
* a - vector of doubles 3x1
* b - vector of doubles 3x1
* n - output, vector of doubles 3x1
*/

void norm_cross(double a[3], double b[3], double n[3]) {

    n[0]=a[1]*b[2]-a[2]*b[1];
    n[1]=a[2]*b[0]-a[0]*b[2];
    n[2]=a[0]*b[1]-a[1]*b[0];
    
    unit_vector(n);
    
    }
}

/* Scalar multiplication of two vectors of length 3 x 1
   Arguments:
   a,b = double vectors 3 x 1
   function dot returns double scalar 
*/

double dot(double a[3], double b[3]) {
    double d;
    d = a[0]*b[0] + a[1]*b[1] + a[2]*b[2];

    return d;
}


/* Modulus of a vector or norm of the vector 
   Arguments:
   a - double vector of 3 x 1
   function returns real positive double value 
   TODO: in essence it is a sqrt(dot(a,a))
   could be depreciated for simplicity
*/
void modu(double a[3], double *m) {
    double m;

    m = sqrt(dot(a,a));
}



/* returns a unit vector, normalized by the norm */
void unit_vector(double a[3]){
    double dummy; 

    dummy = modu(a);    
    
    /* if the vector is zero length we return it as is */
    if (dummy < EPS) dummy = 1.0;
    
    a[0] = a[0]/dummy;
    a[1] = a[1]/dummy;
    a[2] = a[2]/dummy;
}

