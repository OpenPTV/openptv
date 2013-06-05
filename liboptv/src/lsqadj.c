/* parts of code of adjlib.c from Horst Beyer, Hannes Kirndorfer */

#include "ptv.h"

void ata ( a, ata, m, n )
int      m, n;
double   *a, *ata;  /* matrix a and resultmatrix ata = at a 
		       a is m * n, ata is n * n  */
{
  register int      i, j, k;
  
  for (i = 0; i < n; i++)
    {
      for (j = 0; j < n; j++)
	{
	  *(ata+i*n+j) = 0.0;
	  for (k = 0; k < m; k++)
	    *(ata+i*n+j) +=  *(a+k*n+i)  * *(a+k*n+j);
	}
    }
}	/* end ata.c */

void ata_v2 ( a, ata, m, n, n_large )
int      m, n, n_large;
double   *a, *ata;  /* matrix a and resultmatrix ata = at a 
		       a is m * n, ata is n * n  */
{
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
}	/* end ata.c */


void atl (u, a, l, m, n)
int      m, n;
double   *a, *u, *l;  /* matrix a , vector l and 
			 resultvector u = at l ,  a(m,n)  */

{  
  int      i, k;
  
  for (i = 0; i < n; i++)
    {
      *(u + i) = 0.0;
      for (k = 0; k < m; k++)
	*(u + i) += *(a + k * n + i) * *(l + k);
    }
  
} /* end atl.c */

void atl_v2 (u, a, l, m, n, n_large)
int      m, n, n_large;
double   *a, *u, *l;  /* matrix a , vector l and 
			 resultvector u = at l ,  a(m,n)  */

{  
  int      i, k;
  
  for (i = 0; i < n; i++)
    {
      *(u + i) = 0.0;
      for (k = 0; k < m; k++)
	*(u + i) += *(a + k * n_large + i) * *(l + k);
    }
  
} /* end atl.c */


void matinv (a, n)

double   *a;	/* input matrix size n * n */
int      n;         /* number of observations */

{
  int      ipiv, irow, icol;
  double   pivot;	/* pivot element = 1.0 / aii */
  double	npivot;	/*	negative of pivot */
  
  
  for (ipiv = 0; ipiv < n; ipiv++)
    {
      pivot = 1.0 / *(a + ipiv * n + ipiv);
      npivot = - pivot;
      for (irow = 0; irow < n; irow++)
	{
	  for (icol = 0; icol < n; icol++)
	    {
	      if (irow != ipiv && icol != ipiv)
		{
		  *(a + irow * n + icol) -= *(a + ipiv * n + icol) * 
		    *(a + irow * n + ipiv) * pivot;
		}
	    }
	}
      for (icol = 0; icol < n; icol++)
	{
	  if (ipiv != icol) 
	    *(a + ipiv * n + icol) *= npivot;
	}
      for (irow = 0; irow < n; irow++)
	{
	  if (ipiv != irow)
	    *(a + irow * n + ipiv) *= pivot;
	}
      *(a + ipiv * n + ipiv) = pivot;
    }
}	/* end matinv */

void matinv_v2 (a, n, n_large)

double   *a;	/* input matrix size n * n */
int      n, n_large;         /* number of observations */

{
  int      ipiv, irow, icol;
  double   pivot;	/* pivot element = 1.0 / aii */
  double	npivot;	/*	negative of pivot */
  
  
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
}	/* end matinv */


void matmul (a,b,c,m,n,k)
int    m,n,k;
double  *a,*b,*c;

{  int    i,j,l;
double  x,*pa,*pb,*pc;

for (i=0; i<k; i++)
  {  pb = b;
  pa = a++;
  for (j=0; j<m; j++)
    {  pc = c;
    x = 0.0;
    for (l=0; l<n; l++)
      {  x = x + *pb++ * *pc;
      pc += k;
      }
    *pa = x;
    pa += k;
    }
  c++;
  }
}

void matmul_v2 (a,b,c,m,n,k,m_large,n_large)
int    m,n,k,m_large,n_large;
double  *a,*b,*c;

{  int    i,j,l;
double  x,*pa,*pb,*pc;

for (i=0; i<k; i++)
  {  pb = b;
  pa = a++;
  for (j=0; j<m; j++)
    {  pc = c;
    x = 0.0;
    for (l=0; l<n; l++)
      {  x = x + *pb++ * *pc;
      pc += k;
      }
	for (l=0;l<n_large-n;l++)
	  {pb++;
       pc += k;
	  }
    *pa = x;
    pa += k;
    }
  for (j=0;j<m_large-m;j++)
    {pa += k;}
  c++;
  }
}

void transp (a,m,n)
double  a[];
int    m,n;
{  
  double  *b,*c,*d,*e;
  int    i,j;
  
  b = (double*) malloc (m*n*sizeof(double));
  if (b == 0) goto err;
  d = a;
  e = b;
  
  for (i=0; i<m; i++)
    {  c = b++;
    for (j=0; j<n; j++)
      {  *c = *a++;
      c += m;
      }
    }
  
  for (i=0; i<m*n; i++)
    *d++ = *e++;
  /*
    free (b);
    */   
  return;
  
err:
  printf ("\n\n ***   no memory space in C-subroutine transp   ***");
  printf ("\n\n");
  exit (-1);
}

void mat_transpose (mat1, mat2, m, n)
double	*mat1, *mat2;
int		n, m;
{
  int		i, j;
   
  for (i=0; i<m; i++)	for (j=0; j<n; j++)	*(mat2+j*m+i) = *(mat1+i*n+j);
}
