#include <stdio.h>
#include <stdlib.h>
#include "ray_tracing.h"
#include "lsqadj.h"
#include "math.h"
#include "calibration.h"

/* run as
gcc main_test.c ray_tracing.c lsqadj.c -o main_test -I ../include/
*/

int main()
{
	double a[] = {1.0,1.0,1.0};
	double b[] = {2.0,2.0,2.0};
	double n[3];
	
	
	Exterior test_ext = {
        0.0, 0.0, 100.0,
        0.0, 0.0, 0.0, 
        {{1.0, 0.2, -0.3}, 
        {0.2, 1.0, 0.0},
        {-0.3, 0.0, 1.0}}};
	
	norm_cross(a,b,&n[0],&n[1],&n[2]);
		
    printf("\nHello World\n");
    printf("Printing the result of norm_cross\n");
    printf("%6.3f %6.3f %6.3f x %6.3f %6.3f %6.3f\n", a[0],a[1],a[2],b[0],b[1],b[2]);
    printf("%6.3f %6.3f %6.3f\n", n[0],n[1],n[2]);
    
    printf("Matmul test:\n");
    b[0] = 0.0;
    b[1] = 0.0;
    b[2] = 0.0;
    
    printf("a: %6.3f %6.3f %6.3f\n", a[0],a[1],a[2]);
    printf("b: %6.3f %6.3f %6.3f\n", b[0],b[1],b[2]);
    
    matmul (b, (double *) test_ext.dm, a, 3,3,1);
    
    printf("a: %6.3f %6.3f %6.3f\n", a[0],a[1],a[2]);
    printf("b: %6.3f %6.3f %6.3f\n", b[0],b[1],b[2]);
    
    
    
}