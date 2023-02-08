/****************************************************************************

Routine:                peakfitting.c

Author/Copyright:       Hans-Gerd Maas

Address:                Institute of Geodesy and Photogrammetry
                    ETH - Hoenggerberg
                    CH - 8093 Zurich

Creation Date:          Feb. 1990

Description:

****************************************************************************/

#include "segmentation.h"
#include <string.h>
#include <stdio.h>

typedef short targpix[2];


/*  targ_rec() thresholding and center of gravity with a peak fitting technique 
    uses 4 neighbours for connectivity and 8 to find local maxima, see:
    https://en.wikipedia.org/wiki/Connected-component_labeling     

Arguments:
    unsigned char *img - image buffer
    target_par *targ_par - target dots parameters: size limits, sum of grey 
        values, intensity thresholds, etc.
    int xmin, xmax, ymin, ymax - search area, limits of the field of view.
    control_par *cpar - control parameters.
    int num_cam - number of the camera (0... cal_par->num_cams), used for 
        finding the relevant threshold in the target parameters struct.

Output:
    target pix[] - array of found dots of centroids, sum of grey values and 
        a point identifier (number of the dot).

Returns:
    number of targets found.
*/

int targ_rec (unsigned char *img, target_par *targ_par, int xmin, 
    int xmax, int ymin, int ymax, control_par *cpar, int num_cam, target pix[])
{
    register int  i, j, m;
    int           n=0, n_wait=0, n_targets=0, sumg;
    int           numpix;
    int           thres, disco;
    int           xa,ya,xb,yb, x4[4],y4[4], xn,yn, nx, ny; 
    double            x, y;
    unsigned char *img0;

    register unsigned char    gv, gvref;

    targpix waitlist[2048];

    /* avoid many dereferences */
    int imx, imy;
    imx = cpar->imx;
    imy = cpar->imy;

    thres = targ_par->gvthres[num_cam];
    disco = targ_par->discont;
    
    // printf("thres %d disco %d \n", thres, disco);

    /* copy image to a temporary mask */
    img0 = (unsigned char *) calloc (imx*imy, 1);
    memcpy(img0, img, imx*imy);

    // printf("in segmentation.c memcpy succeded\n");

    /* Make sure the min/max coordinates don't cause us to access memory
       outside the image memory.
    */
    if (xmin <= 0) xmin = 1;
    if (ymin <= 0) ymin = 1;
    if (xmax >= imx) xmax = imx - 1;
    if (ymax >= imy) ymax = imy - 1;
    
    /*  thresholding and connectivity analysis in image  */
    for (i=ymin; i<ymax; i++)  for (j=xmin; j<xmax; j++)
    {
        gv = *(img0 + i*imx + j);
        if ( gv > thres)
            if (gv >= *(img0 + i*imx + j-1)
            &&  gv >= *(img0 + i*imx + j+1)
            &&  gv >= *(img0 + (i-1)*imx + j)
            &&  gv >= *(img0 + (i+1)*imx + j)
            &&  gv >= *(img0 + (i-1)*imx + j-1)
            &&  gv >= *(img0 + (i+1)*imx + j-1)
            &&  gv >= *(img0 + (i-1)*imx + j+1)
            &&  gv >= *(img0 + (i+1)*imx + j+1) )
      /* => local maximum, 'peak' */
          {
            yn=i;  xn=j;
            sumg = gv;  *(img0 + i*imx + j) = 0;
            xa = xn;  xb = xn;  ya = yn;  yb = yn;
            gv -= thres;
            x = (xn) * gv;
            y = yn * gv;
            numpix = 1;
            waitlist[0][0] = j;  waitlist[0][1] = i;  n_wait = 1;


            while (n_wait > 0) {
                gvref = *(img + imx*(waitlist[0][1]) + (waitlist[0][0]));

                x4[0] = waitlist[0][0] - 1;  y4[0] = waitlist[0][1];
                x4[1] = waitlist[0][0] + 1;  y4[1] = waitlist[0][1];
                x4[2] = waitlist[0][0];  y4[2] = waitlist[0][1] - 1;
                x4[3] = waitlist[0][0];  y4[3] = waitlist[0][1] + 1;


                for (n=0; n<4; n++) {
                    xn = x4[n];  yn = y4[n];
                    if (!(xn < xmax) || !(yn < ymax)) continue;
                    gv = *(img0 + imx*yn + xn);

                    /* conditions for threshold, discontinuity, image borders */
                    /* and peak fitting */
                    if (   (gv > thres)
                       && (xn > xmin - 1) && (xn < xmax + 1) 
                       && (yn > ymin - 1) && (yn < ymax + 1)
                       && (gv <= gvref+disco)
                       && (gvref + disco >= *(img + imx*(yn-1) + xn))
                       && (gvref + disco >= *(img + imx*(yn+1) + xn))
                       && (gvref + disco >= *(img + imx*yn + (xn-1)))
                       && (gvref + disco >= *(img + imx*yn + (xn+1)))  )
                      {
                        sumg += gv;  *(img0 + imx*yn + xn) = 0;
                        if (xn < xa) 
                            xa = xn;
                        if (xn > xb)
                            xb = xn;
                        if (yn < ya)
                            ya = yn;
                        if (yn > yb)
                            yb = yn;
                        waitlist[n_wait][0] = xn;   waitlist[n_wait][1] = yn;
                        
                        /* Coordinates are weighted by grey value, normed 
                           later. */
                        x += (xn) * (gv - thres);
                        y += yn * (gv - thres);
                        
                        numpix++;
                        n_wait++;
                      }
                }

                n_wait--;
                for (m=0; m<n_wait; m++) {
                    waitlist[m][0] = waitlist[m+1][0];
                    waitlist[m][1] = waitlist[m+1][1];
                }
                waitlist[n_wait][0] = 0;  waitlist[n_wait][1] = 0;

            }   /*  end of while-loop  */

            /* check whether target touches image borders */
            if (xa == (xmin - 1) || ya == (ymin - 1) || 
                xb == (xmax + 1)|| yb == (ymax + 1)) 
            {
                continue;
            }

            /* get targets extensions in x and y */
            nx = xb - xa + 1;  
            ny = yb - ya + 1;

            if (   (numpix >= targ_par->nnmin) && (numpix <= targ_par->nnmax)
               && (nx >= targ_par->nxmin) && (nx <= targ_par->nxmax)
               && (ny >= targ_par->nymin) && (ny <= targ_par->nymax)
               && (sumg > targ_par->sumg_min) )
              {
                    pix[n_targets].n = numpix;
                    pix[n_targets].nx = nx;
                    pix[n_targets].ny = ny;
                    pix[n_targets].sumg = sumg;
                    sumg -= (numpix*thres);

                    /* finish the grey-value weighting: */
                    x /= sumg;  x += 0.5;   
                    y /= sumg;  y += 0.5;
                    
                    pix[n_targets].x = x;
                    pix[n_targets].y = y;
                    pix[n_targets].tnr = CORRES_NONE;
                    pix[n_targets].pnr = n_targets;
                    n_targets++;
                    
                    xn = x;  
                    yn = y;
                    // printf("%d %d %d \n",n_targets, xn, yn);
              }
          } /*  end of if-loop  */
    }
    free(img0);
    /* protect pix from zero memory */
    if (n_targets < 1){
        pix[0].n = 1;
        pix[0].nx = 1;
        pix[0].ny = 1;
        pix[0].sumg = 1;
        pix[0].x = 1;
        pix[0].y = 1;
        pix[0].tnr = CORRES_NONE;
        pix[0].pnr = 1;
        n_targets++;
    }
    // printf("final n_targets %d\n",n_targets);
    return(n_targets);
}



/* 
  peak fitting technique for particle coordinate determination  
  labeling with discontinuity, reunification with distance and profile criteria
  based on the two-pass component labeling method. first pass marks the points above
  the threshold and searches for local maximum. 
    
Arguments:
    unsigned char   *img - image
    target_par *targ_par - target dots parameters: size limits, sum of grey values
        intensity thresholds, etc.
    int xmin, xmax, ymin, ymax - search area, limits of the field of view.
    control_par *cpar - control parameters.
    int num_cam - number of the camera (0... cal_par->num_cams), used for 
        selecting threshold from the overall target parameters.
    
Output:
    target pix[] - array of found dots of centroids, sum of grey values and 
        an identifier (number of the dot)

Returns:
    number of targets found.
*/

int peak_fit (unsigned char *img, target_par *targ_par, int xmin, int xmax, 
int ymin, int ymax, control_par *cpar, int num_cam, target pix[]){
    int imx, imy; /* save dereferencing of same in cpar */
    int           n_peaks=0;        /* # of peaks detected */
    int           n_wait;               /* size of waitlist for connectivity */
    int           x8[4], y8[4];         /* neighbours for connectivity */
    int           p2;               /* considered point number */
    int           thres, disco;
    /* parameters for target acceptance */
    int           pnr, sumg, xn, yn;  /* collecting variables for center of gravity */
    int           n_target=0;         /* # of targets detected */
    int           intx1, inty1;       /* pixels for profile test and crosses */
    int           unify;              /* flag for unification of targets */
    int           unified=0;          /* # of unified targets */
    int       non_unified=0;          /* # of tested, but not unified targets */
    register int  i, j, k, l, m, n;     /* loop variables */
    unsigned char gv, gvref;      /* current and reference greyvalue */
    unsigned char gv1, gv2;           /* greyvalues for profile test */
    double        x1,x2,y1,y2,s12;    /* values for profile test */
    short         *label_img;           /* target number labeling */
    /* the following were not assigned, probably from globals */
    int nmax = 1024; 
    peak          *peaks, *ptr_peak;    /* detected peaks */
    targpix       waitlist[2048];     /* pix to be tested for connectivity */

    imx = cpar->imx;
    imy = cpar->imy;

    thres = targ_par->gvthres[num_cam];
    disco = targ_par->discont;

    /* allocate memory */
    label_img = (short *) calloc (imx*imy, sizeof(short));
    peaks = (peak *) calloc (4*nmax, sizeof(peak));
    ptr_peak = peaks;


    /* connectivity analysis with peak search and discontinuity criterion */

    for (i=ymin; i<ymax-1; i++) for (j=xmin; j<xmax; j++) {
          n = i*imx + j;

          /* compare with threshold */
          gv = *(img + n);   if (gv <= thres)   continue;

          /* skip already labeled pixel */
          if (*(label_img + n) != 0)    continue;

          /* check, wether pixel is a local maximum */
          if ( gv >= *(img + n-1)
           &&   gv >= *(img + n+1)
           &&   gv >= *(img + n-imx)
           &&   gv >= *(img + n+imx)
           &&   gv >= *(img + n-imx-1)
           &&   gv >= *(img + n+imx-1)
           &&   gv >= *(img + n-imx+1)
           &&   gv >= *(img + n+imx+1) )
            {
              /* label peak in label_img, initialize peak */
              n_peaks++;
              *(label_img + n) = n_peaks;
              ptr_peak->pos = n;
              ptr_peak->status = 1;
              ptr_peak->xmin = j;   ptr_peak->xmax = j;
              ptr_peak->ymin = i;   ptr_peak->ymax = i;
              ptr_peak->unr = 0;
              ptr_peak->n = 0;
              ptr_peak->sumg = 0;
              ptr_peak->x = 0;
              ptr_peak->y = 0;
              ptr_peak->n_touch = 0;
              for (k=0; k<4; k++)   ptr_peak->touch[k] = 0;
              ptr_peak++;

              waitlist[0][0] = j;  waitlist[0][1] = i;  n_wait = 1;

              while (n_wait > 0) {
                  gvref = *(img + imx*(waitlist[0][1]) + (waitlist[0][0]));

                  x8[0] = waitlist[0][0] - 1;   y8[0] = waitlist[0][1];
                  x8[1] = waitlist[0][0] + 1;   y8[1] = waitlist[0][1];
                  x8[2] = waitlist[0][0];       y8[2] = waitlist[0][1] - 1;
                  x8[3] = waitlist[0][0];       y8[3] = waitlist[0][1] + 1;
                  
                  for (k=0; k<4; k++) {
                      yn = y8[k];
                      xn = x8[k];
                      
                      if (xn<0 || xn>imx || yn<0 || yn>imy) 
                        continue;
                      
                      n = imx*yn + xn;
                      if (*(label_img + n) != 0)
                        continue;
                      
                      gv = *(img + n);

                      /* conditions for threshold, discontinuity, image borders */
                      /* and peak fitting */
                      if (   (gv > thres)
                         && (xn>=xmin)&&(xn<xmax) && (yn>=ymin)&&(yn<ymax-1)
                         && (gv <= gvref+disco)
                         && (gvref + disco >= *(img + imx*(yn-1) + xn))
                         && (gvref + disco >= *(img + imx*(yn+1) + xn))
                         && (gvref + disco >= *(img + imx*yn + (xn-1)))
                         && (gvref + disco >= *(img + imx*yn + (xn+1)))
                         )
                        {
                          *(label_img + imx*yn + xn) = n_peaks;

                          waitlist[n_wait][0] = xn; waitlist[n_wait][1] = yn;
                          n_wait++;
                        }
                  }

                  n_wait--;
                  for (m=0; m<n_wait; m++) {  
                    waitlist[m][0] = waitlist[m+1][0];
                    waitlist[m][1] = waitlist[m+1][1];
                  }
                  waitlist[n_wait][0] = 0;  waitlist[n_wait][1] = 0;
              }   /*  end of while-loop  */
            }
    }

    /* 2.:    process label image */
    /*        (collect data for center of gravity, shape and brightness parameters) */
    /*        get touch events */

    for (i=ymin; i<ymax; i++)  for (j=xmin; j<xmax; j++) {
          n = i*imx + j;

          if (*(label_img+n) > 0) {
              /* process pixel */
              pnr = *(label_img+n);
              gv = *(img+n);
              ptr_peak = peaks + pnr - 1;
              ptr_peak->n++;
              ptr_peak->sumg += gv;
              ptr_peak->x += (j * gv);
              ptr_peak->y += (i * gv);
              
              if (j < ptr_peak->xmin)   ptr_peak->xmin = j;
              if (j > ptr_peak->xmax)   ptr_peak->xmax = j;
              if (i < ptr_peak->ymin)   ptr_peak->ymin = i;
              if (i > ptr_peak->ymax)   ptr_peak->ymax = i;


              /* get touch events */

              if (i>0 && j>1)           check_touch (ptr_peak, pnr, *(label_img+n-imx-1));
              if (i>0)                  check_touch (ptr_peak, pnr, *(label_img+n-imx));
              if (i>0 && j<imy-1)       check_touch (ptr_peak, pnr, *(label_img+n-imx+1));

              if (j>0)                  check_touch (ptr_peak, pnr, *(label_img+n-1));
              if (j<imy-1)              check_touch (ptr_peak, pnr, *(label_img+n+1));

              if (i<imx-1 && j>0)       check_touch (ptr_peak, pnr, *(label_img+n+imx-1));
              if (i<imx-1)              check_touch (ptr_peak, pnr, *(label_img+n+imx));
              if (i<imx-1 && j<imy-1)   check_touch (ptr_peak, pnr, *(label_img+n+imx+1));
          }
    }

    /* 3.:    reunification test: profile and distance */

    for (i=0; i<n_peaks; i++) {
      if (peaks[i].n_touch == 0)    continue;       /* no touching targets */
      if (peaks[i].unr != 0)        continue;       /* target already unified */

      /* profile criterion */
      /* point 1 */
      x1 = peaks[i].x / peaks[i].sumg;
      y1 = peaks[i].y / peaks[i].sumg;
      gv1 = *(img + peaks[i].pos);

      /* consider all touching points */
      for (j=0; j<peaks[i].n_touch; j++) {
          p2 = peaks[i].touch[j] - 1;

          if (p2 >= n_peaks) continue;  /* workaround memory overwrite problem */
          if ( p2 <0) continue;         /*  workaround memory overwrite problem */
          if (peaks[p2].unr != 0)       continue; /* target already unified */

          /* point 2 */
          x2 = peaks[p2].x / peaks[p2].sumg;
          y2 = peaks[p2].y / peaks[p2].sumg;

          gv2 = *(img + peaks[p2].pos);

          s12 = sqrt ((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1));

          /* consider profile dot for dot */
          /* if any points is by more than disco below profile, do not unify */
          if (s12 < 2.0)    unify = 1;
          else for (unify=1, l=1; l<s12; l++) {
              intx1 = (int) (x1 + l * (x2 - x1) / s12);
              inty1 = (int) (y1 + l * (y2 - y1) / s12);
              gv = *(img + inty1*imx + intx1) + disco;
              if (gv < (gv1+l*(gv2-gv1)/s12) || gv<gv1 || gv<gv2)   unify = 0;
              if (unify == 0)   break;
          }
          if (unify == 0) {
              non_unified++;
              continue;
          }

          /* otherwise unify targets */
          unified++;
          peaks[i].unr = p2;
          peaks[p2].x += peaks[i].x;
          peaks[p2].y += peaks[i].y;
          peaks[p2].sumg += peaks[i].sumg;
          peaks[p2].n += peaks[i].n;
          if (peaks[i].xmin < peaks[p2].xmin)   peaks[p2].xmin = peaks[i].xmin;
          if (peaks[i].ymin < peaks[p2].ymin)   peaks[p2].ymin = peaks[i].ymin;
          if (peaks[i].xmax > peaks[p2].xmax)   peaks[p2].xmax = peaks[i].xmax;
          if (peaks[i].ymax > peaks[p2].ymax)   peaks[p2].ymax = peaks[i].ymax;
      }
    }

      /* 4.:    process targets */

    for (i = 0; i < n_peaks; i++) {
      /* check whether target touches image borders */
      if (peaks[i].xmin == xmin  &&  (xmax-xmin) > 32)  continue;
      if (peaks[i].ymin == ymin  &&  (xmax-xmin) > 32)  continue;
      if (peaks[i].xmax == xmax-1  &&  (xmax-xmin) > 32)    continue;
      if (peaks[i].ymax == ymax-1  &&  (xmax-xmin) > 32)    continue;

      if (   peaks[i].unr == 0
         && peaks[i].sumg > targ_par->sumg_min
         && (peaks[i].xmax - peaks[i].xmin + 1) >= targ_par->nxmin
         && (peaks[i].ymax - peaks[i].ymin + 1) >= targ_par->nymin
         && (peaks[i].xmax - peaks[i].xmin) < targ_par->nxmax
         && (peaks[i].ymax - peaks[i].ymin) < targ_par->nymax
         && peaks[i].n >= targ_par->nnmin
         && peaks[i].n <= targ_par->nnmax)
        {
          sumg = peaks[i].sumg;

          /* target coordinates */
          pix[n_target].x = 0.5 + peaks[i].x / sumg;
          pix[n_target].y = 0.5 + peaks[i].y / sumg;


          /* target shape parameters */
          pix[n_target].sumg = sumg;
          pix[n_target].n = peaks[i].n;
          pix[n_target].nx = peaks[i].xmax - peaks[i].xmin + 1;
          pix[n_target].ny = peaks[i].ymax - peaks[i].ymin + 1;
          pix[n_target].tnr = CORRES_NONE;
          pix[n_target].pnr = n_target;
          n_target++;
        }
    }
    
    free (label_img);
    free (peaks);

    return (n_target);
}

/* check_touch () checks wether p1, p2 are already marked as touching 
    and mark them otherwise 
Arguments:
    peak *tpeak     array of peaks 
    int p1, p2      pointers to two dots that might be touching
Output:
    tpeak->n_touch will hold a pointer or a flag of touch events
*/
void check_touch (peak *tpeak, int p1, int p2){
  int   m, done;


  if (p2 == 0)  return;     /* p2 not labeled */
  if (p2 == p1) return;     /* p2 belongs to p1 */


  /* check wether p1, p2 are already marked as touching */
  for (done=0, m=0; m<tpeak->n_touch; m++)
    {
      if (tpeak->touch[m] == p2)    done = 1;
    }

  /* mark touch event */
  if (done == 0)
    {
      tpeak->touch[tpeak->n_touch] = p2;
      tpeak->n_touch++;
      /* don't allow for more than 4 touchs */
      if (tpeak->n_touch > 3)   tpeak->n_touch = 3;
    }
}

