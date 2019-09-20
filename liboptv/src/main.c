/* main.c 
I need this file to start preparing some structure in my head. Alex
*/

#define MAXTARGETS 2048
#define BUFFER_LENGTH 4 // we do something very simple and studpid here 
#include "main.h"

// These functions are part of the a test suite, see under /tests 

void read_all_calibration(Calibration *calib[4], control_par *cpar) {
    char ori_tmpl[] = "testing_fodder/cal/sym_cam%d.tif.ori";
    char added_name[] = "testing_fodder/cal/cam1.tif.addpar";
    char ori_name[40];
    int cam;
    
    for (cam = 0; cam < cpar->num_cams; cam++) {
        sprintf(ori_name, ori_tmpl, cam + 1);
        calib[cam] = read_calibration(ori_name, added_name, NULL);
    }
}

/*  correct_frame() performs the transition from pixel to metric to flat 
    coordinates and x-sorting as required by the correspondence code.
    
    Arguments:
    frame *frm - target information for all cameras.
    control_par *cpar - parameters of image size, pixel size etc.
    tol - tolerance parameter for iterative flattening phase, see 
        trafo.h:correct_brown_affine_exact().
*/
coord_2d **correct_frame(frame *frm, Calibration *calib[], control_par *cpar, 
    double tol) 
{
    coord_2d **corrected;
    int cam, part;
    
    corrected = (coord_2d **) malloc(cpar->num_cams * sizeof(coord_2d *));
    for (cam = 0; cam < cpar->num_cams; cam++) {
        corrected[cam] = (coord_2d *) malloc(
            frm->num_targets[cam] * sizeof(coord_2d));
        if (corrected[cam] == NULL) {
            /* roll back allocations and fail */
            for (cam -= 1; cam >= 0; cam--) free(corrected[cam]);
            free(corrected);
            return NULL;
        }
        
        for (part = 0; part < frm->num_targets[cam]; part++) {
            pixel_to_metric(&corrected[cam][part].x, &corrected[cam][part].y,
                          frm->targets[cam][part].x, frm->targets[cam][part].y,
                          cpar);
            
            dist_to_flat(corrected[cam][part].x, corrected[cam][part].y,
                calib[cam], &corrected[cam][part].x, &corrected[cam][part].y,
                tol);
        
            corrected[cam][part].pnr = frm->targets[cam][part].pnr;
        }
        
        /* This is expected by find_candidate() */
        quicksort_coord2d_x(corrected[cam], frm->num_targets[cam]);
    }
    return corrected;
}


int main( int argc, const char* argv[] )
{
    // initialize variables

    int i, ntargets;
    // DIR *dirp;
    // struct dirent *dp;
    char file_name[256];
    int step, cam;
    unsigned char *img, *img_hp;
    target pix[MAXTARGETS], targ_t[MAXTARGETS];
    coord_2d **corrected;
    int match_counts[4];
    n_tupel *con;
    


    // read parameters from the working directory
    // for simplicity all names are default and hard coded (sorry)

    
    
    // 1. process inputs: directory, first frame, last frame
  
  printf ("This program was called with \"%s\".\n",argv[0]);
  
    if (argc != 2 && argc != 4)
    {
        printf("Wrong number of inputs, expecting: \n");
        printf(" ./openptv test_cavity \n");
        printf(" or \n");
        printf(" ./openptv test_cavity 10000 10004 \n");
        return 0;
    }

    // change directory to the user-supplied working folder
    chdir(argv[1]);
    
    // 2. read parameters and calibrations
    Calibration *calib[4]; // sorry only for 4 cameras now
    
    control_par *cpar = read_control_par("parameters/ptv.par");
    read_all_calibration(calib, cpar);
    free_control_par(cpar);
    
    tracking_run *run = tr_new_legacy("parameters/sequence.par",
                                      "parameters/track.par", "parameters/criteria.par",
                                      "parameters/ptv.par", calib);

    if (argc == 4)
    {
        run->seq_par->first = atoi(argv[2]);
        run->seq_par->last = atoi(argv[3]);
    }
    printf("from frame %d to frame %d \n", run->seq_par->first, run->seq_par->last);
    
    target_par *targ_read = read_target_par("parameters/targ_rec.par");

    // initialize memory buffers

    // for (step = 0; step < N_FRAMES_IN_DIRECTORY-BUFFER_LENGTH-1; step+BUFFER_LENGTH){
    // MAIN LOOP - see below we will just give inputs of 10000 10004 as a very simple approach

            // for each camera and for each time step the images are processed
    for (step = run->seq_par->first; step < run->seq_par->last+1; step++) {
        for (i = 1; i<run->cpar->num_cams+1; i++) {
        
            // a. read image
            sprintf(file_name, "img/cam%d.%d", i, step);
            img = (unsigned char *) malloc(run->cpar->imx*run->cpar->imy* \
                                           sizeof(unsigned char));
            img_hp = (unsigned char *) malloc(run->cpar->imx*run->cpar->imy* \
                                              sizeof(unsigned char));
            imread(img, file_name);
            // b. highpass
            if (run->cpar->hp_flag)
            {
                prepare_image(img, img_hp, 1, 0, 0, run->cpar);
            } else {
                memcpy(img_hp, img, run->cpar->imx*run->cpar->imy);
            }
            // c. segmentation
            // detection
            //ntargets = peak_fit(img_hp, targ_read, 0, run->cpar->imx, 0, run->cpar->imy, run->cpar, 1, pix);
            run->fb->buf[step]->num_targets[i] = targ_rec(img_hp, targ_read, 0, run->cpar->imx, 0, run->cpar->imy, run->cpar, 1, run->fb->buf[step]->targets[i]);
            
            // release temporary memory
            free(img);
            free(img_hp);
       } // inner loop is camera
        coord_2d **corrected = correct_frame(run->fb->buf[step], calib, cpar, 0.0001);
        con = correspondences(run->fb->buf[step], corrected, run->vpar, run->cpar, calib, match_counts);
        run->fb->buf[step]->num_parts = match_counts[3]; // sum of all matches? 
       // so here is missing frame into run->frame ?
       // WORK HERE 

    } // external loop is through frames

    // ok, theoretically we have now a buffer full of stuff from 4 frames
    // it's a good buffer on which we can just track stuff
    // and then we need to jump to a next chunk, remove all and start over.
    // the missing part is how to "chain the chunks" or make a smart use of
    // memory and buffers, it's beyond me now


    run->tpar->add = 0;
    track_forward_start(run);

    for (step = run->seq_par->first + 1; step < run->seq_par->last; step++) {
        trackcorr_c_loop(run, step);
    }
    trackcorr_c_finish(run, run->seq_par->last);

    // probably here we need to send something to plot it 
    // or store in some memory for the next chunk? 
    // basically we have a human in the loop here - his/her brain
    // will simply follow the velocity values in time like a movie
    // and we will store it to binary files. Later if someone wants to do
    // tracking, our simmple solution is not good enough. we kind of doing 3D-PIV here
    // of 4 frames and show the vectors. The quasi-vectors are not really connected. if we
    // will create nice animation - then the user will build trajectories him/herself. 


    /* Clean up */
    for (i = 1; i<run->cpar->num_cams+1; i++) 
        free(corrected[i]);
    // deallocate_adjacency_lists(correspond* lists[4][4], cpar->num_cams);
    // deallocate_target_usage_marks(tusage, cpar->num_cams);
    free(con);
    free(run->vpar);
    free_control_par(run->cpar);

    return 0;


} // should be end of main now 