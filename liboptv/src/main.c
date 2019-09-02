/* main.c 
I need this file to start preparing some structure in my head. Alex
*/

#define MAXTARGETS 2048
#define BUFFER_LENGTH 4 // we do something very simple and studpid here 
#include "main.h"

int main( int argc, const char* argv[] )
{
    // initialize variables

    int i, ntargets;
    // DIR *dirp;
    // struct dirent *dp;
    char file_name[256];
    int step;
    unsigned char *img, *img_hp;
    target pix[MAXTARGETS], targ_t[MAXTARGETS];
    coord_2d **corrected;
    int match_counts[4];
    frame frm;


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
    read_all_calibration(calib, cpar->num_cams);
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
        // prepare frame for correspondences, see below below image segmentation
        frame_init(&frm, run->cpar->num_cams, MAXTARGETS);

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
            free(img);
            free(img_hp);
            // c. segmentation
            // detection
            //ntargets = peak_fit(img_hp, targ_read, 0, run->cpar->imx, 0, run->cpar->imy, run->cpar, 1, pix);
            ntargets = targ_rec(img_hp, targ_read, 0, run->cpar->imx, 0, run->cpar->imy, run->cpar, 1, pix);
            // here we fill the frame with the targets for the next step - correspondence
            frm.num_targets[i] = ntargets;
            frm.targets[i] = targ_t;
       } // inner loop is camera
       
       correspondences(&frm, corrected, run->vpar, run->cpar, calib, match_counts);

    } // external loop is frames



        // image segmentation of this step and 4 steps forward
        for (i = 0; i<BUFFER_LENGTH; i++){
            // per camera:
            for (cam = 0; cam < NUM_CAMS; cam++){
                // highpass filter this frame 
                // detect
                // fill buffer of targets
            } 
         
            // stereomatching
            // find correspondences in this frame
            // fill buffer of correspondences


            // 3d triangulation of this frame
            // fill buffer of 3d positions 
        }
        // apparently all the buffers are full and we can just start tracking
        // fill buffer with path_info (path is a trajectory)
        // plot and move to the next chunk, jump 4 frames forward and do it again
    }
}