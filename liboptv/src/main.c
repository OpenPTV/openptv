/* main.c 
I need this file to start preparing some structure in my head. Alex
*/

#define MAXTARGETS 2048
#define BUFFER_LENGTH 4 // we do something very simple and studpid here
#include "main.h"

// These functions are part of the a test suite, see under /tests

void read_all_calibration(Calibration *calib[4], control_par *cpar)
{
    char ori_tmpl[] = "cal/cam%d.tif.ori";
    char added_name[] = "cal/cam1.tif.addpar";
    char ori_name[40];
    int cam;

    for (cam = 0; cam < cpar->num_cams; cam++)
    {
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
    coord_2d **corrected, **tmp;
    int cam, part;

    corrected = (coord_2d **)malloc(cpar->num_cams * sizeof(coord_2d *));
    tmp = (coord_2d **)malloc(cpar->num_cams * sizeof(coord_2d *));
    for (cam = 0; cam < cpar->num_cams; cam++)
    {
        corrected[cam] = (coord_2d *)malloc(
            frm->num_targets[cam] * sizeof(coord_2d));
        tmp[cam] = (coord_2d *)malloc(frm->num_targets[cam] * sizeof(coord_2d));
        if (corrected[cam] == NULL)
        {
            /* roll back allocations and fail */
            for (cam -= 1; cam >= 0; cam--)
                free(corrected[cam]);
            free(corrected);
            return NULL;
        }

        for (part = 0; part < frm->num_targets[cam]; part++)
        {
            pixel_to_metric(&tmp[cam][part].x, &tmp[cam][part].y,
                            frm->targets[cam][part].x, frm->targets[cam][part].y,
                            cpar);

            dist_to_flat(tmp[cam][part].x, tmp[cam][part].y,
                         calib[cam], &corrected[cam][part].x, &corrected[cam][part].y,
                         tol);
            corrected[cam][part].pnr = frm->targets[cam][part].pnr;
        }

        /* This is expected by find_candidate() */
        quicksort_coord2d_x(corrected[cam], frm->num_targets[cam]);
    }
    return corrected;
}

// int main(int argc, const char *argv[])
int main()
{
    // initialize variables

    int i, cam, ntargets, step, lstep, geo_id;
    coord_2d **corrected;
    int match_counts[4];
    n_tupel *corresp_buf;
    tracking_run *run;
    vec3d res;
    vec2d targ[4],tmp;
    framebuf *frm; 

    // temporary variables will be replaced per particle 
    corres t_corres = { 3, {96, 66, 26, 26} };
    P t_path = {
        .x = {45.219, -20.269, 25.946},
        .prev = -1,
        .next = -2,
        .prio = 4,
        .finaldecis = 1000000.0,
        .inlist = 0.
    };

    // read parameters from the working directory
    // for simplicity all names are default and hard coded (sorry)

    // 1. process inputs: directory, first frame, last frame

    // printf("This program was called with \"%s\".\n", argv[0]);

    // argc = 4;

    // argv[1] = '../tests/testing_fodder/test_cavity/';
    // argv[2] = 10001;
    // argv[3] = 10004;

    // if (argc != 2 && argc != 4)
    // {
    //     printf("Wrong number of inputs, expecting: \n");
    //     printf(" ./main ../tests/testing_fodder/test_cavity/ 10001 10004 \n");
    //     return 0;
    // }

    // argv[1] = '../tests/testing_fodder/test_cavity/';
    // argv[2] = 10001;
    // argv[3] = 10004;

    // change directory to the user-supplied working folder
    // chdir(argv[1]);

    // chdir("../tests/testing_fodder/test_cavity/");
    chdir("/Users/alex/Documents/OpenPTV/test_cavity");
    // chdir("/Users/alex/Documents/OpenPTV/one-dot-example/working_folder");

    // printf("changed directory to %s\n", argv[1]);

    // 2. read parameters and calibrations
    Calibration *calib[4]; // sorry only for 4 cameras now

    control_par *cpar = read_control_par("parameters/ptv.par");
    read_all_calibration(calib, cpar);
    free_control_par(cpar);

    run = tr_new_legacy("parameters/sequence.par",
                        "parameters/track.par", "parameters/criteria.par",
                        "parameters/ptv.par", calib);


    // if (argc == 4)
    // {
    //     run->seq_par->first = atoi(argv[2]);
    //     run->seq_par->last = atoi(argv[3]);
    // }
    run->seq_par->first = 10000;
    run->seq_par->last = 10004;

    // printf("from frame %d to frame %d \n", run->seq_par->first, run->seq_par->last);

    // for each camera and for each time step the images are processed
    for (step = run->seq_par->first; step < run->seq_par->last + 1; step++)
    {
        lstep = step - run->seq_par->first; // local step
         
        for (cam = 0; cam < run->cpar->num_cams; cam++)
        {
            // we decided to focus camust on the _targets, so we will read them from the
            // test directory test_cavity

            run->fb->buf[lstep]->num_targets[cam] = read_targets(
                run->fb->buf[lstep]->targets[cam], run->fb->target_file_base[cam], step);

            quicksort_target_y(run->fb->buf[lstep]->targets[cam], run->fb->buf[lstep]->num_targets[cam]);

            for (i = 0; i < run->fb->buf[lstep]->num_targets[cam]; i++)
               run->fb->buf[lstep]->targets[cam][i].pnr = i;


        } // inner loop is per camera
        corrected = correct_frame(run->fb->buf[lstep], run->cal, run->cpar, 0.00001);
        corresp_buf = correspondences(run->fb->buf[lstep], corrected, run->vpar, run->cpar, run->cal, match_counts);

        run->fb->buf[lstep]->num_parts = match_counts[0];

        // first we need to create 3d points after correspondences and fill it into the buffer
        // use point_position and loop through the num_parts
        // probably feed it directly into the buffer

        // so we split into two parts:
        // first i copy the code from correspondences.pyx
        // and create the same types of arrays in C
        // then we will convert those to 3D using similar approach to what is now in Python


        // shortcut
        int p[4];
        double x,y;
        float skew_dist; 

        //printf("looks like rt_is.%d\n",step);

        for (i=0; i<run->fb->buf[lstep]->num_parts; i++) {
            for (cam = 0; cam < run->cpar->num_cams; cam++) {
                if (corresp_buf[i].p[cam] > -1){  
                    p[cam] = corrected[cam][corresp_buf[i].p[cam]].pnr;
                    x = run->fb->buf[lstep]->targets[cam][p[cam]].x;
                    y = run->fb->buf[lstep]->targets[cam][p[cam]].y;

                    pixel_to_metric(&x, &y, x, y, run->cpar);
                    dist_to_flat(x, y,run->cal[cam], &x, &y,0.00001);

                    targ[cam][0] = x;  
                    targ[cam][1] = y;       
                }else{
                    targ[cam][0] = 1e-10;
                    targ[cam][1] = 1e-10;
                }

                t_corres.p[cam] = p[cam];
            }

                t_corres.nr = i+1; // identical to what I see in rt_is.10004 file
                run->fb->buf[lstep]->correspond[i] = t_corres;
                




                skew_dist = point_position(targ, run->cpar->num_cams, run->cpar->mm, run->cal, res);

                // printf("%d \t ",run->fb->buf[lstep]->correspond[i].nr);
                // printf("%f,\t %f,\t %f\t ",res[0],res[1],res[2]);
                // for (cam=0;cam<run->cpar->num_cams; cam++) {
                //     printf("%d\t ",run->fb->buf[lstep]->correspond[i].p[cam]);
                // }
                // printf("\n");


                t_path.x[0] = res[0];
                t_path.x[1] = res[1];
                t_path.x[2] = res[2];

                run->fb->buf[lstep]->path_info[i] = t_path;
            //}
        }

    } // external loop is through frames

    run->tpar->add = 0; 
    // we do not need to read frames - it's all in memory now
    // track_forward_start(run); 

    // for (step = run->seq_par->first + 1 ; step < run->seq_par->last; step++)
    // // for (step = run->seq_par->first + 1; step < run->seq_par->last; step++)
    // {
    //     trackcorr_c_loop(run, step);
    // }
    // trackcorr_c_finish(run, run->seq_par->last);

    trackcorr_c_loop(run, run->seq_par->first);

    // probably here we need to send something to plot it
    // or store in some memory for the next chunk?
    // basically we have a human in the loop here - his/her brain
    // will simply follow the velocity values in time like a movie
    // and we will store it to binary files. Later if someone wants to do
    // tracking, our simmple solution is not good enough. we kind of doing 3D-PIV here
    // of 4 frames and show the vectors. The quasi-vectors are not really connected. if we
    // will create nice animation - then the user will build tracamectories him/herself.

    for (cam -= 1; cam >= 0; cam--)
        free(corrected[cam]);

    free(corrected);
    free(corresp_buf);
    free(run->vpar);
    free_control_par(run->cpar);

    return 0;

} // should be end of main now