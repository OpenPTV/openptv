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
    coord_2d **corrected;
    int cam, part;

    corrected = (coord_2d **)malloc(cpar->num_cams * sizeof(coord_2d *));
    for (cam = 0; cam < cpar->num_cams; cam++)
    {
        corrected[cam] = (coord_2d *)malloc(
            frm->num_targets[cam] * sizeof(coord_2d));
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

int main(int argc, const char *argv[])
{
    // initialize variables

    int i, ntargets;
    // DIR *dirp;
    // struct dirent *dp;
    char file_name[256];
    int step, cam, geo_id;
    target pix[MAXTARGETS], targ_t[MAXTARGETS], targ;
    coord_2d **corrected, **sorted_pos, **flat;
    int **sorted_corresp;
    int match_counts[4];
    n_tupel *corresp_buf;
    tracking_run *run;
    vec3d res;

    // read parameters from the working directory
    // for simplicity all names are default and hard coded (sorry)

    // 1. process inputs: directory, first frame, last frame

    printf("This program was called with \"%s\".\n", argv[0]);

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

    printf("changed directory to %s\n", argv[1]);

    // 2. read parameters and calibrations
    Calibration *calib[4]; // sorry only for 4 cameras now

    control_par *cpar = read_control_par("parameters/ptv.par");
    read_all_calibration(calib, cpar);
    free_control_par(cpar);
    printf("read calibrations\n");

    run = tr_new_legacy("parameters/sequence.par",
                        "parameters/track.par", "parameters/criteria.par",
                        "parameters/ptv.par", calib);

    if (argc == 4)
    {
        run->seq_par->first = atoi(argv[2]);
        run->seq_par->last = atoi(argv[3]);
    }
    printf("from frame %d to frame %d \n", run->seq_par->first, run->seq_par->last);

    // target_par *targ_read = read_target_par("parameters/targ_rec.par");

    // initialize memory buffers

    // for (step = 0; step < N_FRAMES_IN_DIRECTORY-BUFFER_LENGTH-1; step+BUFFER_LENGTH){
    // MAIN LOOP - see below we will just give inputs of 10000 10004 as a very simple approach

    // for each camera and for each time step the images are processed
    for (step = run->seq_par->first; step < run->seq_par->last + 1; step++)
    {
        for (cam = 0; cam < run->cpar->num_cams; cam++)
        {
            // we decided to focus just on the _targets, so we will read them from the
            // test directory test_cavity
            printf("reading targets from %s%d\n", run->fb->target_file_base[cam], step);

            run->fb->buf[step - run->seq_par->first]->num_targets[cam] = read_targets(
                run->fb->buf[step - run->seq_par->first]->targets[cam], run->fb->target_file_base[cam], step);

            quicksort_target_y(run->fb->buf[step - run->seq_par->first]->targets[cam], run->fb->buf[step - run->seq_par->first]->num_targets[cam]);

            for (i = 0; i < run->fb->buf[step - run->seq_par->first]->num_targets[cam]; i++)
                run->fb->buf[step - run->seq_par->first]->targets[cam][i].pnr = i;

            // debugging purposes print the status of targets - see below another print.
            printf("%d targets and the first is %f,%f \n ",
                   run->fb->buf[step - run->seq_par->first]->num_targets[cam],
                   run->fb->buf[step - run->seq_par->first]->targets[cam][0].x,
                   run->fb->buf[step - run->seq_par->first]->targets[cam][0].y);

        } // inner loop is per camera
        corrected = correct_frame(run->fb->buf[step - run->seq_par->first], calib, cpar, 0.0001);
        corresp_buf = correspondences(run->fb->buf[step - run->seq_par->first], corrected, run->vpar, run->cpar, calib, match_counts);
        run->fb->buf[step - run->seq_par->first]->num_parts = match_counts[run->cpar->num_cams - 1];
        printf("number of matched points is %d \n ", run->fb->buf[step - run->seq_par->first]->num_parts);

        // first we need to create 3d points after correspondences and fill it into the buffer
        // use point_position and loop through the num_parts
        // probably feed it directly into the buffer

        // so we split into two parts:
        // first i copy the code from correspondences.pyx
        // and create the same types of arrays in C
        // then we will convert those to 3D using similar approach to what is now in Python


        // shortcut
        int num_parts = run->fb->buf[step - run->seq_par->first]->num_parts; 

        // return structures
        sorted_pos = (coord_2d **)malloc(run->cpar->num_cams * sizeof(coord_2d *));
        sorted_corresp = (int **)malloc(run->cpar->num_cams * sizeof(int *));

        for (cam = 0; cam < run->cpar->num_cams; cam++)
        {
            sorted_pos[cam] = (coord_2d *)malloc(num_parts * sizeof(coord_2d));
            if (sorted_pos[cam] == NULL)
            {
                /* roll back allocations and fail */
                for (cam -= 1; cam >= 0; cam--)
                    free(sorted_pos[cam]);
                free(sorted_pos);
                return NULL;
            }
                        
            sorted_corresp[cam]  = (int *)malloc(num_parts * sizeof(int));

            if (sorted_corresp[cam] == NULL)
            {
                /* roll back allocations and fail */
                for (cam -= 1; cam >= 0; cam--)
                    free(sorted_corresp[cam]);
                free(sorted_corresp);
                return NULL;
            }
        }

        int last_count = 0;

        for (int clique_type = 0; clique_type < run->cpar->num_cams; clique_type++)
        {
            num_points = match_counts[4 - run->cpar->num_cams + clique_type] // for 1-4 cameras
                
            for (cam = 0; cam < run->cpar->num_cams; cam++) 
            {
                for ( pt = last_count; pt < num_points; pt++)
                {
                    geo_id = corresp_buf[pt + last_count].p[cam];
                    if (geo_id < 0)
                        continue;

                    p1 = corrected[cam][geo_id].pnr;
                    sorted_corresp[cam][pt] = p1;

                    if (p1 > -1) 
                    {
                        targ = run->fb->buf[step - run->seq_par->first]->targets[cam][p1];
                        sorted_pos[cam][pt][0] = targ.x;
                        sorted_pos[cam][pt][1] = targ.y;
                    }
                } // points
            } // cam 

            last_count += num_points;
        } // 


    // sort corrected by the sorted_corresp:
    // prepare the memory for

    flat = (coord_2d **)malloc(cpar->num_cams * sizeof(coord_2d *));
    for (cam = 0; cam < run->cpar->num_cams; cam++)
    {
        flat[cam] = (coord_2d *)malloc(
            run->fb->buf[step - run->seq_par->first]->num_targets[cam] * sizeof(coord_2d));
        if (flat[cam] == NULL)
        {
            /* roll back allocations and fail */
            for (cam -= 1; cam >= 0; cam--)
                free(flat[cam]);
            free(flat);
            return NULL;
        }
    }

    
    for (i=0;i<num_points;i++){


    }
    flat = np.array([corrected[i].get_by_pnrs(sorted_corresp[i]) \
                     for i in range(len(cals))])
    pos, rcm = point_positions(
        flat.transpose(1,0,2), cpar, cals, vpar)


        vec2d targs_plain[4]; // x,y coordinates in image space for 4 Cameras

    skew_dist = point_position(targs_plain, num_cams, &media_par, calib, res);

        for
            pt in range(num_targets) : 
            targ = targets[pt] 
            rcm[pt] = point_position(<vec2d *>(targ.data), num_cams,
                      cparam._control_par.mm, calib, <vec3d> np.PyArray_GETPTR2(res, pt, 0))

        // second we need to reassign pointers to targets and refill the buffer with targets
        // probably we do not:
        // if I understand correctly, this line inside correspondences says that we already updated the frame buffer with the right tnr pointers.
        // frm->targets[j][p1].tnr= i;

    } // external loop is through frames

    // ok, theoretically we have now a buffer full of stuff from 4 frames
    // it's a good buffer on which we can just track stuff
    // and then we need to jump to a next chunk, remove all and start over.
    // the missing part is how to "chain the chunks" or make a smart use of
    // memory and buffers, it's beyond me now

    run->tpar->add = 0;
    track_forward_start(run);

    for (step = run->seq_par->first + 1; step < run->seq_par->last; step++)
    {
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

    for (cam -= 1; cam >= 0; cam--)
        free(corrected[cam]);

    free(corrected);
    free(con);
    free(run->vpar);
    free_control_par(run->cpar);

    return 0;

} // should be end of main now