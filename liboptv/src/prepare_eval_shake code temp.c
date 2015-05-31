/* This is very similar to prepare_eval, but it is sufficiently different in
   small but devious ways, not only parameters, that for now it'll be a
   different function. */
void prepare_eval_shake(int n_img) {
    char* target_file_base[4];
    int i_img, i, filenumber, step_shake, count = 0, frame_count = 0;
    int seq_first, seq_last;
    int frame_used, part_used;
    int max_shake_points, max_shake_frames;
	double  dummy;
    sequence_par *seq_par;

    int part_pointer; /* Holds index of particle later */

    frame frm;
    frame_init(&frm, n_img, MAX_TARGETS);

	seq_par = read_sequence_par("parameters/sequence.par");

	fpp = fopen ("parameters/shaking.par", "r");
    fscanf (fpp,"%d\n", &seq_first);
    fscanf (fpp,"%d\n", &seq_last);
    fscanf (fpp,"%d\n", &max_shake_points);
    fscanf (fpp,"%d\n", &max_shake_frames);
    fclose (fpp);

    step_shake = (int)((double)(seq_last - seq_first + 1) /
        (double)max_shake_frames + 0.5);
    for (filenumber = seq_first + 2; filenumber < seq_last - 1; \
        filenumber += step_shake)
    {
        frame_used = 0;
        read_frame(&frm, "res/rt_is", "res/ptv_is", NULL,
            seq_par->img_base_name, filenumber);

        for (i = 0; i < frm.num_parts; i++) {
            part_used = 0;

            for (i_img = 0; i_img < n_img; i_img++) {
                part_pointer = frm.correspond[i].p[i_img];
                if ((part_pointer != CORRES_NONE) && \
                    (frm.path_info[i].prev != PREV_NONE) && \
                    (frm.path_info[i].next != NEXT_NONE) )
                {
                    pix[i_img][count].x = frm.targets[i_img][part_pointer].x;
                    pix[i_img][count].y = frm.targets[i_img][part_pointer].y;
                    pix[i_img][count].pnr = count;

                    pixel_to_metric (pix[i_img][count].x, pix[i_img][count].y,
                        imx,imy, pix_x, pix_y,
                        &crd[i_img][count].x, &crd[i_img][count].y, chfield);
                    crd[i_img][count].pnr = count;

                    frame_used = 1;
                    part_used = 1;
                }
            }
            if (part_used == 1) count++;
            if (count >= max_shake_points) break;
        }
        if (frame_used == 1) frame_count++;
        if ((count >= max_shake_points) || (frame_count > max_shake_frames))
            break;
    }
    free_frame(&frm);
    nfix = count;
}
