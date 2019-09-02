/* main.c 
I need this file to start preparing some structure in my head. Alex
*/


int main( int argc, const char* argv[] )
{
	printf( "\nHello liboptv World\n\n" );

    // read parameters from the working directory
    // for simplicity all names are default and hard coded (sorry)


    // initialize memory buffers

    for (step = 0; step < N_FRAMES_IN_DIRECTORY-BUFFER_LENGTH-1; step+BUFFER_LENGTH){
        // MAIN LOOP

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