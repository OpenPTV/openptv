
include_directories("../include/")
add_library (optv SHARED tracking_frame_buf.c calibration.c parameters.c lsqadj.c ray_tracing.c trafo.c vec_utils.c image_processing.c multimed.c imgcoord.c)


if(UNIX)
  target_link_libraries(optv m)
endif(UNIX)

install(TARGETS optv DESTINATION lib)
