
include_directories("../include/")
add_library (optv SHARED tracking_frame_buf.c calibration.c parameters.c lsqadj.c ray_tracing.c trafo.c)


if(UNIX)
  target_link_libraries(optv m)
endif(UNIX)

install(TARGETS optv DESTINATION lib)
