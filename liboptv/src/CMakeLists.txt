
include_directories("../include/")
add_library (optv SHARED tracking_frame_buf.c parameters.c ray_tracing.c lsqadj.c)

if(UNIX)
  target_link_libraries(optv m)
endif(UNIX)
