
include_directories("../include/")
add_library (optv SHARED tracking_frame_buf.c calibration.c)

if(UNIX)
  target_link_libraries(optv m)
endif(UNIX)
