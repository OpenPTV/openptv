
include_directories("../include/")
add_library (optv SHARED tracking_frame_buf.c calibration.c parameters.c lsqadj.c)


if(UNIX)
  target_link_libraries(optv m)
endif(UNIX)

install(TARGETS optv LIBRARY DESTINATION lib)
