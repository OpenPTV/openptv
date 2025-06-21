# Turn on more warnings
if(MSVC)
  # Force to always compile with W4
  if(CMAKE_C_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
  else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W4")
  endif()
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  # Update if necessary
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
endif()

include_directories("../include/")

add_library (optv SHARED tracking_frame_buf.c calibration.c parameters.c lsqadj.c ray_tracing.c trafo.c vec_utils.c image_processing.c multimed.c imgcoord.c epi.c orientation.c sortgrid.c segmentation.c correspondences.c track.c tracking_run.c track3d.c)


if(UNIX)
  target_link_libraries(optv PRIVATE m)
  target_link_libraries(optv PRIVATE ${CMAKE_DL_LIBS})
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_link_libraries(optv PRIVATE efence)
  endif()
endif(UNIX)

if(WIN32)
  target_link_libraries(optv PRIVATE ws2_32)  # Add Windows socket library if needed
endif(WIN32)

install(TARGETS optv DESTINATION lib)

# Set Debug as the default build type if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
endif()
