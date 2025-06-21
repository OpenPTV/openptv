enable_testing()
find_package(Check REQUIRED)
include_directories(${CHECK_INCLUDE_DIRS})
include_directories(. ../src ../include/)
link_directories(${CHECK_LIBRARY_DIRS})

add_custom_target(testing_data)
add_custom_command(TARGET testing_data 
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/testing_fodder ${CMAKE_BINARY_DIR}/tests/testing_fodder
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/tests/testing_fodder/track/res/
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/tests/testing_fodder/track/img/
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/tests/testing_fodder/test_cavity/res/
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/tests/testing_fodder/test_cavity/img/)

add_executable(check_fb EXCLUDE_FROM_ALL check_fb.c)
add_dependencies(check_fb optv testing_data)

add_executable(check_calibration EXCLUDE_FROM_ALL check_calibration.c)
add_dependencies(check_calibration optv testing_data)

add_executable(check_parameters EXCLUDE_FROM_ALL check_parameters.c)
add_dependencies(check_parameters optv testing_data)

add_executable(check_lsqadj EXCLUDE_FROM_ALL check_lsqadj.c)
add_dependencies(check_lsqadj optv testing_data)

add_executable(check_ray_tracing EXCLUDE_FROM_ALL check_ray_tracing.c)
add_dependencies(check_ray_tracing optv testing_data)

add_executable(check_trafo EXCLUDE_FROM_ALL check_trafo.c)
add_dependencies(check_trafo optv testing_data)

add_executable(check_vec_utils EXCLUDE_FROM_ALL check_vec_utils.c)
add_dependencies(check_vec_utils optv testing_data)

add_executable(check_image_proc EXCLUDE_FROM_ALL check_image_proc.c)
add_dependencies(check_image_proc optv testing_data)

add_executable(check_multimed EXCLUDE_FROM_ALL check_multimed.c)
add_dependencies(check_multimed optv testing_data)

add_executable(check_imgcoord EXCLUDE_FROM_ALL check_imgcoord.c)
add_dependencies(check_imgcoord optv testing_data)

add_executable(check_sortgrid EXCLUDE_FROM_ALL check_sortgrid.c)
add_dependencies(check_sortgrid optv testing_data)

add_executable(check_orientation EXCLUDE_FROM_ALL check_orientation.c)
add_dependencies(check_orientation optv testing_data)

add_executable(check_epi EXCLUDE_FROM_ALL check_epi.c)
add_dependencies(check_epi optv testing_data)

add_executable(check_segmentation EXCLUDE_FROM_ALL check_segmentation.c)
add_dependencies(check_segmentation optv testing_data)

add_executable(check_correspondences EXCLUDE_FROM_ALL check_correspondences.c)
add_dependencies(check_correspondences optv testing_data)

add_executable(check_track EXCLUDE_FROM_ALL check_track.c)
add_dependencies(check_track optv testing_data)

add_executable(check_track3d EXCLUDE_FROM_ALL ../tests/check_track3d.c)
add_dependencies(check_track3d optv testing_data)

set(LIBS ${LIBS} ${CHECK_LIBRARIES} optv)

# Satysfy the new dynamic linker rules on Linux.
if (UNIX)
    set(LIBS ${LIBS} -lm -pthread)
endif (UNIX)

if (APPLE)
    set(LIBS ${LIBS} -L/usr/local/lib)
endif (APPLE)

target_link_libraries(check_fb ${LIBS})
add_test(check_fb ${CMAKE_CURRENT_BINARY_DIR}/check_fb)

target_link_libraries(check_vec_utils ${LIBS})
add_test(check_vec_utils ${CMAKE_CURRENT_BINARY_DIR}/check_vec_utils)

target_link_libraries(check_calibration ${LIBS})
add_test(check_calibration ${CMAKE_CURRENT_BINARY_DIR}/check_calibration)

target_link_libraries(check_parameters ${LIBS})
add_test(check_parameters ${CMAKE_CURRENT_BINARY_DIR}/check_parameters)

target_link_libraries(check_lsqadj ${LIBS})
add_test(check_lsqadj ${CMAKE_CURRENT_BINARY_DIR}/check_lsqadj)

target_link_libraries(check_ray_tracing ${LIBS})
add_test(check_ray_tracing ${CMAKE_CURRENT_BINARY_DIR}/check_ray_tracing)

target_link_libraries(check_trafo ${LIBS})
add_test(check_trafo ${CMAKE_CURRENT_BINARY_DIR}/check_trafo)

target_link_libraries(check_image_proc ${LIBS})
add_test(check_image_proc ${CMAKE_CURRENT_BINARY_DIR}/check_image_proc)

target_link_libraries(check_multimed ${LIBS})
add_test(check_multimed ${CMAKE_CURRENT_BINARY_DIR}/check_multimed)

target_link_libraries(check_imgcoord ${LIBS})
add_test(check_imgcoord ${CMAKE_CURRENT_BINARY_DIR}/check_imgcoord)

target_link_libraries(check_sortgrid ${LIBS})
add_test(check_sortgrid ${CMAKE_CURRENT_BINARY_DIR}/check_sortgrid)

target_link_libraries(check_orientation ${LIBS})
add_test(check_orientation ${CMAKE_CURRENT_BINARY_DIR}/check_orientation)

target_link_libraries(check_epi ${LIBS})
add_test(check_epi ${CMAKE_CURRENT_BINARY_DIR}/check_epi)

target_link_libraries(check_segmentation ${LIBS})
add_test(check_segmentation ${CMAKE_CURRENT_BINARY_DIR}/check_segmentation)

target_link_libraries(check_correspondences ${LIBS})
add_test(check_correspondences ${CMAKE_CURRENT_BINARY_DIR}/check_correspondences)

target_link_libraries(check_track ${LIBS})
add_test(check_track ${CMAKE_CURRENT_BINARY_DIR}/check_track)

target_link_libraries(check_track3d ${LIBS})
add_test(check_track3d ${CMAKE_CURRENT_BINARY_DIR}/check_track3d)

add_custom_target(verify COMMAND ${CMAKE_CTEST_COMMAND})
add_dependencies(verify check_fb check_calibration check_parameters check_lsqadj
check_ray_tracing check_trafo check_vec_utils check_image_proc check_multimed
check_imgcoord check_orientation check_epi check_sortgrid check_segmentation
check_correspondences check_track check_track3d)
