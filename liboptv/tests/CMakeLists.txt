enable_testing()
find_package(Check REQUIRED)
include_directories(${CHECK_INCLUDE_DIRS})
include_directories(. ../src ../include/)

add_executable(check_fb EXCLUDE_FROM_ALL check_fb.c)
add_dependencies(check_fb optv)

add_executable(check_calibration EXCLUDE_FROM_ALL check_calibration.c)
add_dependencies(check_calibration optv)

add_executable(check_parameters EXCLUDE_FROM_ALL check_parameters.c)
add_dependencies(check_parameters optv)


add_executable(check_lsqadj EXCLUDE_FROM_ALL check_lsqadj.c)
add_dependencies(check_lsqadj optv)


get_property(OPTV_LIBRARY TARGET optv PROPERTY LOCATION)
set(LIBS ${LIBS} ${CHECK_LIBRARIES} ${OPTV_LIBRARY})

# Satysfy the new dynamic linker rules on Linux.
if (UNIX)
    set(LIBS ${LIBS} -lm -pthread)
endif (UNIX)

target_link_libraries(check_fb ${LIBS})
add_test(check_fb ${CMAKE_CURRENT_BINARY_DIR}/check_fb)

add_custom_command(TARGET check_fb PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/testing_fodder $<TARGET_FILE_DIR:check_fb>/testing_fodder)

target_link_libraries(check_calibration ${LIBS})
add_test(check_calibration ${CMAKE_CURRENT_BINARY_DIR}/check_calibration)

add_custom_command(TARGET check_calibration PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/testing_fodder $<TARGET_FILE_DIR:check_calibration>/testing_fodder)

target_link_libraries(check_parameters ${LIBS})
add_test(check_parameters ${CMAKE_CURRENT_BINARY_DIR}/check_parameters)

add_custom_command(TARGET check_parameters PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/testing_fodder $<TARGET_FILE_DIR:check_parameters>/testing_fodder)

target_link_libraries(check_lsqadj ${LIBS})
add_test(check_lsqadj ${CMAKE_CURRENT_BINARY_DIR}/check_lsqadj)

add_custom_command(TARGET check_lsqadj PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/testing_fodder $<TARGET_FILE_DIR:check_lsqadj>/testing_fodder)

add_custom_target(verify COMMAND ${CMAKE_CTEST_COMMAND})
add_dependencies(verify check_fb check_calibration check_parameters check_lsqadj)

