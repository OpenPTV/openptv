enable_testing()
find_package(Check REQUIRED)
include_directories(${CHECK_INCLUDE_DIRS})
include_directories(. ../src ../include/)

add_executable(check_fb check_fb.c)
add_dependencies(check_fb optv)

get_property(OPTV_LIBRARY TARGET optv PROPERTY LOCATION)
set(LIBS ${LIBS} ${CHECK_LIBRARIES} ${OPTV_LIBRARY})
target_link_libraries(check_fb ${LIBS})
add_test(check_fb ${CMAKE_CURRENT_BINARY_DIR}/check_fb)

add_custom_command(TARGET check_fb PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/testing_fodder $<TARGET_FILE_DIR:check_fb>/testing_fodder)

