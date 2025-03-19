cmake_minimum_required(VERSION 2.8)
set (CMAKE_BUILD_WITH_INSTALL_RPATH  TRUE)
set (CMAKE_INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN/")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMakeModules/")
project(OpenPTV)

# Read version from file
file(READ "${CMAKE_SOURCE_DIR}/../py_bind/version.txt" FULL_VERSION)
string(STRIP "${FULL_VERSION}" FULL_VERSION)
string(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)" _ "${FULL_VERSION}")
set(VERSION_MAJOR "${CMAKE_MATCH_1}")
set(VERSION_MINOR "${CMAKE_MATCH_2}")
set(VERSION_PATCH "${CMAKE_MATCH_3}")

enable_testing()
add_subdirectory(src)
add_subdirectory(tests)

INSTALL(DIRECTORY include/ DESTINATION include/optv/)

set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/README.txt")
set(CPACK_PACKAGE_VERSION_MAJOR "${VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${VERSION_PATCH}")
set(CPACK_PACKAGE_CONTACT "OpenPTV community <openptv@googlegroups.com>")

set(CPACK_SOURCE_IGNORE_FILES 
    "CTestTestfile.cmake"
    "/build/"
    "\\\\.exe$"
    "\\\\.tar\\\\.gz$"
    "\\\\.swp$"
    "/.deps/"
    "/.libs/"
    ".gitignore")

include(CPack)

# Add near the top, after project(OpenPTV)
if(UNIX)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# Add visibility settings
if(UNIX)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden")
    add_definitions(-DOPTV_EXPORT=__attribute__((visibility(\"default\"))))
else()
    add_definitions(-DOPTV_EXPORT=)
endif()