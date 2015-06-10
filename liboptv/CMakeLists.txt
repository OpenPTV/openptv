cmake_minimum_required(VERSION 2.8)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMakeModules/")
project(OpenPTV)
enable_testing()
add_subdirectory(src)
add_subdirectory(tests)

INSTALL(DIRECTORY include/ DESTINATION include/optv/)

set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/README.txt")
set(CPACK_PACKAGE_VERSION_MAJOR "0")
set(CPACK_PACKAGE_VERSION_MINOR "1")
set(CPACK_PACKAGE_VERSION_PATCH "0")
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
