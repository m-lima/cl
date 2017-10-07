cmake_minimum_required(VERSION 3.2)
project(Cracker CXX)

# Current state
if (WIN32)
  message(STATUS "WIN32")
endif()
if (MSVC)
  message(STATUS "MSVC")
endif()
if (APPLE)
  message(STATUS "APPLE")
endif()

# Handle Mac's Clang
if(APPLE)
  set(CMAKE_CXX_FLAGS "-std=c++14")
endif()

# Make $HOME compatible with windows
STRING(REGEX REPLACE "\\\\" "/" ENV_HOME_DIR $ENV{HOME})

# Add custom modules
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/findcmake")

# Set base directories
set(CPP_SRC_DIR "src/cpp")
set(LIB_DIR "${CMAKE_SOURCE_DIR}/lib")

# Add manual libs
list (APPEND INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/include")

##------------------------------------------------------------------------------
## Dependencies
##

add_subdirectory("${LIB_DIR}/fmt")
list(APPEND INCLUDE_DIRS "${LIB_DIR}/fmt")
list(APPEND LIBRARIES fmt::fmt-header-only)

find_package(OPENCL REQUIRED)
list(APPEND INCLUDE_DIRS ${OpenCL_INCLUDE_DIR})
list(APPEND LIBRARIES ${OpenCL_LIBRARIES})

##------------------------------------------------------------------------------
## Sources
##

set(SOURCES
  ${CPP_SRC_DIR}/main.cpp
  ${CPP_SRC_DIR}/cracker.cpp
  ${CPP_SRC_DIR}/hash/md5.cpp
  )

##------------------------------------------------------------------------------
## Targets
##

# Create project
add_executable(cracker ${SOURCES})
set_target_properties(cracker PROPERTIES CXX_LANGUAGE_STANDARD 14)

# Link with FIND_PACKAGE
target_link_libraries(cracker PRIVATE ${LIBRARIES})

# Add manual includes
target_include_directories(cracker PRIVATE ${INCLUDE_DIRS})
