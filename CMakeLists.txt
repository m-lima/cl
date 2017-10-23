set(MFL_CL_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include" PARENT_SCOPE)
set(MFL_CL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/runner.cpp" PARENT_SCOPE)
set(MFL_CL_HEADERS
  "${CMAKE_CURRENT_SOURCE_DIR}/include/mfl/cl/program.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/mfl/cl/runner.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/mfl/cl/util.hpp"
  PARENT_SCOPE
)
