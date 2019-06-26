# author: Viktor Richter vrichter@techfak.uni-bielefeld.de
# licence: MIT
project(fformation-gco)
cmake_minimum_required(VERSION 2.8)

set(PROJECT_VERSION_MAJOR 3)
set(PROJECT_VERSION_MINOR 0)
set(PROJECT_VERSIOM_PATCH 0)
set(PROJECT_NUMBER "v${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSIOM_PATCH}")

# add c++11 compiler flag
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -DBOOST_SIGNALS_NO_DEPRECATION_WARNING")
elseif(COMPILER_SUPPORTS_CXX0X)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
  message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

find_package(Boost 1.54 COMPONENTS program_options REQUIRED)
find_package(fformation 3.0 REQUIRED)
find_package(Lua51 5.1 REQUIRED)
add_subdirectory(gco)

set(LIB_SRC
  "${PROJECT_SOURCE_DIR}/src/GraphCutsOptimization.cpp"
  "${PROJECT_SOURCE_DIR}/src/GraphCutsOptimization.h"
)

add_library("${PROJECT_NAME}" STATIC  ${LIB_SRC})

add_library("${PROJECT_NAME}_lua" SHARED ${PROJECT_SOURCE_DIR}/src/lua.cpp ${LIB_SRC}
    "gco/src/GCoptimization.cpp"
    "gco/src/GCoptimization.h"
    "gco/src/LinkedBlockList.cpp"
    "gco/src/LinkedBlockList.h"
    "gco/src/block.h"
    "gco/src/energy.h"
    "gco/src/graph.h"
  )

target_link_libraries("${PROJECT_NAME}_lua"
    ${FFORMATION_LIBRARIES}
)
set_property(TARGET "${PROJECT_NAME}_lua" PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)

set(APPS
  "evaluation"
)

include_directories(
  "${PROJECT_SOURCE_DIR}/src"
  "${FFORMATION_INCLUDE_DIRS}"
  "${PROJECT_SOURCE_DIR}/luwra/lib"
  "${LUA_INCLUDE_DIR}"
)

link_directories(
  ${Boost_LIBRARY_DIRS}
  ${LUA_LIBRARIES}
  )

foreach(EXAMPLE ${APPS})
  add_executable("${PROJECT_NAME}-${EXAMPLE}" "${PROJECT_SOURCE_DIR}/app/${EXAMPLE}.cpp")
  target_link_libraries("${PROJECT_NAME}-${EXAMPLE}"
    ${PROJECT_NAME}
    ${FFORMATION_LIBRARIES}
    ${Boost_LIBRARIES}
    gco
    )
endforeach(EXAMPLE)

install(DIRECTORY "${PROJECT_SOURCE_DIR}/src/"
        DESTINATION "include/${PROJECT_NAME}/"
        FILES_MATCHING PATTERN "*.h" PATTERN "*.inl"
        PATTERN "matlab" EXCLUDE)

install(TARGETS ${PROJECT_NAME}
  "${PROJECT_NAME}_lua"
  DESTINATION "lib/"
  )
