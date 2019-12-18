# author: Viktor Richter vrichter@techfak.uni-bielefeld.de
# licence: MIT
cmake_minimum_required(VERSION 3.0)
project(fformation-gco VERSION 3.0.0)

set(CMAKE_CXX_STANDARD 11)

option(BUILD_LUA_BINDING "Build lua binding" OFF)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

find_package(Boost 1.54 COMPONENTS program_options REQUIRED)
find_package(fformation 3.0 REQUIRED)

set(LIB_SRC
  "${PROJECT_SOURCE_DIR}/src/GraphCutsOptimization.cpp"
  "${PROJECT_SOURCE_DIR}/src/GraphCutsOptimization.h"
)
set(GCO_SRC
  "${PROJECT_SOURCE_DIR}/gco/src/GCoptimization.cpp"
  "${PROJECT_SOURCE_DIR}/gco/src/GCoptimization.h"
  "${PROJECT_SOURCE_DIR}/gco/src/LinkedBlockList.cpp"
  "${PROJECT_SOURCE_DIR}/gco/src/LinkedBlockList.h"
  "${PROJECT_SOURCE_DIR}/gco/src/block.h"
  "${PROJECT_SOURCE_DIR}/gco/src/energy.h"
  "${PROJECT_SOURCE_DIR}/gco/src/graph.h"
  )
add_library("${PROJECT_NAME}" SHARED  ${LIB_SRC} ${GCO_SRC})
target_include_directories("${PROJECT_NAME}" PUBLIC
  $<BUILD_INTERFACE:
      "${PROJECT_SOURCE_DIR}/src"
      "${FFORMATION_INCLUDE_DIRS}"
  >
  $<INSTALL_INTERFACE:include>
  )
install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}-export
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(FILES src/GraphCutsOptimization.h DESTINATION include/${PROJECT_NAME})
install(EXPORT ${PROJECT_NAME}-export
  FILE
    ${PROJECT_NAME}Targets.cmake
  NAMESPACE
    ${PROJECT_NAME}::
  DESTINATION
    lib/cmake/${PROJECT_NAME}
)


if(${BUILD_LUA_BINDING})
    find_package(Lua 5.1 REQUIRED)
    add_library("${PROJECT_NAME}_lua" SHARED
      ${PROJECT_SOURCE_DIR}/src/lua.cpp
      ${LIB_SRC} ${GCO_SRC}
    )
    target_include_directories("${PROJECT_NAME}_lua" PUBLIC 
      $<BUILD_INTERFACE:
          "${FFORMATION_INCLUDE_DIRS}"
          ${LUA_INCLUDE_DIR}
          "${PROJECT_SOURCE_DIR}/luwra/lib"
      >
    )
    target_link_libraries("${PROJECT_NAME}_lua"
        ${FFORMATION_LIBRARIES}
        ${LUA_LIBRARIES}
    )
    set_property(TARGET "${PROJECT_NAME}_lua" PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)
    install(TARGETS "${PROJECT_NAME}_lua" DESTINATION "lib/" )
endif()

set(APPS
  "evaluation"
  "evaluation-personal"
)

foreach(EXAMPLE ${APPS})
  add_executable("${PROJECT_NAME}-${EXAMPLE}" "${PROJECT_SOURCE_DIR}/app/${EXAMPLE}.cpp")
  target_link_libraries("${PROJECT_NAME}-${EXAMPLE}"
    ${PROJECT_NAME}
    ${FFORMATION_LIBRARIES}
    ${Boost_LIBRARIES}
    pthread
    )
  target_include_directories("${PROJECT_NAME}-${EXAMPLE}" PUBLIC 
    $<BUILD_INTERFACE: "${PROJECT_SOURCE_DIR}/src"
                       "${FFORMATION_INCLUDE_DIRS}"
    )
endforeach(EXAMPLE)
