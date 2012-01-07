cmake_minimum_required(VERSION 2.6)
PROJECT(ITKOpCannyEdgeDetectionImageFilter)
FIND_PACKAGE(ITK)
IF(ITK_FOUND)
INCLUDE(${ITK_USE_FILE})
ELSE(ITK_FOUND)
MESSAGE(FATAL_ERROR
"ITK not found. Please set ITK_DIR.")
ENDIF(ITK_FOUND)

#SET(PROJECT_BINARY_DIR
#  "$bin/${CMAKE_BUILD_TYPE}"
#  CACHE PATH
#  "Single output directory for building all executables."
#)

SET(PROJECT_SOURCE_DIR
.
#  "${CMAKE_CURRENT_SOURCE_DIR}/src/${PROJECT_NAME}"
)

#SET(CMAKE_BUILD_TYPE
#  "Debug"
#   CACHE
#  "Build type."
#)

SET(EXECUTABLE_OUTPUT_PATH
#  "${CMAKE_CURRENT_SOURCE_DIR}/bin/${CMAKE_BUILD_TYPE}"
  "../../bin/${CMAKE_BUILD_TYPE}"
)

SET(LIBRARY_OUTPUT_PATH
#  "${CMAKE_CURRENT_SOURCE_DIR}/bin/${CMAKE_BUILD_TYPE}"
  "../../bin/${CMAKE_BUILD_TYPE}"
)

#${PROJECT_SOURCE_DIR}/StopWatch.cpp ${PROJECT_SOURCE_DIR}/StopWatch.h)
ADD_EXECUTABLE(opcanny ${PROJECT_SOURCE_DIR}/main.cpp) 
ADD_LIBRARY(ITKOpCannyEdgeDetectionImageFilter SHARED ${PROJECT_SOURCE_DIR}/itkOpCannyEdgeDetectionImageFilter.txx 
                                                      ${PROJECT_SOURCE_DIR}/itkOpCannyEdgeDetectionImageFilter.h 
                                                      ${PROJECT_SOURCE_DIR}/itkOpGaussianOperator.txx 
                                                      ${PROJECT_SOURCE_DIR}/itkOpGaussianOperator.h)
TARGET_LINK_LIBRARIES(ITKOpCannyEdgeDetectionImageFilter ITKCommon ITKIO)
TARGET_LINK_LIBRARIES(opcanny ITKCommon ITKIO ITKOpCannyEdgeDetectionImageFilter)
SET_TARGET_PROPERTIES(ITKOpCannyEdgeDetectionImageFilter PROPERTIES LINKER_LANGUAGE C)


MESSAGE("PROJECT_SOURCE_DIR=${PROJECT_SOURCE_DIR}")
MESSAGE("CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
MESSAGE("EXECUTABLE_OUTPUT_PATH=${CMAKE_BUILD_TYPE}")
MESSAGE("LIBRARY_OUTPUT_PATH=${LIBRARY_OUTPUT_PATH}")


