cmake_minimum_required(VERSION 2.8)
PROJECT(ITKOpCannyEdgeDetectionImageFilter)
FIND_PACKAGE(ITK)
IF(ITK_FOUND)
INCLUDE(${ITK_USE_FILE})
ELSE(ITK_FOUND)
MESSAGE(FATAL_ERROR
"ITK not found. Please set ITK_DIR.")
ENDIF(ITK_FOUND)

SET(EXECUTABLE_OUTPUT_PATH
  "${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}"
)

SET(LIBRARY_OUTPUT_PATH
  "${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}"
)

#MESSAGE("${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}")

ADD_EXECUTABLE(opcanny main.cpp util.h) 

ADD_LIBRARY(ITKOpCannyEdgeDetectionImageFilter SHARED itkOpCannyEdgeDetectionImageFilter.txx 
                                                      itkOpCannyEdgeDetectionImageFilter.h 
                                                      itkOpGaussianOperator.txx 
                                                      itkOpGaussianOperator.h
                                                      opConvolutionFilter.cpp
                                                      opConvolutionFilter.h)
TARGET_LINK_LIBRARIES(opcanny ITKCommon ITKIO ITKOpCannyEdgeDetectionImageFilter)
TARGET_LINK_LIBRARIES(ITKOpCannyEdgeDetectionImageFilter ITKCommon ITKIO)

INSTALL(TARGETS ITKOpCannyEdgeDetectionImageFilter LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
INSTALL(TARGETS opcanny RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
INSTALL(FILES test.cfg DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

SET_TARGET_PROPERTIES(ITKOpCannyEdgeDetectionImageFilter PROPERTIES
                      LINKER_LANGUAGE C
                      INSTALL_RPATH_USE_LINK_PATH TRUE)
SET_TARGET_PROPERTIES(opcanny PROPERTIES
                      LINKER_LANGUAGE C
                      INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib
                      INSTALL_RPATH_USE_LINK_PATH TRUE)

SET(CONFIG_FILES test.cfg)

#FILE(MAKE_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/resources )
FOREACH(file ${CONFIG_FILES} )
  GET_FILENAME_COMPONENT(name ${file} NAME)
  CONFIGURE_FILE( ${file} ${EXECUTABLE_OUTPUT_PATH} COPYONLY)
ENDFOREACH(file) 






