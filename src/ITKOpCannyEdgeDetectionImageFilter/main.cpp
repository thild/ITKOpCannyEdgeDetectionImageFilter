/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: Image5.cxx,v $
  Language:  C++
  Date:      $Date: 2009-03-17 21:11:41 $
  Version:   $Revision: 1.16 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 ) 
#endif  
 
// Software Guide : BeginLatex
//
// This example illustrates how to import data into the \doxygen{Image}
// class. This is particularly useful for interfacing with other software
// systems. Many systems use a contiguous block of memory as a buffer
// for image pixel data. The current example assumes this is the case and
// feeds the buffer into an \doxygen{ImportImageFilter}, thereby producing an
// Image as output.

//
// For fun we create a synthetic image with a centered sphere in
// a locally allocated buffer and pass this block of memory to the
// ImportImageFilter. This example is set up so that on execution, the
// user must provide the name of an output file as a command-line argument.
//
// \index{itk::ImportImageFilter!Instantiation}
// \index{itk::ImportImageFilter!Header}
//
// First, the header file of the ImportImageFilter class must be
// included.
//
// Software Guide : EndLatex 


// Software Guide : BeginCodeSnippet
#include "itkImage.h"
#include "itkImportImageFilter.h"
// Software Guide : EndCodeSnippet

#include "itkImageFileWriter.h"                 
#include "itkImageFileReader.h"
#include "itkObjectFactoryBase.h"
#include "itkDynamicLoader.h"
#include "itkOpCannyEdgeDetectionImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkCannyEdgeDetectionImageFilter.h"

#include <iostream> 
#include <iomanip>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <list>
#include <math.h>
#include <omp.h>
#include "itkStopWatch.h"
#include <sys/time.h>
#include <sys/resource.h>

using std::cout;
using std::cerr;   
using std::endl;
using std::setw;
using std::string;
using std::ifstream; 

using namespace std;
 

int main (int argc, char *argv[])
{
 cout.setf(ios::fixed);  
 cout.precision(3);

if( argc < 3 ) 
    {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << " inputImage outputImage opOutputImage [variance upperThreshold lowerThreshold opUpperThreshold opLowerThreshold iterations]" << std::endl;
    return EXIT_FAILURE; 
    }
   
  const char *inputFilename  = argv[1];
  const char *outputFilename = argv[2];
  const char *opOutputFilename = argv[3];
  float variance = 1.4;
  float upperThreshold = 7.0;
  float lowerThreshold = 4.0;
  float opUpperThreshold = 7.0;
  float opLowerThreshold = 4.0;
  int iterations = 10; 

  if( argc > 4 )
    {
    variance = atof( argv[4] );
    }

  if( argc > 5 )
    {
    upperThreshold = atof( argv[5] );
    }

  if( argc > 6 )
    {
    lowerThreshold = atof( argv[6] );
    }

  if( argc > 7 )
    {
    opUpperThreshold = atof( argv[7] );
    }

  if( argc > 8 )
    {
    opLowerThreshold = atof( argv[8] );
    }

  if( argc > 9 )
    {
    iterations = atoi( argv[9] );
    }
    
  std::cout << "##### Begin Test #####" << std::endl;
  std::cout << "InputFilename = " << inputFilename << std::endl;
  std::cout << "Variance = " << variance << std::endl;
  std::cout << "UpperThreshold = " << upperThreshold << std::endl;
  std::cout << "LowerThreshold = " << lowerThreshold << std::endl;
  std::cout << "OpUpperThreshold = " << opUpperThreshold << std::endl;
  std::cout << "OpLowerThreshold = " << opLowerThreshold << std::endl;
  std::cout << "Iterations = " << iterations << std::endl;

  typedef unsigned char     CharPixelType;  //  IO
  typedef float             RealPixelType;  //  Operations
  const   unsigned int      Dimension = 2;

  typedef itk::Image< CharPixelType, Dimension >    CharImageType;
  typedef itk::Image< RealPixelType, Dimension >    RealImageType;

  typedef itk::ImageFileReader< CharImageType >  ReaderType;
  typedef itk::ImageFileWriter< CharImageType >  WriterType;

  typedef itk::ImageRegionIterator<RealImageType> ImageIterator;

  //  Software Guide : BeginLatex
  //
  //  This filter operates on image of pixel type float. It is then necessary
  //  to cast the type of the input images that are usually of integer type.
  //  The \doxygen{CastImageFilter} is used here for that purpose. Its image 
  //  template parameters are defined for casting from the input type to the
  //  float type using for processing.
  //
  //  Software Guide : EndLatex

  // Software Guide : BeginCodeSnippet
  typedef itk::CastImageFilter< CharImageType, RealImageType> CastToRealFilterType;
  typedef itk::CastImageFilter< RealImageType, CharImageType> CastToCharFilterType;
  // Software Guide : EndCodeSnippet

  typedef itk::RescaleIntensityImageFilter<RealImageType, CharImageType > RescaleFilter;


  //  Software Guide : BeginLatex
  //
  //  The \doxygen{CannyEdgeDetectionImageFilter} is instantiated using the float image type.
  //
  //  \index{itk::CannyEdgeDetectionImageFilter|textbf}
  //
  //  Software Guide : EndLatex


  typedef itk::OpCannyEdgeDetectionImageFilter<RealImageType, RealImageType> OpCannyFilter;
  typedef itk::CannyEdgeDetectionImageFilter<RealImageType, RealImageType> CannyFilter;

  //Setting the IO 

  ReaderType::Pointer reader = ReaderType::New();

  //Setting the ITK pipeline filter
  double elapsed = 0; 
  double opElapsed = 0; 

  //itk::MultiThreader::SetGlobalMaximumNumberOfThreads( 1 );

  try { 
   
//        typedef itk::Image< float, 2 >  ImageType;
//        ImageType::Pointer image = ImageType::New();
//        ImageType::SizeType size;
//        size[0] = 64; // size along X
//        size[1] = 64; // size along Y
//        
//        ImageType::IndexType start;
//        start[0] = 0; // first index on X
//        start[1] = 0; // first index on Y
//        
//        ImageType::RegionType region;
//        region.SetSize( size );
//        region.SetIndex( start );         
//        
//        image->SetRegions( region );
//        image->Allocate();         
//        
//        float k = 1; 
//        for(int i = 0; i < 64; i++) {
//          for(int j = 0; j < 64; j++) {
//            ImageType::IndexType pixelIndex;
//            pixelIndex[0] = j; // x position
//            pixelIndex[1] = i; // y position
//            image->SetPixel (pixelIndex, k++);
//          }
//        }
                 
        ReaderType::Pointer reader = ReaderType::New(); 
        reader->SetFileName( inputFilename);
        
//        reader->SetFileName( inputFilename  );
//        RescaleFilter::Pointer reascale = RescaleFilter::New();
//        reascale->SetInput( reader->GetOutput() );
//        WriterType::Pointer writer = WriterType::New();
//        writer->SetFileName("bla.png");
//        writer->SetInput( reascale->GetOutput() );
//        writer->Update();  
//         
        //struct rusage usage; 
        //getrusage(RUSAGE_SELF, &usage);  
        //cout << endl << "Initial total memory usage = " << usage.ru_maxrss << endl;
        cout << endl << "ITK OpCanny - " << iterations << " iteration(s)" << endl;
        
        for(int i = 0; i < iterations; i++) {
        
            CastToRealFilterType::Pointer toReal = CastToRealFilterType::New();
            toReal->SetInput( reader->GetOutput() );
    
            OpCannyFilter::Pointer opCannyFilter = OpCannyFilter::New();
            opCannyFilter->SetInput( toReal->GetOutput() );
            opCannyFilter->SetVariance( variance );
            opCannyFilter->SetUpperThreshold( opUpperThreshold );
            opCannyFilter->SetLowerThreshold( opLowerThreshold );
         
         
            RescaleFilter::Pointer rescale = RescaleFilter::New();
            rescale->SetOutputMinimum(   0 );
            rescale->SetOutputMaximum( 255 );
            rescale->SetInput( opCannyFilter->GetOutput() );
             
            WriterType::Pointer writer = WriterType::New();
            writer->SetFileName(opOutputFilename);
            writer->SetInput( rescale->GetOutput() );    
            writer->Update();          

            StopWatch st = opCannyFilter->GetStopWatch();
            vector<MeasuringStep> ms = st.GetMeasuringSteps();
            for ( vector<MeasuringStep>::iterator it = ms.begin(); it != ms.end(); it++ ) { 
                  cout << std::setprecision(4) << std::setiosflags(std::ios::fixed) << left << setw(40)
                    << it->Step << ": " << it->Instant << " : " << it->Elapsed << endl;
            }
            cout << "Iteration " << i + 1 << " of 10 elapsed: " << st.GetElapsedTime() << endl;
            opElapsed += st.GetElapsedTime();         
        }     
        
//        
//        
//        //getrusage(RUSAGE_SELF, &usage);  
//        //cout << endl << "Final total memory usage = " << usage.ru_maxrss << endl;
//        
////        counter.Stop();
////        cout << endl << "##### ITK OpCanny including writing" << endl; 
//        
        reader = ReaderType::New(); 
        reader->SetFileName( inputFilename  );  
        

        cout << endl << "ITK Canny - " << iterations << " iteration(s)" << endl;
  
        for(int i = 0; i < iterations; i++) {    
  
            CastToRealFilterType::Pointer toReal = CastToRealFilterType::New();  
            toReal->SetInput( reader->GetOutput() );
     
            CannyFilter::Pointer cannyFilter = CannyFilter::New();
            cannyFilter->SetInput( toReal->GetOutput() );
            cannyFilter->SetVariance( variance );
            cannyFilter->SetUpperThreshold( upperThreshold );
            cannyFilter->SetLowerThreshold( lowerThreshold );
                                
            //The output of an edge filter is 0 or 1
            RescaleFilter::Pointer rescale = RescaleFilter::New();
            rescale->SetOutputMinimum(   0 );
            rescale->SetOutputMaximum( 255 );
            rescale->SetInput( cannyFilter->GetOutput() );
                       
            //time = counter.GetElapsedTime();
                    
            //cout << endl << "##### ITK Canny including writing" << endl;
            //counter.Reset();
            WriterType::Pointer writer = WriterType::New();
            writer->SetFileName( outputFilename );
            writer->SetInput( rescale->GetOutput() );
            writer->Update();

            StopWatch st = cannyFilter->GetStopWatch();
            vector<MeasuringStep> ms = st.GetMeasuringSteps();
            for ( vector<MeasuringStep>::iterator it = ms.begin(); it != ms.end(); it++ ) { 
                  cout << std::setprecision(4) << std::setiosflags(std::ios::fixed) << left << setw(40)
                    << it->Step << ": " << it->Instant << " : " << it->Elapsed << endl;
            }
            cout << "Iteration " << i + 1 << " of 10 elapsed: " << st.GetElapsedTime() << endl;
            elapsed += st.GetElapsedTime();         
        }
        
//        counter.Stop();
//        cout << counter.GetElapsedTime() << endl;
        
        cout << endl << "ITK OpCanny " << iterations << " iteration(s) median" << endl;
        cout << opElapsed / iterations << endl;

        cout << endl << "ITK Canny " << iterations << " iteration(s) median" << endl;
        cout << elapsed / iterations << endl;  
        
        cout << endl << "ITK Canny / ITK OpCanny" << endl;   
        cout << (elapsed / iterations) /  (opElapsed / iterations) << "x faster." << endl;

        std::cout << endl << "##### End Test #####" << std::endl << std::endl << std::endl;
           
    }
  catch( itk::ExceptionObject & err ) 
    { 
    std::cout << "ExceptionObject caught !" << std::endl; 
    std::cout << err << std::endl; 
    return EXIT_FAILURE;
    } 


  return EXIT_SUCCESS;	
	
}


