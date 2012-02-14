/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkOpCannyEdgeDetectionImageFilter.txx,v $
  Language:  C++
  Date:      $Date: 2009-08-17 12:01:33 $
  Version:   $Revision: 1.56 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkOpCannyEdgeDetectionImageFilter_txx
#define __itkOpCannyEdgeDetectionImageFilter_txx
#include "itkOpCannyEdgeDetectionImageFilter.h"

#include "itkZeroCrossingImageFilter.h"
#include "itkNeighborhoodInnerProduct.h"
#include "itkNumericTraits.h"
#include "itkProgressReporter.h"
#include "itkGradientMagnitudeImageFilter.h"

#include "itkStopWatch.h"

#include <valgrind/callgrind.h>

#include "opConvolutionFilter.h"

#include <omp.h>
#ifdef __SSE4_1__
#include <smmintrin.h>  
#endif 

#include <emmintrin.h>
#include <xmmintrin.h>

#include <assert.h> 

namespace itk
{

using std::cout;
using std::endl;
  
template <class TInputImage, class TOutputImage>
OpCannyEdgeDetectionImageFilter<TInputImage, TOutputImage>::
OpCannyEdgeDetectionImageFilter()
{
  m_Variance.Fill(0.0);
  m_MaximumError.Fill(0.01);

  m_OutsideValue = NumericTraits<OutputImagePixelType>::Zero;
  m_Threshold = NumericTraits<OutputImagePixelType>::Zero;
  m_UpperThreshold = NumericTraits<OutputImagePixelType>::Zero;
  m_LowerThreshold = NumericTraits<OutputImagePixelType>::Zero;
  
  m_UpdateBuffer  = OutputImageType::New();  
  m_GaussianBuffer  = OutputImageType::New();  
  
  m_BoundaryBuffer1  = OutputImageType::New();  
  m_BoundaryBuffer2  = OutputImageType::New();  
  
  //Initialize the list
  m_NodeStore = ListNodeStorageType::New();
  m_NodeList = ListType::New();
}



template <class TInputImage, class TOutputImage>
void
OpCannyEdgeDetectionImageFilter<TInputImage, TOutputImage>
::AllocateUpdateBuffer()
{
  // The update buffer looks just like the input.

  typename TInputImage::ConstPointer input = this->GetInput();
  
  //m_UpdateBuffer->CopyInformation( input );
  m_UpdateBuffer->SetRequestedRegion(input->GetRequestedRegion());
  m_UpdateBuffer->SetBufferedRegion(input->GetBufferedRegion());
  m_UpdateBuffer->Allocate();  
  
  m_GaussianBuffer->SetRequestedRegion(input->GetRequestedRegion());
  m_GaussianBuffer->SetBufferedRegion(input->GetBufferedRegion());
  m_GaussianBuffer->Allocate();  

    typename TInputImage::SizeType regionSize = 
        input->GetRequestedRegion().GetSize();
  
  uint maxSide = regionSize[0] > regionSize[1] ? regionSize[0] : regionSize[1];

  int stride = calculateAlignedStride (maxSide + 2, sizeof(InputImagePixelType), 16 ); 
  regionSize[0] = stride;
  regionSize[1] = 16;

  m_BoundaryBuffer1->SetRequestedRegion(regionSize);
  m_BoundaryBuffer1->SetBufferedRegion(regionSize);
  m_BoundaryBuffer1->Allocate();  

  m_BoundaryBuffer2->SetRequestedRegion(regionSize);
  m_BoundaryBuffer2->SetBufferedRegion(regionSize);
  m_BoundaryBuffer2->Allocate();  
        
}
 
template <class TInputImage, class TOutputImage>
void 
OpCannyEdgeDetectionImageFilter<TInputImage,TOutputImage>
::GenerateInputRequestedRegion() throw(InvalidRequestedRegionError)
{
  // call the superclass' implementation of this method
  Superclass::GenerateInputRequestedRegion();  
  return;  
  // get pointers to the input and output
  typename Superclass::InputImagePointer  inputPtr = 
    const_cast< TInputImage * >( this->GetInput());
  typename Superclass::OutputImagePointer outputPtr = this->GetOutput();
  
  if ( !inputPtr || !outputPtr )
    {
    return;
    }

  //Set the kernel size.
  // fix me this needs to be based on the variance of the gaussian filter
  typename TInputImage::SizeValueType radius = 1;
  
  // get a copy of the input requested region (should equal the output
  // requested region)
  typename TInputImage::RegionType inputRequestedRegion;
  inputRequestedRegion = inputPtr->GetRequestedRegion();

  // pad the input requested region by the operator radius
  inputRequestedRegion.PadByRadius( radius );

  // crop the input requested region at the input's largest possible region
  if ( inputRequestedRegion.Crop(inputPtr->GetLargestPossibleRegion()) )
    {
    inputPtr->SetRequestedRegion( inputRequestedRegion );
    return;
    }
  else
    {
    // Couldn't crop the region (requested region is outside the largest
    // possible region).  Throw an exception.

    // store what we tried to request (prior to trying to crop)
    inputPtr->SetRequestedRegion( inputRequestedRegion );
    
    // build an exception
    InvalidRequestedRegionError e(__FILE__, __LINE__);
    OStringStream msg;
    msg << this->GetNameOfClass()
        << "::GenerateInputRequestedRegion()";
    e.SetLocation(msg.str().c_str());
    e.SetDescription("Requested region is (at least partially) outside the largest possible region.");
    e.SetDataObject(inputPtr);
    throw e;
    }
}


template< class TInputImage, class TOutputImage >
void
OpCannyEdgeDetectionImageFilter< TInputImage, TOutputImage >
::GenerateData()
{
//  // Allocate the output
  this->GetOutput()->SetBufferedRegion( this->GetOutput()->GetRequestedRegion() );
  this->GetOutput()->Allocate();
  
  
    typename  InputImageType::ConstPointer  input  = this->GetInput();
    typename TInputImage::SizeType regionSize = 
        input->GetRequestedRegion().GetSize();
          
    int nHeight = regionSize[1];
    int stride = input->GetOffsetTable()[1];
    
  this->AllocateUpdateBuffer();
  
  StopWatch* sw = StopWatchPool::GetStopWatch("OpCannyEdgeDetectionImageFilter"); 
  sw->StartNew();  
  sw->AddCheckpoint("Begin GaussianBlur", true);  
  // blur the image to reduce noise    
  this->GaussianBlur ( input->GetBufferPointer(), m_GaussianBuffer->GetBufferPointer() );
  sw->AddCheckpoint("GaussianBlur");  
    
  //2. Calculate 2nd order directional derivative-------
  // Calculate the 2nd order directional derivative of the smoothed image.
  // The output of this filter will be used to store the directional
  // derivative.
  sw->AddCheckpoint("Begin Compute2ndDerivative", true);  
  this->Compute2ndDerivative();    
  sw->AddCheckpoint("Compute2ndDerivative");  
    
  sw->AddCheckpoint("Begin Compute2ndDerivativePos", true);  
  this->Compute2ndDerivativePos();    
  sw->AddCheckpoint("Compute2ndDerivativePos");

  sw->AddCheckpoint("Begin ZeroCrossing", true);  
  this->ZeroCrossing();    
  sw->AddCheckpoint("ZeroCrossing");  
  
  CALLGRIND_START_INSTRUMENTATION;
  sw->AddCheckpoint("Begin Multiply", true);
  this->Multiply(stride, nHeight,  
                 m_UpdateBuffer->GetBufferPointer(), 
                 m_GaussianBuffer->GetBufferPointer(), 
                 m_GaussianBuffer->GetBufferPointer());    
  sw->AddCheckpoint("Multiply");  
                 
  sw->AddCheckpoint("Begin HysteresisThresholding", true);
  StopWatch b;
  b.Start();
  CALLGRIND_TOGGLE_COLLECT;
  this->HysteresisThresholding();                    
  CALLGRIND_TOGGLE_COLLECT;
  CALLGRIND_STOP_INSTRUMENTATION;  
  
  b.Stop();
  sw->AddCheckpoint("HysteresisThresholding");
             
  sw->Stop();
  sw->Stop();
}





// Calculate the second derivative
template< class TInputImage, class TOutputImage >
void
OpCannyEdgeDetectionImageFilter< TInputImage, TOutputImage >
::Compute2ndDerivative() 
{
//    return;
  typename TInputImage::SizeType regionSize = 
  this->GetInput()->GetRequestedRegion().GetSize();
    
  const int imageWidth = regionSize[0];
  const int imageHeight = regionSize[1];
  const int imageStride = this->GetInput()->GetOffsetTable()[1]; 
  
  float* inputImage  = m_GaussianBuffer->GetBufferPointer(); 
  float* outputImage = this->GetOutput()->GetBufferPointer();
  //float* outputImage  = m_UpdateBuffer->GetBufferPointer();
//  printImage(imageWidth, imageHeight, imageStride, inputImage);
  
  Compute2ndDerivative(imageStride, imageWidth, imageHeight, inputImage, outputImage, imageStride);
//  printImage(imageWidth, imageHeight, imageStride, outputImage);

  float* boundaryImage = m_BoundaryBuffer1->GetBufferPointer(); 
  
    
//void copy2DBoundaryChunk(const float* inBuffer, float* outBuffer,
//                           const int outStride, const int outWidth, const int outHeight, 
//                           const int replicateLeft, const int replicateTop,
//                           const int replicateRight, const int replicateBottom,
//                           const int inStride,  const int inWidth, const int inHeight);        
  //left boundaries
  int stride = calculateAlignedStride (6, sizeof(InputImagePixelType), 16 ); 
  
  copy2DBoundaryChunk(inputImage, boundaryImage,
                     stride, 6, imageHeight + 2, 
                     1, 1,
                     0, 1,
                     imageStride, imageWidth, imageHeight);  
                      
                      
                      
//  printImage(6, imageHeight + 2, stride, boundaryImage);
  Compute2ndDerivative(stride, 6, imageHeight + 2, boundaryImage, outputImage - imageStride - 1, imageStride);
//  printImage(imageWidth, imageHeight, imageStride, outputImage);
   
  //right boundaries
  copy2DBoundaryChunk(inputImage, boundaryImage,
                     stride, 6, imageHeight + 2, 
                     0, 1,
                     1, 1,
                     imageStride, imageWidth, imageHeight);  
  
//  printImage(6, imageHeight + 2, stride, boundaryImage);
  Compute2ndDerivative(stride, 6, imageHeight + 2, boundaryImage, outputImage - imageStride + imageWidth - 5, imageStride);
//  printImage(imageWidth, imageHeight, imageStride, outputImage);
  
  stride = calculateAlignedStride (imageWidth + 2, sizeof(InputImagePixelType), 16 ); 
  
  //top boundaries
  copy2DBoundaryChunk(inputImage, boundaryImage,
                     stride, imageWidth + 2, 3, 
                     1, 1,
                     1, 0,
                     imageStride, imageWidth, imageHeight);  
  
//  printImage(imageWidth + 2, 3, stride, boundaryImage);
  Compute2ndDerivative(stride, imageWidth + 2, 3, boundaryImage, outputImage - imageStride - 1, imageStride);
//  printImage(imageWidth, imageHeight, imageStride, outputImage);
  
  //bottom boundaries
  copy2DBoundaryChunk(inputImage, boundaryImage,
                     stride, imageWidth + 2, 3, 
                     1, 0,
                     1, 1,
                     imageStride, imageWidth, imageHeight);  
  
//  printImage(imageWidth + 2, 3, stride, boundaryImage);
  Compute2ndDerivative(stride, imageWidth + 2, 3, boundaryImage, outputImage + (imageHeight - 2) * imageStride - 1, imageStride);
//  cout << "aqui" << endl;
//  printImage(imageWidth, imageHeight, imageStride, outputImage);
}

 
 



//// Calculate the second derivative
//template< class TInputImage, class TOutputImage >
//void
//OpCannyEdgeDetectionImageFilter< TInputImage, TOutputImage >
//::Compute2ndDerivative(const int imageStride, const int imageWidth, 
//                       const int imageHeight, 
//                       const float* inputImage, float* outputImage, const int outStride) 
//{ 
//    const int kernelWidth = 3;
//    const int halfKernel = kernelWidth / 2;
//    
//    int startX  = 0;
//    int stopX   = imageWidth - 2 * halfKernel;
//    int startY  = 0;
//    int stopY   = imageHeight - 2 * halfKernel;
//
//      
//    #pragma omp parallel for shared (inputImage, outputImage) 
//    for (int y = startY; y < stopY; ++y) {
//      const register __m128 kdx0 = _mm_set_ps(0.5, 0.5, 0.5, 0.5);        
//      const register __m128 kdx2 = _mm_set_ps(-0.5, -0.5, -0.5, -0.5);    
//      
//      const register __m128 kdxx0 = _mm_set_ps(1.0, 1.0, 1.0, 1.0);       
//      const register __m128 kdxx1 = _mm_set_ps(-2.0, -2.0, -2.0, -2.0);   
//      
//      for (int x = startX; x < stopX; x += 4) {
//        __m128 dx_x, dx_y, dxx_x, dxx_y, dxy;
//        //dxy = _mm_setzero_ps();
//        
//        __m128 inv0 = _mm_load_ps(&inputImage[y * imageStride + x]);            
//        __m128 inv1 = _mm_load_ps(&inputImage[(y + 1) * imageStride + x]);   
//        __m128 inv2 = _mm_load_ps(&inputImage[(y + 2) * imageStride + x]);   
//        
//        register __m128 ktmp = _mm_setzero_ps();                             
//        
//        dx_x /*inv0 + 4*/ = _mm_load_ps(&inputImage[y * imageStride + x + 4]);               
//        dxx_x /*inv2 + 4*/= _mm_load_ps(&inputImage[(y + 2) * imageStride + x + 4]);         
//        
//        dx_y /*kdxy0*/ = _mm_set_ps(0.0, -0.25, 0.0, 0.25);    
//        dxx_y /*kdxy2*/ = _mm_set_ps(0.0, 0.25, 0.0, -0.25);   
//        
//        //A
//        dxy = _mm_dp113_ps(inv0, dx_y);
//        ROTATE_RIGHT(dx_y);                        
//        dxy += _mm_dp226_ps(inv0, dx_y);         
//        ROTATE_RIGHT_BLEND(dx_y, ktmp);                                         
//        dxy += _mm_dp196_ps(inv0, dx_y) + _mm_dp20_ps(dx_x, ktmp);
//        ROTATE_RIGHT(ktmp);                                                     
//        ROTATE_RIGHT_BLEND(dx_y, ktmp);                                         
//        dxy += _mm_dp136_ps(inv0, dx_y) + _mm_dp56_ps(dx_x, ktmp);                    
//         
//        ktmp = _mm_setzero_ps();                                              
//        dxy += _mm_dp113_ps(inv2, dxx_y);
//        ROTATE_RIGHT(dxx_y);                                                     
//        dxy += _mm_dp226_ps(inv2, dxx_y);                                   
//        ROTATE_RIGHT_BLEND(dxx_y, ktmp);                                         
//        dxy += _mm_dp196_ps(inv2, dxx_y) + _mm_dp20_ps(dxx_x, ktmp);      
//        ROTATE_RIGHT(ktmp);                                                     
//        ROTATE_RIGHT_BLEND(dxx_y, ktmp);                                         
//        dxy += _mm_dp136_ps(inv2, dxx_y) + _mm_dp56_ps(dxx_x, ktmp);
//        
//        //B
//        ktmp = inv0;
//        BLEND_ROTATE_LEFT(ktmp, dx_x);
//        
//        dx_y = kdx0 * ktmp;                                                   
//        dxx_y = kdxx0 * ktmp;                                                 
//        
//        //C
//        ktmp = inv1;
//        dx_x = _mm_load_ps(&inputImage[(y + 1) * imageStride + x + 4]);       
//        BLEND_ROTATE_LEFT(ktmp, dx_x);
//        
//        dxx_y += kdxx1 * ktmp;                                                
//        
//        //D
//        ktmp = inv2;
//        BLEND_ROTATE_LEFT(ktmp, dxx_x);
//        dx_y += kdx2 * ktmp;                                                  
//        dxx_y += kdxx0 * ktmp;                                                
//        
//        //E
//        register __m128 kdx = _mm_set_ps(0.0, 1.0, -2.0, 1.0);                  
//        ktmp = _mm_setzero_ps();                                                
//        dxx_x = _mm_dp113_ps(inv1, kdx);                                      
//        ROTATE_RIGHT(kdx);                                                     
//        dxx_x += _mm_dp226_ps(inv1, kdx);                                     
//        ROTATE_RIGHT_BLEND(kdx, ktmp);                                         
//        inv2 = dx_x;                                                            
//        dxx_x += _mm_dp196_ps(inv1, kdx) + _mm_dp20_ps(inv2, ktmp);
//        ROTATE_RIGHT(ktmp);                                                     
//        ROTATE_RIGHT_BLEND(kdx, ktmp);                                         
//        dxx_x += _mm_dp136_ps(inv1, kdx) + _mm_dp56_ps(inv2, ktmp);         
//        
//        //F
//        kdx = _mm_set_ps(0.0, -0.5, 0.0, 0.5);              
//        inv2 = dx_x;        PRINT_VECTOR(inv2);
//        dx_x = _mm_dp113_ps(inv1, kdx);
//        ROTATE_RIGHT(kdx);                                                     
//        dx_x += _mm_dp226_ps(inv1, kdx);
//        ROTATE_RIGHT_BLEND(kdx, ktmp);                                         
//        dx_x += _mm_dp196_ps(inv1, kdx) + _mm_dp20_ps(inv2, ktmp);
//        ROTATE_RIGHT(ktmp);                                                     
//        ROTATE_RIGHT_BLEND(kdx, ktmp);                                         
//        dx_x += _mm_dp136_ps(inv1, kdx) + _mm_dp56_ps(inv2, ktmp);
//            
//        //G
//        inv0 /*deriv*/ = _mm_set1_ps(2.0) * dx_y * dx_x * dxy;                  
//        
//        //H
//        dx_x = dx_x * dx_x;
//        dx_y = dx_y * dx_y;
//        
//        //I
//        inv0 += (dx_x * dxx_x) + (dx_y * dxx_y);                                
//        
//        //J
//        inv1 /*gradMag*/ = _mm_set1_ps(0.0001); 
//        inv1 += dx_x + dx_y;                                                    
//
//        //K
//        inv0 = inv0 / inv1;                                                     
//        
//        _mm_storeu_ps(&outputImage[(y + halfKernel) * outStride + (x + halfKernel)], inv0); 
//        
//            
//    }
//  }
//}


// Calculate the second derivative
template< class TInputImage, class TOutputImage >
void
OpCannyEdgeDetectionImageFilter< TInputImage, TOutputImage >
::Compute2ndDerivative(const int imageStride, const int imageWidth, 
                       const int imageHeight, 
                       const float* inputImage, float* outputImage, const int outStride) 
{ 
    const int kernelWidth = 3;  
    const int halfKernel = kernelWidth / 2;
    
    int startX  = 0;
    int stopX   = imageWidth - 2 * halfKernel;
    int startY  = 0;
    int stopY   = (imageHeight - 2 * halfKernel);
    
    #pragma omp parallel for shared (inputImage, outputImage) 
    for (int y = startY; y < stopY; ++y) {
      for (int x = startX; x < stopX; x += 4) { 
        register __m128 dx, dy, dxx, dyy, dxy;
        register __m128 kv0, kv1, kv2;  
        const register __m128 kdxy0 = _mm_set_ps(0.25, 0.25, -0.25, -0.25);  
        const register __m128 kdxy2 = _mm_set_ps(-0.25, -0.25, 0.25, 0.25);  
        register __m128 iv00, iv01, iv02, iv10, iv11, iv12;    
        
        const int y0 = (y + 0) * imageStride + x;
        const int y1 = (y + 1) * imageStride + x;
        const int y2 = (y + 2) * imageStride + x;
        
        iv00 = _mm_load_ps(&inputImage[y0]);  
        iv01 = _mm_load_ps(&inputImage[y1]); 
        iv02 = _mm_load_ps(&inputImage[y2]);  
  
        iv10 = _mm_load_ps(&inputImage[y0 + 4]);  
        iv11 = _mm_load_ps(&inputImage[y1 + 4]);  
        iv12 = _mm_load_ps(&inputImage[y2 + 4]);   
        
        //dx        
        kv0 = _mm_set_ps(0.5, 0.5, -0.5, -0.5);
        dx = _mm_dpil_ps(iv01, kv0);
        //dxx
        kv0 = _mm_set_ps(0.0, 1.0, -2.0, 1.0);
        dxx = _mm_dp113_ps(iv01, kv0); 
        //dxy
        dxy = _mm_vdpil_ps(iv00, kdxy0, iv02, kdxy2);
        
        BLEND_ROTATE31_LEFT(iv00, iv10, iv01, iv11, iv02, iv12)
        
        //dy
        kv0 = _mm_set_ps(0.5, 0.5, 0.5, 0.5);  
        kv2 = _mm_set_ps(-0.5, -0.5, -0.5, -0.5);   
        dy = _mm_v2dp_ps(iv00, kv0, iv02, kv2);
        //dyy
        kv1 = _mm_set_ps(-2.0, -2.0, -2.0, -2.0);  
        dyy = _mm_v3mdp_ps(iv00, iv01, iv02, kv1);
        //dxx
        kv0 = _mm_set_ps(0.0, 1.0, -2.0, 1.0);
        dxx += _mm_dp114_ps(iv01, kv0); 
  
        BLEND_ROTATE31_LEFT(iv00, iv10, iv01, iv11, iv02, iv12)
        //dx        
        kv0 = _mm_set_ps(0.5, 0.5, -0.5, -0.5);
        dx += _mm_dpih_ps(iv01, kv0);  
        //dxx
        kv0 = _mm_set_ps(0.0, 1.0, -2.0, 1.0); 
        dxx += _mm_dp116_ps(iv01, kv0); 
        //dxy
        dxy += _mm_vdpih_ps(iv00, kdxy0, iv02, kdxy2);
        
        BLEND_ROTATE1_LEFT(iv01, iv11)
        //dxx
        kv0 = _mm_set_ps(0.0, 1.0, -2.0, 1.0); 
        dxx += _mm_dp120_ps(iv01, kv0); 
        //calc
        _mm_storeu_ps(&outputImage[(y + halfKernel) * outStride + (x + halfKernel)], 
                      _mm_lvv_ps(dx, dy, dxx, dyy, dxy));  
    }
  }
}


 
template< class TInputImage, class TOutputImage >
void
OpCannyEdgeDetectionImageFilter< TInputImage, TOutputImage >
::Compute2ndDerivativePos() 
{
//    return;
  typename TInputImage::SizeType regionSize = 
  this->GetInput()->GetRequestedRegion().GetSize();
    
  const int imageWidth = regionSize[0];
  const int imageHeight = regionSize[1];
  const int imageStride = this->GetInput()->GetOffsetTable()[1]; 
  
  float* gaussianInput = m_GaussianBuffer->GetBufferPointer();
  float* dxInput = this->GetOutput()->GetBufferPointer();
  float* outputImage  = m_UpdateBuffer->GetBufferPointer();
  
//  printImage(imageWidth, imageHeight, imageStride, gaussianInput);
//  printImage(imageWidth, imageHeight, imageStride, dxInput);
  
  Compute2ndDerivativePos(imageStride, imageWidth, imageHeight, gaussianInput, dxInput, outputImage, imageStride);
//  printImage(imageWidth, imageHeight, imageStride, outputImage);

  float* bGaussianInput = m_BoundaryBuffer1->GetBufferPointer();  
  float* bDxInput = m_BoundaryBuffer2->GetBufferPointer(); 
  
  
  int stride = calculateAlignedStride (6, sizeof(InputImagePixelType), 16 ); 
  
  copy2DBoundaryChunk(gaussianInput, bGaussianInput,
                     stride, 6, imageHeight + 2, 
                     1, 1,
                     0, 1, 
                     imageStride, imageWidth, imageHeight);  
                      
  copy2DBoundaryChunk(dxInput, bDxInput,
                     stride, 6, imageHeight + 2, 
                     1, 1,
                     0, 1,
                     imageStride, imageWidth, imageHeight);   
                      
//  printImage(6, imageHeight + 2, stride, bGaussianInput);
//  printImage(6, imageHeight + 2, stride, bDxInput);
  
  Compute2ndDerivativePos(stride, 6, imageHeight + 2, bGaussianInput, bDxInput, outputImage - imageStride - 1, imageStride);
//  printImage(imageWidth, imageHeight, imageStride, outputImage);
   
  //right boundaries
  copy2DBoundaryChunk(gaussianInput, bGaussianInput,
                     stride, 6, imageHeight + 2, 
                     0, 1,
                     1, 1,
                     imageStride, imageWidth, imageHeight);  
  
  copy2DBoundaryChunk(dxInput, bDxInput,
                     stride, 6, imageHeight + 2, 
                     0, 1,
                     1, 1,
                     imageStride, imageWidth, imageHeight);  
  
//  printImage(6, imageHeight + 2, stride, bGaussianInput);
//  printImage(6, imageHeight + 2, stride, bDxInput);
  Compute2ndDerivativePos(stride, 6, imageHeight + 2, bGaussianInput, bDxInput, outputImage - imageStride + imageWidth - 5, imageStride);
//  printImage(imageWidth, imageHeight, imageStride, outputImage);
  
  
  stride = calculateAlignedStride (imageWidth + 2, sizeof(InputImagePixelType), 16 ); 
  
  //top boundaries
  copy2DBoundaryChunk(gaussianInput, bGaussianInput,
                     stride, imageWidth + 2, 3, 
                     1, 1,
                     1, 0,
                     imageStride, imageWidth, imageHeight);  
  
  copy2DBoundaryChunk(dxInput, bDxInput,
                     stride, imageWidth + 2, 3, 
                     1, 1,
                     1, 0, 
                     imageStride, imageWidth, imageHeight);  
  
//  printImage(imageWidth + 2, 3, stride, bGaussianInput);
//  printImage(imageWidth + 2, 3, stride, bDxInput);
  Compute2ndDerivativePos(stride, imageWidth + 2, 3, bGaussianInput, bDxInput, outputImage - imageStride - 1, imageStride);
//  printImage(imageWidth, imageHeight, imageStride, outputImage);
  
  //bottom boundaries
  copy2DBoundaryChunk(gaussianInput, bGaussianInput,
                     stride, imageWidth + 2, 3, 
                     1, 0,
                     1, 1,
                     imageStride, imageWidth, imageHeight);  
  
  copy2DBoundaryChunk(dxInput, bDxInput,
                     stride, imageWidth + 2, 3, 
                     1, 0,
                     1, 1,
                     imageStride, imageWidth, imageHeight);  
   
//  printImage(imageWidth + 2, 3, stride, bGaussianInput);
//  printImage(imageWidth + 2, 3, stride, bDxInput);
  Compute2ndDerivativePos(stride, imageWidth + 2, 3, bGaussianInput, bDxInput, outputImage + (imageHeight - 2) * imageStride - 1, imageStride);
//  printImage(imageWidth, imageHeight, imageStride, outputImage);
}

  
 
template< class TInputImage, class TOutputImage >
void
OpCannyEdgeDetectionImageFilter< TInputImage, TOutputImage >
::Compute2ndDerivativePos(const int imageStride, const int imageWidth, 
                       const int imageHeight, 
                       const float* lvInput, const float* lvvInput,
                       float* outputImage, const int outStride)
{

    const int kernelWidth = 3;
    const int halfKernel = kernelWidth / 2;
    
    int startX  = 0;
    int stopX   = imageWidth - 2 * halfKernel;
    int startY  = 0;
    int stopY   = imageHeight - 2 * halfKernel;

      
    #pragma omp parallel for shared (lvInput, lvvInput, outputImage) 
    for (int y = startY; y < stopY; ++y) {
      for (int x = startX; x < stopX; x += 4) { 
        register __m128 lx, ly, lvvx, lvvy, lv;
        const register __m128 kv0 = _mm_set_ps(0.5, 0.5, -0.5, -0.5);
        const register __m128 kv1 = _mm_set_ps(0.5, 0.5, 0.5, 0.5);
        const register __m128 kv2 = _mm_set_ps(-0.5, -0.5, -0.5, -0.5);
        __m128 iv00, iv01, iv02, iv10, iv11, iv12;   
        
        const int y0 = (y + 0) * imageStride + x;
        const int y1 = (y + 1) * imageStride + x;
        const int y2 = (y + 2) * imageStride + x;
        
        iv00 = _mm_load_ps(&lvInput[y0]); //_mm_prefetch(&lvInput[y0] + 128, _MM_HINT_T0);
        iv01 = _mm_load_ps(&lvInput[y1]); //_mm_prefetch(&lvInput[y1] + 128, _MM_HINT_T0);                                                   
        iv02 = _mm_load_ps(&lvInput[y2]); //_mm_prefetch(&lvInput[y2] + 128, _MM_HINT_T0);                                                        
        
        iv10 = _mm_load_ps(&lvInput[y0 + 4]);
        iv11 = _mm_load_ps(&lvInput[y1 + 4]);
        iv12 = _mm_load_ps(&lvInput[y2 + 4]);
        
        //lx
        lx = _mm_dpil_ps(iv01, kv0);   
        BLEND_ROTATE31_LEFT(iv00, iv10, iv01, iv11, iv02, iv12)
        //ly
        ly = _mm_v2dp_ps(iv00, kv1, iv02, kv2); 
        
        BLEND_ROTATE1_LEFT(iv01, iv11)
        lx += _mm_dpih_ps(iv01, kv0);   
        
        iv00 = _mm_load_ps(&lvvInput[y0]); //_mm_prefetch(&lvvInput[y0] + 128, _MM_HINT_T0);                                                   
        iv01 = _mm_load_ps(&lvvInput[y1]); //_mm_prefetch(&lvvInput[y1] + 128, _MM_HINT_T0);                                                    
        iv02 = _mm_load_ps(&lvvInput[y2]); //_mm_prefetch(&lvvInput[y2] + 128, _MM_HINT_T0);       
                                                          
        iv10 = _mm_load_ps(&lvvInput[y0 + 4]);                                                    
        iv11 = _mm_load_ps(&lvvInput[y1 + 4]);                                                    
        iv12 = _mm_load_ps(&lvvInput[y2 + 4]);                                                        
        //Lvvx                                  
        lvvx = _mm_dpil_ps(iv01, kv0);                                                                         
        BLEND_ROTATE31_LEFT(iv00, iv10, iv01, iv11, iv02, iv12) 
        //Lvvy                                                                                                
        lvvy = _mm_v2dp_ps(iv00, kv1, iv02, kv2); 

        BLEND_ROTATE1_LEFT(iv01, iv11) 
        //lvvx
        lvvx += _mm_dpih_ps(iv01, kv0);                                                                        
                
        PRINT_VECTOR_TRACE(ly);
        PRINT_VECTOR_TRACE(lx);
        PRINT_VECTOR_TRACE(lvvy);
        PRINT_VECTOR_TRACE(lvvx);

                                                                                                                                                                                                                            
        iv00 = lx * lx; 
        iv01 = ly * ly;
        lv = iv00 + iv01;
        lv = _mm_sqrt_ps (lv);
        iv00 = lvvx * (lx / lv);
        iv00 += lvvy * (ly / lv);
        iv00 = _mm_cmple_ps(iv00,  _mm_setzero_ps());
        iv00 = _mm_and_ps(iv00, _mm_set1_ps(0x00000001));
        iv00 *= lv;                                     
         
        _mm_storeu_ps(&outputImage[(y + halfKernel) * outStride + (x + halfKernel)], iv00);  
    }
  }
}

//
// 
//template< class TInputImage, class TOutputImage >
//void
//OpCannyEdgeDetectionImageFilter< TInputImage, TOutputImage >
//::Compute2ndDerivativePos(const int imageStride, const int imageWidth, 
//                       const int imageHeight, 
//                       const float* gaussianInput, const float* dxInput,
//                       float* outputImage, const int outStride)
//{
//
//    const int kernelWidth = 3;
//    const int halfKernel = kernelWidth / 2;
//    
//    int startX  = 0;
//    int stopX   = imageWidth - 2 * halfKernel;
//    int startY  = 0;
//    int stopY   = imageHeight - 2 * halfKernel;
//
//      
//    #pragma omp parallel for shared (gaussianInput, dxInput, outputImage) 
//    for (int y = startY; y < stopY; ++y) {
//      const register __m128 kdx0 = _mm_set_ps(0.5, 0.5, 0.5, 0.5);              
//      const register __m128 kdx2 = _mm_set_ps(-0.5, -0.5, -0.5, -0.5);          
//      
//      for (int x = startX; x < stopX; x += 4) {
//        
//        __m128 dx2dx_x, dx2dx_y, dxg_x, dxg_y;
//        
//        __m128 inv0 = _mm_load_ps(&dxInput[y * imageStride + x]);            
//        __m128 inv1 = _mm_load_ps(&dxInput[(y + 1) * imageStride + x]);      
//        __m128 inv2 = _mm_load_ps(&dxInput[(y + 2) * imageStride + x]);      
//        
//        __m128 inv0_4 = _mm_load_ps(&dxInput[y * imageStride + x + 4]);      
//        __m128 inv1_4 = _mm_load_ps(&dxInput[(y + 1) * imageStride + x + 4]);
//        __m128 inv2_4 = _mm_load_ps(&dxInput[(y + 2) * imageStride + x + 4]);
//        
//        register __m128 ktmp;
//        
//        ktmp = inv0;
//        BLEND_ROTATE_LEFT(ktmp, inv0_4);
//        dx2dx_y = kdx0 * ktmp;                                                  
//        
//        ktmp = inv2;
//        BLEND_ROTATE_LEFT(ktmp, inv2_4);
//        dx2dx_y += kdx2 * ktmp; 
//        
//        register __m128 kdx = _mm_set_ps(0.0, -0.5, 0.0, 0.5);
//        ktmp = _mm_setzero_ps();
//        dx2dx_x = _mm_dp113_ps(inv1, kdx);
//        ROTATE_RIGHT(kdx);                                                     
//        dx2dx_x += _mm_dp226_ps(inv1, kdx);
//        ROTATE_RIGHT_BLEND(kdx, ktmp);                                         
//        dx2dx_x += _mm_dp196_ps(inv1, kdx) + _mm_dp20_ps(inv1_4, ktmp);
//        ROTATE_RIGHT(ktmp);                                                     
//        ROTATE_RIGHT_BLEND(kdx, ktmp);                                         
//        dx2dx_x += _mm_dp136_ps(inv1, kdx) + _mm_dp56_ps(inv1_4, ktmp);
//
//        inv0 = _mm_load_ps(&gaussianInput[y * imageStride + x]);            
//        inv1 = _mm_load_ps(&gaussianInput[(y + 1) * imageStride + x]);      
//        inv2 = _mm_load_ps(&gaussianInput[(y + 2) * imageStride + x]);      
//        
//        inv0_4 = _mm_load_ps(&gaussianInput[y * imageStride + x + 4]);      
//        inv1_4 = _mm_load_ps(&gaussianInput[(y + 1) * imageStride + x + 4]);
//        inv2_4 = _mm_load_ps(&gaussianInput[(y + 2) * imageStride + x + 4]);
//                                                            
//        ktmp = _mm_setzero_ps();
//        
//        ktmp = inv0;
//        BLEND_ROTATE_LEFT(ktmp, inv0_4);
//        dxg_y = kdx0 * ktmp;
//        
//        ktmp = inv2;
//        BLEND_ROTATE_LEFT(ktmp, inv2_4);
//        dxg_y += kdx2 * ktmp;
//        
//        kdx = _mm_set_ps(0.0, -0.5, 0.0, 0.5);
//        ktmp = _mm_setzero_ps();
//        dxg_x = _mm_dp113_ps(inv1, kdx);
//        ROTATE_RIGHT(kdx);                                                     
//        dxg_x += _mm_dp226_ps(inv1, kdx);
//        ROTATE_RIGHT_BLEND(kdx, ktmp);                                         
//        dxg_x += _mm_dp196_ps(inv1, kdx) + _mm_dp20_ps(inv1_4, ktmp);
//        ROTATE_RIGHT(ktmp);                                                     
//        ROTATE_RIGHT_BLEND(kdx, ktmp);                                         
//        dxg_x += _mm_dp136_ps(inv1, kdx) + _mm_dp56_ps(inv1_4, ktmp);
// 
//        register __m128 gradMag = _mm_set1_ps(0.0001);
//        register __m128 derivPos = _mm_setzero_ps();
//         
//        gradMag += dxg_x * dxg_x;
//        gradMag += dxg_y * dxg_y;
//        
//        gradMag = _mm_sqrt_ps (gradMag);
//        
//        //calculate gradient of 2nd derivative
//        derivPos = dx2dx_x * (dxg_x / gradMag /*calculate the directional derivative*/); 
//        derivPos += dx2dx_y * (dxg_y / gradMag);
//        derivPos = _mm_cmple_ps(derivPos,  _mm_setzero_ps());
//        static const __m128 one = _mm_set1_ps(0x00000001);
//        derivPos = _mm_and_ps(derivPos, one);
//        derivPos = derivPos * gradMag;
//        _mm_storeu_ps(&outputImage[(y + halfKernel) * outStride + (x + halfKernel)], derivPos); 
//    }
//  }
//}
//

// Calculate the second derivative
template< class TInputImage, class TOutputImage >
void
OpCannyEdgeDetectionImageFilter< TInputImage, TOutputImage >
::ZeroCrossing() 
{
//    return;
  typename TInputImage::SizeType regionSize = 
  this->GetInput()->GetRequestedRegion().GetSize();
    
  const int imageWidth = regionSize[0];
  const int imageHeight = regionSize[1];
  const int imageStride = this->GetInput()->GetOffsetTable()[1]; 
  
  float* inputImage = this->GetOutput()->GetBufferPointer();
  float* outputImage  = m_GaussianBuffer->GetBufferPointer();
//  printImage(imageWidth, imageHeight, imageStride, inputImage);
  
  ZeroCrossing(imageStride, imageWidth, imageHeight, inputImage, outputImage, imageStride, 0, 0);
//  printImage(imageWidth, imageHeight, imageStride, outputImage);

  float* boundaryImage = m_BoundaryBuffer1->GetBufferPointer(); 
  
    
//void copy2DBoundaryChunk(const float* inBuffer, float* outBuffer,
//                           const int outStride, const int outWidth, const int outHeight, 
//                           const int replicateLeft, const int replicateTop,
//                           const int replicateRight, const int replicateBottom,
//                           const int inStride,  const int inWidth, const int inHeight);        

  int stride = calculateAlignedStride (imageWidth, sizeof(InputImagePixelType), 16 ); 
  
  //top boundaries
  copy2DBoundaryChunk(inputImage, boundaryImage,
                     stride, imageWidth, 3, 
                     0, 1,
                     0, 0,
                     imageStride, imageWidth, imageHeight);  
  
//  printImage(imageWidth + 8, 3, stride, boundaryImage);
  ZeroCrossing(stride, imageWidth, 3, boundaryImage, outputImage - imageStride, imageStride, 0, 0);
//  cout << "top boundaries" << endl;
//  printImage(imageWidth, imageHeight, imageStride, outputImage);
  
  //bottom boundaries
  copy2DBoundaryChunk(inputImage, boundaryImage,
                     stride, imageWidth, 3,  
                     0, 0,
                     0, 1,
                     imageStride, imageWidth, imageHeight);  
                     
  
//  printImage(imageWidth + 8, 3, stride, boundaryImage);
  ZeroCrossing(stride, imageWidth, 3, boundaryImage, outputImage + (imageHeight - 2) * imageStride, imageStride, 0, 0);
//  cout << "bottom boundaries" << endl;
  
  //left boundaries
  stride = calculateAlignedStride (12, sizeof(InputImagePixelType), 16 ); 
  
  copy2DBoundaryChunk(inputImage, boundaryImage,
                     stride, 12, imageHeight + 2, 
                     4, 1,
                     0, 1,
                     imageStride, imageWidth, imageHeight);  
                      
//  printImage(12, imageHeight + 2, stride, boundaryImage);
  ZeroCrossing(stride, 12, imageHeight + 2, boundaryImage, outputImage - imageStride - 4, imageStride, 4, 8);
//  cout << "left boundaries" << endl;
//  printImage(imageWidth, imageHeight, imageStride, outputImage);

  //right boundaries
  copy2DBoundaryChunk(inputImage, boundaryImage,
                     stride, 12, imageHeight + 2, 
                     0, 1,
                     4, 1,
                     imageStride, imageWidth, imageHeight);  
  
  
//  printImage(12, imageHeight + 2, stride, boundaryImage);
  ZeroCrossing(stride, 12, imageHeight + 2, boundaryImage, outputImage - imageStride + imageWidth - 8, imageStride, 4, 8);
//  cout << "right boundaries" << endl;
//  printImage(imageWidth, imageHeight, imageStride, outputImage); 
  
//  printImage(imageWidth, imageHeight, imageStride, outputImage);  
}
 

 

template< class TInputImage, class TOutputImage >
void
OpCannyEdgeDetectionImageFilter< TInputImage, TOutputImage >
::ZeroCrossing(const int imageStride, const int imageWidth, 
               const int imageHeight, 
               const float* inputImage, 
               float* outputImage, 
               const int outStride, 
               const int startX, 
               const int stopX)
{

    const int kernelWidth = 3;
    const int halfKernel = kernelWidth / 2;
    int sstopX   = stopX ? stopX : (imageWidth - 2 * halfKernel);
    int startY  = 0;
    int stopY   = imageHeight - 2 * halfKernel;
      
    #pragma omp parallel for shared (inputImage, outputImage) 
    for (int y = startY; y < stopY; ++y) {
      for (int x = startX; x < sstopX; x += 4) {
        static const __m128 sign = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
        static const __m128 one = _mm_set1_ps(0x00000001);
        static const __m128 abs = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
        
        __m128 thisOne = _mm_load_ps(&inputImage[(y + 1) * imageStride + x]);      
        __m128 that = _mm_load_ps(&inputImage[y * imageStride + x]);            
        __m128 absThat = _mm_and_ps(that, abs);
        __m128 absThis = _mm_and_ps(thisOne, abs);
        __m128 maskThis = _mm_and_ps(thisOne, sign);
        __m128 maskThat = _mm_and_ps(that, sign);
        
        maskThis = _mm_or_ps(maskThis, one);
        maskThat = _mm_or_ps(maskThat, one);
        
        maskThat = _mm_cmpneq_ps(maskThis, maskThat);

        __m128 lessThan = _mm_cmplt_ps(absThis, absThat);
        
        __m128 equal = _mm_cmpeq_ps(absThis, absThat);
        
        lessThan = _mm_or_ps(lessThan, equal);
        
        __m128 out = _mm_and_ps(lessThan, maskThat);
        
//        if (_mm_movemask_ps(out) == 0xf) {
//          out = _mm_and_ps(out, one);
//          _mm_storeu_ps(&outputImage[(y + 1) * outStride + x], out); 
//          continue;
//        }
        
        that = _mm_load_ps(&inputImage[(y + 2) * imageStride + x]);  
                
        maskThat = _mm_and_ps(that, sign);
        maskThat = _mm_or_ps(maskThat, one);
        
        maskThat = _mm_cmpneq_ps(maskThis, maskThat);
        maskThat = _mm_or_ps(maskThat, out);

        absThat = _mm_and_ps(that, abs);
//        absThis = _mm_and_ps(thisOne, abs);
              
        lessThan = _mm_cmplt_ps(absThis, absThat);
        
        equal = _mm_cmpeq_ps(absThis, absThat);
        
        lessThan = _mm_or_ps(lessThan, equal);
                
        maskThat = _mm_and_ps(lessThan, maskThat);
        
        out = _mm_or_ps(maskThat, out);
        
//        if (_mm_movemask_ps(out) == 0xf) {
//          out = _mm_and_ps(out, one);
//          _mm_storeu_ps(&outputImage[(y + 1) * outStride + x], out); 
//          continue;
//        }

        that = _mm_load_ps(&inputImage[(y + 1) * imageStride + x - 4]);  
//#ifdef __SSE4_1__
        that = _mm_blend_ps(thisOne, that, 8); 
//#else
//        ROTATE_RIGHT(thisOne);
//        ROTATE_RIGHT(that);
//        that = _mm_move_ss(thisOne, that); 
//        ROTATE_LEFT(thisOne);
//        ROTATE_LEFT(that);
//#endif        
        ROTATE_RIGHT(that);
        
        maskThat = _mm_and_ps(that, sign);
        maskThat = _mm_or_ps(maskThat, one);
        
        maskThat = _mm_cmpneq_ps(maskThis, maskThat);
        maskThat = _mm_or_ps(maskThat, out);

        absThat = _mm_and_ps(that, abs);
//        absThis = _mm_and_ps(thisOne, abs);
  
        lessThan = _mm_cmplt_ps(absThis, absThat);
        
        maskThat = _mm_and_ps(lessThan, maskThat);
        
        out = _mm_or_ps(maskThat, out);
        
        
//        if (_mm_movemask_ps(out) == 0xf) {
//          out = _mm_and_ps(out, one);
//          _mm_storeu_ps(&outputImage[(y + 1) * outStride + x], out); 
//          continue;
//        }
        
        that = _mm_load_ps(&inputImage[(y + 1) * imageStride + x + 4]);  
        that = _mm_move_ss(thisOne, that); 
        ROTATE_LEFT(that);
        maskThat = _mm_and_ps(that, sign);
        maskThat = _mm_or_ps(maskThat, one);
        
        maskThat = _mm_cmpneq_ps(maskThis, maskThat);
        maskThat = _mm_or_ps(maskThat, out);

        absThat = _mm_and_ps(that, abs);
//        absThis = _mm_and_ps(thisOne, abs);
              
        lessThan = _mm_cmplt_ps(absThis, absThat);
        
        maskThat = _mm_and_ps(lessThan, maskThat);
        
        out = _mm_or_ps(maskThat, out);
        out = _mm_and_ps(out, one);
        _mm_storeu_ps(&outputImage[(y + 1) * outStride + x], out); 
        
            
    }
  }
//  Compute2ndDerivativeBondaries(imageStride, imageWidth, 
//                                imageHeight, inputImage, inputImage,
//                                outputImage);      
}



template< class TInputImage, class TOutputImage >
void
OpCannyEdgeDetectionImageFilter< TInputImage, TOutputImage >
::HysteresisThresholding()
{
//  return;
  typename TInputImage::SizeType regionSize = 
        this->GetInput()->GetRequestedRegion().GetSize();
          
    int imageWidth = regionSize[0];
    int imageHeight = regionSize[1];

    float* inputImage  = m_GaussianBuffer->GetBufferPointer();
    
    float* outputImage = this->GetOutput()->GetBufferPointer();
    
    PRINT_LABEL ("HysteresisThresholding");


    const int imageStride = this->GetInput()->GetOffsetTable()[1]; 


    clear2DBuffer(outputImage, 
                  imageStride, 
                  imageHeight );
    
    clear2DBuffer(m_UpdateBuffer->GetBufferPointer(), 
                  imageStride, 
                  imageHeight );
    
    PRINT(imageWidth);
    
    int startX  = 0;
    int stopX   = imageWidth - 4;
    int startY  = 0;
    int stopY   = imageHeight;
    
//    cout << imageWidth << " x " << imageHeight << endl;

//    HysteresisEdgeIndex* buffer = (HysteresisEdgeIndex*)(m_UpdateBuffer->GetBufferPointer());
//    HysteresisQueue queue = HysteresisQueue(buffer, imageStride * imageHeight - 1);
      
    HysteresisEdgeIndex* buffer = (HysteresisEdgeIndex*)(m_UpdateBuffer->GetBufferPointer());
    int offset = (imageStride * imageHeight) / omp_get_max_threads();
    
    HysteresisQueue* queues = new HysteresisQueue[omp_get_max_threads()]; 
    
    for(int i = 0; i < omp_get_max_threads(); ++i) {
      queues[i] = HysteresisQueue(buffer, offset);
      buffer += offset;
    }
    
    #pragma omp parallel for shared (queues, inputImage, outputImage)  
    for (int y = startY; y < stopY; ++y) { 
        HysteresisQueue queue = queues[omp_get_thread_num()];
        const register __m128 upperThreshold = _mm_set1_ps(m_UpperThreshold);
      
      int x = startX;
      for (; x < stopX; x += 4) {
        
        __m128 value = _mm_load_ps(&inputImage[y * imageStride + x]);      PRINT_VECTOR(thisOne);
        
        unsigned int mask = _mm_movemask_ps(_mm_cmpgt_ps(value, upperThreshold)); 
        if (mask == 0x0) continue;
        
        if(mask & 1) //pixel 0  
        {
          queue.Enqueue(x, y);
          FollowEdge(imageStride, imageWidth, imageHeight, queue, inputImage, outputImage);
        } 
             
        if((mask & 2) >> 1) //pixel 1
        {
          queue.Enqueue(x + 1, y);
          FollowEdge(imageStride, imageWidth, imageHeight, queue, inputImage, outputImage);
        } 
             
        if((mask & 4) >> 2) //pixel 2
        {
          queue.Enqueue(x + 2, y);
          FollowEdge(imageStride, imageWidth, imageHeight, queue, inputImage, outputImage);
        } 
             
        if((mask & 8) >> 3) //pixel 3
        {
          queue.Enqueue(x + 3, y);
          FollowEdge(imageStride, imageWidth, imageHeight, queue, inputImage, outputImage);
        } 
      }
      
      x -= 3;
      //cout << x << endl;      
      for (; x < imageWidth; ++x) {
        if(inputImage[y * imageStride + x] > m_UpperThreshold) {
          queue.Enqueue(x, y);
          FollowEdge(imageStride, imageWidth, imageHeight, queue, inputImage, outputImage);
        }
      }
      
    }
//  printImage(imageWidth, imageHeight, imageStride, outputImage);  
    
}

template< class TInputImage, class TOutputImage >
inline void
OpCannyEdgeDetectionImageFilter< TInputImage, TOutputImage >
::FollowEdge(int imageStride, int imageWidth, int imageHeight, HysteresisQueue& queue,
             float* input, float* output)
{
    HysteresisEdgeIndex* idx = queue.Pick();
    
    if(output[idx->Y * imageStride + idx->X] == NumericTraits<OutputImagePixelType>::One )
    {
      // we must remove the node if we are not going to follow it!
      // Pop the front node from the list and read its index value.
      queue.Dequeue();
      return;
    }
    
    int x;
    int y;
    while(!queue.IsEmpty())
    {
    
        idx = queue.Dequeue();
        x = idx->X;
        y = idx->Y;
//        if (x >= imageWidth || y >= imageHeight) {
//          cout << x << " " << y << "; " << flush;
////          return;
//        }
        VerifyEdge(x, y, imageStride, imageWidth, imageHeight, queue,
                   input, output);
    }
}

template< class TInputImage, class TOutputImage >
inline void
OpCannyEdgeDetectionImageFilter< TInputImage, TOutputImage >
::VerifyEdge(int x, int y, int imageStride, int imageWidth, int imageHeight, HysteresisQueue& queue,
             float* input, float* output)
{
      int i;
//      cout << "x: " << x << " y: " << y << "; ";
      
      output[y * imageStride + x] = 1;

      if(x > 0) {
        i = y * imageStride + x - 1; //west
        if(input[i] > m_LowerThreshold && 
           output[i] != 1) {
            queue.Enqueue(x - 1, y);
            output[i] = 1;
        }
      }
      
      if(x > 0 && y > 0) {
        i = (y - 1) * imageStride + x - 1; //north west
        if(input[i] > m_LowerThreshold && 
           output[i] != 1) {
            queue.Enqueue(x - 1, y - 1);
            output[i] = 1;
        }
      }

      if(y > 0) {
        i = (y - 1) * imageStride + x; //north
        if(input[i] > m_LowerThreshold && 
           output[i] != 1) {
            queue.Enqueue(x, y - 1);
            output[i] = 1;
        }
      }

      if(x < imageWidth - 1 && y > 0) {
        i = (y - 1) * imageStride + x + 1; //north east
        if(input[i] > m_LowerThreshold && 
           output[i] != 1) {
            queue.Enqueue(x + 1, y - 1);
            output[i] = 1;
        }
      }
                        
      if(x < imageWidth - 1) {
        i = y * imageStride + x + 1; //east
        if(input[i] > m_LowerThreshold && 
           output[i] != 1) {
            queue.Enqueue(x + 1, y);
            output[i] = 1;
        }
      }
                        
      if(x < imageWidth - 1 && y < imageHeight - 1) {
        i = (y + 1) * imageStride + x + 1; //south east
        if(input[i] > m_LowerThreshold && 
           output[i] != 1) {
            queue.Enqueue(x + 1, y + 1);
            output[i] = 1;
        }
      }
                        
      if(y < imageHeight - 1) {
        i = (y + 1) * imageStride + x; //south
        if(input[i] > m_LowerThreshold && 
           output[i] != 1) {
            queue.Enqueue(x, y + 1);
            output[i] = 1;
        }
      }
                        
      if(x > 0 && y < imageHeight - 1) {
        i = (y + 1) * imageStride + x - 1; //south west
        if(input[i] > m_LowerThreshold && 
           output[i] != 1) {
            queue.Enqueue(x - 1, y + 1);
            output[i] = 1;
        }
      }
}

template< class TInputImage, class TOutputImage >
void
OpCannyEdgeDetectionImageFilter< TInputImage, TOutputImage >
::GaussianBlur (const InputImagePixelType* input, 
                float* output)
{
    typename TInputImage::SizeType regionSize = this->GetInput()->GetRequestedRegion().GetSize();
    int width = regionSize[0];
    int height = regionSize[1];
    
    float* kernel = gaussianKernel1D ();
    int kw = GetGaussianKernelWidth();
    int imageStride = calculateAlignedStride ( width, sizeof(InputImagePixelType), ALIGMENT_BYTES ); 
    
    opSeparableConvolve (imageStride, width, height, kw,
                         input, output, kernel, kernel);
//  printImage(width, height, imageStride, output);
                         
}
 


template< class TInputImage, class TOutputImage >
void
OpCannyEdgeDetectionImageFilter< TInputImage, TOutputImage >
:: Multiply( int stride, int height, 
             const float* __restrict input1, 
             const float* __restrict input2,  
             float* __restrict output )
{

  int stopY = height * stride;
  int startY  = 0;
  
//#ifdef __SSE4_1__
//  for (int y = startY; y < stopY; y += 4) {
//      __m128 inv0 = _mm_load_ps(&input1[y]);   PRINT_VECTOR(inv0);
//      __m128 inv1 = _mm_load_ps(&input2[y]);   PRINT_VECTOR(inv1);
//      _mm_stream_ps(&output[y], _mm_mul_ps(inv0, inv1));
//  }
//#else
  int i = 0;
  static float miniBuffer[31];
  for (int y = stopY - 31; y < stopY; ++y) {
    miniBuffer[i++] = input2[y];
  }
  #pragma omp parallel for
  for (int y = startY; y < stopY - 32; y += 32) {
    int idx = y;      
    __m128 a = _mm_load_ps(&input1[idx]);   PRINT_VECTOR(inv0);
    __m128 b = _mm_load_ps(&input2[idx]);   PRINT_VECTOR(inv1);
    __m128 inv0 = _mm_mul_ps(a, b);
    
    idx += 4;
    a = _mm_load_ps(&input1[idx]);   PRINT_VECTOR(inv0);
    b = _mm_load_ps(&input2[idx]);   PRINT_VECTOR(inv1);
    __m128 inv1 = _mm_mul_ps(a, b);
    
    idx += 4;
    a = _mm_load_ps(&input1[idx]);   PRINT_VECTOR(inv0);
    b = _mm_load_ps(&input2[idx]);   PRINT_VECTOR(inv1);
    __m128 inv2 = _mm_mul_ps(a, b);
    
    idx += 4;
    a = _mm_load_ps(&input1[idx]);   PRINT_VECTOR(inv0);
    b = _mm_load_ps(&input2[idx]);   PRINT_VECTOR(inv1);
    __m128 inv3 = _mm_mul_ps(a, b);

    idx += 4;
    a = _mm_load_ps(&input1[idx]);   PRINT_VECTOR(inv0);
    b = _mm_load_ps(&input2[idx]);   PRINT_VECTOR(inv1);
    __m128 inv4 = _mm_mul_ps(a, b);
    
    idx += 4;
    a = _mm_load_ps(&input1[idx]);   PRINT_VECTOR(inv0);
    b = _mm_load_ps(&input2[idx]);   PRINT_VECTOR(inv1);
    __m128 inv5 = _mm_mul_ps(a, b);
    
    idx += 4;
    a = _mm_load_ps(&input1[idx]);   PRINT_VECTOR(inv0);
    b = _mm_load_ps(&input2[idx]);   PRINT_VECTOR(inv1);
    __m128 inv6 = _mm_mul_ps(a, b);
    
    idx += 4;
    a = _mm_load_ps(&input1[idx]);   PRINT_VECTOR(inv0);
    b = _mm_load_ps(&input2[idx]);   PRINT_VECTOR(inv1);
    __m128 inv7 = _mm_mul_ps(a, b);

    _mm_store_ps(&output[y], inv0);
    _mm_store_ps(&output[y + 4], inv1);
    _mm_store_ps(&output[y + 8], inv2);
    _mm_store_ps(&output[y + 12], inv3);
    _mm_store_ps(&output[y + 16], inv4);
    _mm_store_ps(&output[y + 20], inv5);
    _mm_store_ps(&output[y + 24], inv6);
    _mm_store_ps(&output[y + 28], inv7);
  }
  
//    _mm_sfence();
  i = 0;
  for (int y = stopY - 31; y < stopY; ++y) {
    output[y] = input1[y] * miniBuffer[i++];  
  }    
//#endi6f    
    
}

//
//template< class TInputImage, class TOutputImage >
//void
//OpCannyEdgeDetectionImageFilter< TInputImage, TOutputImage >
//:: Multiply( int stride, int height, 
//             float* input1, float* input2,  float* output )
//{
//    int startY  = 0;
//    int stopY   = height * stride;
//    for (int y = startY; y < stopY; y += 4) {
//        __m128 inv0 = _mm_load_ps(&input1[y]);   PRINT_VECTOR(inv0);
//        __m128 inv1 = _mm_load_ps(&input2[y]);   PRINT_VECTOR(inv1);
//        _mm_stream_ps(&output[y], _mm_mul_ps(inv0, inv1));
//    }
//}
//

template <class TInputImage, class TOutputImage>
void 
OpCannyEdgeDetectionImageFilter<TInputImage,TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);

  os << "Variance: " << m_Variance << std::endl;
  os << "MaximumError: " << m_MaximumError << std::endl;
  os << indent << "Threshold: "
     << static_cast<typename NumericTraits<OutputImagePixelType>::PrintType>(m_Threshold)
     << std::endl;
  os << indent << "UpperThreshold: "
     << static_cast<typename NumericTraits<OutputImagePixelType>::PrintType>(m_UpperThreshold)
     << std::endl;
  os << indent << "LowerThreshold: "
     << static_cast<typename NumericTraits<OutputImagePixelType>::PrintType>(m_LowerThreshold)
     << std::endl;
  os << indent << "OutsideValue: "
     << static_cast<typename NumericTraits<OutputImagePixelType>::PrintType>(m_OutsideValue)
     << std::endl;
  os << "Center: "
     << m_Center << std::endl;
  os << "Stride: "
     << m_Stride << std::endl;
  os << "UpdateBuffer1: " << std::endl;
     m_UpdateBuffer->Print(os,indent.GetNextIndent());
}

}//end of itk namespace
#endif

















//
//
//
//// Calculate the second derivative
//template< class TInputImage, class TOutputImage >
//void
//OpCannyEdgeDetectionImageFilter< TInputImage, TOutputImage >
//::Compute2ndDerivative(const int imageStride, const int imageWidth, 
//                       const int imageHeight, 
//                       const float* inputImage, float* lvv, float* lvvv, const int outStride) 
//{ 
//    const int kernelWidth = 3;
//    const int halfKernel = kernelWidth / 2;
//    
//    int startX  = 0;
//    int stopX   = imageWidth - 2 * halfKernel;
//    int startY  = 2;
//    int stopY   = (imageHeight - 2 * halfKernel);
//    
//    #pragma omp parallel for shared (inputImage, lvv) 
//    for (int x = startX; x < stopX; x += 4) { 
//      __m128 dx, dy, dxx, dyy, dxy;
//      __m128 kv0, kv1, kv2;  
//      __m128 iv00, iv01, iv02, iv10, iv11, iv12;   
//      /*######################### [0,0] #####################*/
//      int y = 0;  
//      iv00 = _mm_load_ps(&inputImage[y * imageStride + x]);  
//      iv01 = _mm_load_ps(&inputImage[(y + 1) * imageStride + x]); 
//      iv02 = _mm_load_ps(&inputImage[(y + 2) * imageStride + x]);  
//
//      iv10 = _mm_load_ps(&inputImage[y * imageStride + 4 + x]);  
//      iv11 = _mm_load_ps(&inputImage[(y + 1) * imageStride + 4 + x]);  
//      iv12 = _mm_load_ps(&inputImage[(y + 2) * imageStride + 4 + x]);   
//      
//      //dx        
//      kv0 = _mm_set_ps(0.5, 0.5, -0.5, -0.5);
//      dx = _mm_dpil_ps(iv01, kv0);
//      //dxx
//      kv0 = _mm_set_ps(0.0, 1.0, -2.0, 1.0);
//      dxx = _mm_dp113_ps(iv01, kv0); 
//      //dxy
//      kv0 = _mm_set_ps(0.25, 0.25, -0.25, -0.25); kv2 = _mm_shuffle_ps(kv0, kv0, _MM_SHUFFLE(1, 0, 3, 2)); 
//      dxy = _mm_vdpil_ps(iv00, kv0, iv02, kv2);
//      
//      BLEND_ROTATE31_LEFT(iv00, iv10, iv01, iv11, iv02, iv12)
//      
//      //dy
//      kv0 = _mm_set_ps(0.5, 0.5, 0.5, 0.5);  kv2 = _mm_set_ps(-0.5, -0.5, -0.5, -0.5);   
//      dy = _mm_v2dp_ps(iv00, kv0, iv02, kv2);
//      //dyy
//      kv1 = _mm_set_ps(-2.0, -2.0, -2.0, -2.0);  
//      dyy = _mm_v3mdp_ps(iv00, iv01, iv02, kv1);
//      //dxx
//      kv0 = _mm_set_ps(0.0, 1.0, -2.0, 1.0);
//      dxx += _mm_dp114_ps(iv01, kv0); 
//
//      BLEND_ROTATE31_LEFT(iv00, iv10, iv01, iv11, iv02, iv12)
//      //dx        
//      kv0 = _mm_set_ps(0.5, 0.5, -0.5, -0.5);
//      dx += _mm_dpih_ps(iv01, kv0);  
//      //dxx
//      kv0 = _mm_set_ps(0.0, 1.0, -2.0, 1.0); 
//      dxx += _mm_dp116_ps(iv01, kv0); 
//      //dxy
//      kv0 = _mm_set_ps(0.25, 0.25, -0.25, -0.25); kv2 = _mm_shuffle_ps(kv0, kv0, _MM_SHUFFLE(1, 0, 3, 2)); 
//      dxy += _mm_vdpih_ps(iv00, kv0, iv02, kv2);
//      
//      BLEND_ROTATE31_LEFT(iv00, iv10, iv01, iv11, iv02, iv12)
//      //dxx
//      kv0 = _mm_set_ps(0.0, 1.0, -2.0, 1.0); 
//      dxx += _mm_dp120_ps(iv01, kv0); 
//      
//      BLEND_ROTATE31_LEFT(iv00, iv10, iv01, iv11, iv02, iv12)
//       
//      //calc
//      _mm_storeu_ps(&lvv[(y + halfKernel) * outStride + (x + halfKernel)], _mm_lvv_ps(dx, dy, dxx, dyy, dxy));  
//       
//                     
//      /*######################### [0,1] #####################*/
//      y = 1;        
//      iv00 = _mm_load_ps(&inputImage[(y + 0) * imageStride + x]);  
//      iv01 = _mm_load_ps(&inputImage[(y + 1) * imageStride + x]); 
//      iv02 = _mm_load_ps(&inputImage[(y + 2) * imageStride + x]);  
//
//      iv10 = _mm_load_ps(&inputImage[(y + 0) *  imageStride + 4 + x]);  
//      iv11 = _mm_load_ps(&inputImage[(y + 1) * imageStride + 4 + x]);  
//      iv12 = _mm_load_ps(&inputImage[(y + 2) * imageStride + 4 + x]);   
//      
//      //dx        
//      kv0 = _mm_set_ps(0.5, 0.5, -0.5, -0.5);
//      dx = _mm_dpil_ps(iv01, kv0);
//      //dxx
//      kv0 = _mm_set_ps(0.0, 1.0, -2.0, 1.0);
//      dxx = _mm_dp113_ps(iv01, kv0); 
//      //dxy
//      kv0 = _mm_set_ps(0.25, 0.25, -0.25, -0.25); kv2 = _mm_shuffle_ps(kv0, kv0, _MM_SHUFFLE(1, 0, 3, 2)); 
//      dxy = _mm_vdpil_ps(iv00, kv0, iv02, kv2);
//      
//      BLEND_ROTATE31_LEFT(iv00, iv10, iv01, iv11, iv02, iv12)
//      
//      //dy
//      kv0 = _mm_set_ps(0.5, 0.5, 0.5, 0.5);  kv2 = _mm_set_ps(-0.5, -0.5, -0.5, -0.5);   
//      dy = _mm_v2dp_ps(iv00, kv0, iv02, kv2);
//      //dyy
//      kv1 = _mm_set_ps(-2.0, -2.0, -2.0, -2.0);  
//      dyy = _mm_v3mdp_ps(iv00, iv01, iv02, kv1);
//      //dxx
//      kv0 = _mm_set_ps(0.0, 1.0, -2.0, 1.0);
//      dxx += _mm_dp114_ps(iv01, kv0); 
//
//      BLEND_ROTATE31_LEFT(iv00, iv10, iv01, iv11, iv02, iv12)
//      //dx        
//      kv0 = _mm_set_ps(0.5, 0.5, -0.5, -0.5);
//      dx += _mm_dpih_ps(iv01, kv0);  
//      //dxx
//      kv0 = _mm_set_ps(0.0, 1.0, -2.0, 1.0); 
//      dxx += _mm_dp116_ps(iv01, kv0); 
//      //dxy
//      kv0 = _mm_set_ps(0.25, 0.25, -0.25, -0.25); kv2 = _mm_shuffle_ps(kv0, kv0, _MM_SHUFFLE(1, 0, 3, 2)); 
//      dxy += _mm_vdpih_ps(iv00, kv0, iv02, kv2);
//      
//      BLEND_ROTATE1_LEFT(iv00, iv10)
//      //dxx
//      kv0 = _mm_set_ps(0.0, 1.0, -2.0, 1.0); 
//      dxx += _mm_dp120_ps(iv01, kv0); 
//      
//      //calc
//      _mm_storeu_ps(&lvv[(y + halfKernel) * outStride + (x + halfKernel)], _mm_lvv_ps(dx, dy, dxx, dyy, dxy));  
//
//
//    }        
//    
//    #pragma omp parallel for shared (inputImage, lvv, lvvv) 
//    for (int y = startY; y < stopY; ++y) {
//      for (int x = startX; x < stopX; x += 4) { 
//        __m128 dx, dy, dxx, dyy, dxy;
//        __m128 kv0, kv1, kv2;  
//        __m128 iv00, iv01, iv02, iv10, iv11, iv12;   
//        /*######################### [0,0] #####################*/
//        iv00 = _mm_load_ps(&inputImage[y * imageStride + x]);  
//        iv01 = _mm_load_ps(&inputImage[(y + 1) * imageStride + x]); 
//        iv02 = _mm_load_ps(&inputImage[(y + 2) * imageStride + x]);  
//  
//        iv10 = _mm_load_ps(&inputImage[y * imageStride + 4 + x]);  
//        iv11 = _mm_load_ps(&inputImage[(y + 1) * imageStride + 4 + x]);  
//        iv12 = _mm_load_ps(&inputImage[(y + 2) * imageStride + 4 + x]);   
//        
//        //dx        
//        kv0 = _mm_set_ps(0.5, 0.5, -0.5, -0.5);
//        dx = _mm_dpil_ps(iv01, kv0);
//        //dxx
//        kv0 = _mm_set_ps(0.0, 1.0, -2.0, 1.0);
//        dxx = _mm_dp113_ps(iv01, kv0); 
//        //dxy
//        kv0 = _mm_set_ps(0.25, 0.25, -0.25, -0.25); kv2 = _mm_shuffle_ps(kv0, kv0, _MM_SHUFFLE(1, 0, 3, 2)); 
//        dxy = _mm_vdpil_ps(iv00, kv0, iv02, kv2);
//        
//        BLEND_ROTATE31_LEFT(iv00, iv10, iv01, iv11, iv02, iv12)
//        
//        //dy
//        kv0 = _mm_set_ps(0.5, 0.5, 0.5, 0.5);  kv2 = _mm_set_ps(-0.5, -0.5, -0.5, -0.5);   
//        dy = _mm_v2dp_ps(iv00, kv0, iv02, kv2);
//        //dyy
//        kv1 = _mm_set_ps(-2.0, -2.0, -2.0, -2.0);  
//        dyy = _mm_v3mdp_ps(iv00, iv01, iv02, kv1);
//        //dxx
//        kv0 = _mm_set_ps(0.0, 1.0, -2.0, 1.0);
//        dxx += _mm_dp114_ps(iv01, kv0); 
//  
//        BLEND_ROTATE31_LEFT(iv00, iv10, iv01, iv11, iv02, iv12)
//        //dx        
//        kv0 = _mm_set_ps(0.5, 0.5, -0.5, -0.5);
//        dx += _mm_dpih_ps(iv01, kv0);  
//        //dxx
//        kv0 = _mm_set_ps(0.0, 1.0, -2.0, 1.0); 
//        dxx += _mm_dp116_ps(iv01, kv0); 
//        //dxy
//        kv0 = _mm_set_ps(0.25, 0.25, -0.25, -0.25); kv2 = _mm_shuffle_ps(kv0, kv0, _MM_SHUFFLE(1, 0, 3, 2)); 
//        dxy += _mm_vdpih_ps(iv00, kv0, iv02, kv2);
//        
//        BLEND_ROTATE1_LEFT(iv01, iv11)
//        //dxx
//        kv0 = _mm_set_ps(0.0, 1.0, -2.0, 1.0); 
//        dxx += _mm_dp120_ps(iv01, kv0); 
//        //calc
//        _mm_storeu_ps(&lvv[(y + halfKernel) * outStride + (x + halfKernel)], _mm_lvv_ps(dx, dy, dxx, dyy, dxy));  
//        
//
//        PRINT_TRACE("Derradero ever"); 
//  
//        iv00 = _mm_load_ps(&inputImage[(y - 2) * imageStride + x]);                                                    
//        iv01 = _mm_load_ps(&inputImage[(y - 1) * imageStride + x]);                                                    
//        iv02 = _mm_load_ps(&inputImage[(y - 0) * imageStride + x]);                                                         
//        
//        iv10 = _mm_load_ps(&inputImage[(y - 2) * imageStride + x + 4]);                                                    
//        iv11 = _mm_load_ps(&inputImage[(y - 1) * imageStride + x + 4]);                                                    
//        iv12 = _mm_load_ps(&inputImage[(y - 0) * imageStride + x + 4]);                                                       
//        
//        PRINT_VECTOR_TRACE(iv00);
//        PRINT_VECTOR_TRACE(iv01); 
//        PRINT_VECTOR_TRACE(iv02);  
//        
//        PRINT_VECTOR_TRACE(iv10);
//        PRINT_VECTOR_TRACE(iv11);//dx        
//        PRINT_VECTOR_TRACE(iv12); 
//                                                                            
//        
//        //dx
//        kv0 = _mm_set_ps(0.5, 0.5, -0.5, -0.5);
//        dx = _mm_dpil_ps(iv01, kv0);   
//        BLEND_ROTATE31_LEFT(iv00, iv10, iv01, iv11, iv02, iv12)
//        //dy
//        kv0 = _mm_set_ps(0.5, 0.5, 0.5, 0.5);  kv2 = _mm_set_ps(-0.5, -0.5, -0.5, -0.5);   
//        dy = _mm_v2dp_ps(iv00, kv0, iv02, kv2); 
//        
//        BLEND_ROTATE1_LEFT(iv01, iv11)
//        kv0 = _mm_set_ps(0.5, 0.5, -0.5, -0.5);
//        dx += _mm_dpih_ps(iv01, kv0);   
//        
//        iv00 = _mm_load_ps(&lvv[(y - 2) * imageStride + x]);                                                    
//        iv01 = _mm_load_ps(&lvv[(y - 1) * imageStride + x]);                                                    
//        iv02 = _mm_load_ps(&lvv[(y - 0) * imageStride + x]);                                                         
//        iv10 = _mm_load_ps(&lvv[(y - 2) * imageStride + x + 4]);                                                    
//        iv11 = _mm_load_ps(&lvv[(y - 1) * imageStride + x + 4]);                                                    
//        iv12 = _mm_load_ps(&lvv[(y - 0) * imageStride + x + 4]);                                                        
//
//        PRINT_VECTOR_TRACE(iv00);
//        PRINT_VECTOR_TRACE(iv01); 
//        PRINT_VECTOR_TRACE(iv02);  
//        
//        PRINT_VECTOR_TRACE(iv10);
//        PRINT_VECTOR_TRACE(iv11);//dx        
//        PRINT_VECTOR_TRACE(iv12);                 
//        
//        //Lvvx                                  
//        kv0 = _mm_set_ps(0.5, 0.5, -0.5, -0.5);                                                              
//        dxx = _mm_dpil_ps(iv01, kv0);                                                                         
//        BLEND_ROTATE31_LEFT(iv00, iv10, iv01, iv11, iv02, iv12) 
//        //Lvvy                                                                                                
//        kv0 = _mm_set_ps(0.5, 0.5, 0.5, 0.5);  kv2 = _mm_set_ps(-0.5, -0.5, -0.5, -0.5);   
//        dyy = _mm_v2dp_ps(iv00, kv0, iv02, kv2); 
//
//        BLEND_ROTATE1_LEFT(iv01, iv11) 
//        //dxx
//        kv0 = _mm_set_ps(0.5, 0.5, -0.5, -0.5);                                                              
//        dxx += _mm_dpih_ps(iv01, kv0);                                                                        
//                
//        PRINT_VECTOR_TRACE(dy);
//        PRINT_VECTOR_TRACE(dx);
//        PRINT_VECTOR_TRACE(dyy);
//        PRINT_VECTOR_TRACE(dxx);
//
//                                                                                                                                                                                                                            
//        dxy = dx * dx + dy * dy;                                                                              PRINT_VECTOR_TRACE(dxy);
//        dxy = _mm_sqrt_ps (dxy);                                                                              PRINT_VECTOR_TRACE(dxy);
//        //calculate gradient of 2nd derivative                                                                
//        /*calculate the directional derivative*/                                                              
//        iv00 /*derivPos*/ = dxx * (dx / dxy);                                                                 PRINT_VECTOR_TRACE(iv00);
//        iv00 += dyy * (dy / dxy);                                                                             PRINT_VECTOR_TRACE(iv00);
//        iv00 = _mm_cmple_ps(iv00,  _mm_setzero_ps());                                                         PRINT_VECTOR_TRACE(iv00); 
//        iv00 = _mm_and_ps(iv00, _mm_set1_ps(0x00000001));                                                     PRINT_VECTOR_TRACE(iv00);
//        iv00 *= dxy;                                                                                          PRINT_VECTOR_TRACE(iv00);  
//         
//        _mm_storeu_ps(&lvvv[(y - 2 + halfKernel) * outStride + (x + halfKernel)], iv00);  
//    }
//  }
//}
