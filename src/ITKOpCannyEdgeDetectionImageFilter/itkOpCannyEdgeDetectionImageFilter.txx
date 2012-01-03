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
#include "Timer.h"

#include <omp.h>
#include <smmintrin.h>
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
          
    int nWidth = regionSize[0];
    int nHeight = regionSize[1];
    int stride = input->GetOffsetTable()[1];
    
  this->AllocateUpdateBuffer();
    
    
    // allocate 16 byte aligned buffers needed by SSE
   // float*  buffer1 = AllocateAlignedBuffer (nWidth, nHeight);

    // blur the image to reduce noise    
//    GaussianBlur ( input->GetBufferPointer(), 
//                   this->GetOutput(0)->GetBufferPointer() );
  //printImage(nWidth, nHeight, nWidth, input->GetBufferPointer());    
  
  m_Timer.start();  
    this->GaussianBlur ( input->GetBufferPointer(), m_GaussianBuffer->GetBufferPointer() );
    
//  cout << "Gaussian OpCanny" << endl;
//  gaussianKernel1D();
//  printImage(nWidth, nHeight, stride, m_GaussianBuffer->GetBufferPointer());    
  //2. Calculate 2nd order directional derivative-------
  // Calculate the 2nd order directional derivative of the smoothed image.
  // The output of this filter will be used to store the directional
  // derivative.
//  cout << "Compute2ndDerivative OpCanny" << endl;
  this->Compute2ndDerivative();    
//  printImage(nWidth, nHeight, stride, this->GetOutput()->GetBufferPointer());     
  
//  cout << "Compute2ndDerivativePos OpCanny" << endl;
  this->Compute2ndDerivativePos();    
//  printImage(nWidth, nHeight, stride, m_UpdateBuffer->GetBufferPointer());     
//  printImage(nWidth, nHeight, nWidth, m_UpdateBuffer->GetBufferPointer());    


//  cout << "ZeroCrossing OpCanny" << endl;
  this->ZeroCrossing();    
//  printImage(nWidth, nHeight, stride, m_GaussianBuffer->GetBufferPointer());     
//
//  cout << "Multiply OpCanny" << endl;
  this->Multiply(stride, nHeight,  
                 m_UpdateBuffer->GetBufferPointer(), 
                 m_GaussianBuffer->GetBufferPointer(), 
                 m_GaussianBuffer->GetBufferPointer());    
                 
  // printImage(nWidth, nHeight, stride, m_GaussianBuffer->GetBufferPointer());     
                 
                 
  this->HysteresisThresholding();                    
  //printImage(nWidth, nHeight, stride, this->GetOutput()->GetBufferPointer());     
                 
  m_Timer.stop();
//  printImage(nWidth, nHeight, stride, m_GaussianBuffer->GetBufferPointer());     

}


template< class TInputImage, class TOutputImage >
int 
OpCannyEdgeDetectionImageFilter< TInputImage, TOutputImage >
::CalculateAlignedStride (int width, int height, int typeSize, int alignInBytes)
{
 
 if(width < alignInBytes) return alignInBytes;

    //TODO: see if cache line size align works and is faster
    int wBytes = width * typeSize;
    int sPixel = typeSize;
    //http://stackoverflow.com/questions/1855896/memory-alignment-on-modern-processors
    //http://cboard.cprogramming.com/c-programming/105136-memory-granularity-processor.html
    //((width * bytesPerPixel) + 3) & ~3; /* Aligned to 4 bytes */
    //int bytesPerPixel = typeSize;
    //int a = ((width * bytesPerPixel) + alignInBytes) & ~alignInBytes; /* Aligned to 4 bytes */
    //PRINT(a/sPixel); 
    //return a;
    //PRINT((wBytes + alignInBytes - (wBytes % alignInBytes)) / sPixel);
    return wBytes % alignInBytes == 0 ? width : (wBytes + alignInBytes - (wBytes % alignInBytes)) / sPixel;
}


// Calculate the second derivative
template< class TInputImage, class TOutputImage >
void
OpCannyEdgeDetectionImageFilter< TInputImage, TOutputImage >
::Compute2ndDerivative() 
{

    typename TInputImage::SizeType regionSize = 
        this->GetInput()->GetRequestedRegion().GetSize();
          
    int imageWidth = regionSize[0];
    int imageHeight = regionSize[1];

    float* inputImage  = m_GaussianBuffer->GetBufferPointer(); 
    float* outputImage = this->GetOutput()->GetBufferPointer();
    
    const int kernelWidth = 3;
    const int halfKernel = kernelWidth / 2;
    const int imageStride = this->GetInput()->GetOffsetTable()[1];
    
    int startX  = 0;
    int stopX   = imageWidth - 2 * halfKernel;
    int startY  = 0;
    int stopY   = imageHeight - 2 * halfKernel;

      
    //#pragma omp parallel for shared (inputImage, outputImage) 
    for (int y = startY; y < stopY; ++y) {
      const register __m128 kdx0 = _mm_set_ps(0.5, 0.5, 0.5, 0.5);        
      const register __m128 kdx2 = _mm_set_ps(-0.5, -0.5, -0.5, -0.5);    
      
      const register __m128 kdxx0 = _mm_set_ps(1.0, 1.0, 1.0, 1.0);       
      const register __m128 kdxx1 = _mm_set_ps(-2.0, -2.0, -2.0, -2.0);   
      
      for (int x = startX; x < stopX; x += 4) {
        __m128 dx_x, dx_y, dxx_x, dxx_y, dxy;
        dxy = _mm_setzero_ps();
        
        __m128 inv0 = _mm_load_ps(&inputImage[y * imageStride + x]);            
        __m128 inv1 = _mm_load_ps(&inputImage[(y + 1) * imageStride + x]);   
        __m128 inv2 = _mm_load_ps(&inputImage[(y + 2) * imageStride + x]);   
        
        register __m128 ktmp = _mm_setzero_ps();                             
        
        dx_x /*inv0 + 4*/ = _mm_load_ps(&inputImage[y * imageStride + x + 4]);               
        dxx_x /*inv2 + 4*/= _mm_load_ps(&inputImage[(y + 2) * imageStride + x + 4]);         
        
        dx_y /*kdxy0*/ = _mm_set_ps(0.0, -0.25, 0.0, 0.25);    
        dxx_y /*kdxy2*/ = _mm_set_ps(0.0, 0.25, 0.0, -0.25);   
        
        //A
        dxy = _mm_dp_ps(inv0, dx_y, 113);          
        ROTATE_RIGHT(dx_y);                        
        dxy += _mm_dp_ps(inv0, dx_y, 226);         
        ROTATE_RIGHT_BLEND(dx_y, ktmp);                                         
        dxy += _mm_dp_ps(inv0, dx_y, 196) + _mm_dp_ps(dx_x, ktmp, 20);         
        ROTATE_RIGHT(ktmp);                                                     
        ROTATE_RIGHT_BLEND(dx_y, ktmp);                                         
        dxy += _mm_dp_ps(inv0, dx_y, 136) + _mm_dp_ps(dx_x, ktmp, 56);                    
         
        ktmp = _mm_setzero_ps();                                              
        dxy += _mm_dp_ps(inv2, dxx_y, 113);                                   
        ROTATE_RIGHT(dxx_y);                                                     
        dxy += _mm_dp_ps(inv2, dxx_y, 226);                                   
        ROTATE_RIGHT_BLEND(dxx_y, ktmp);                                         
        dxy += _mm_dp_ps(inv2, dxx_y, 196) + _mm_dp_ps(dxx_x, ktmp, 20);      
        ROTATE_RIGHT(ktmp);                                                     
        ROTATE_RIGHT_BLEND(dxx_y, ktmp);                                         
        dxy += _mm_dp_ps(inv2, dxx_y, 136) + _mm_dp_ps(dxx_x, ktmp, 56);      
        
        //B
        ktmp = inv0;
        BLEND_ROTATE_LEFT(ktmp, dx_x);
        
        dx_y = kdx0 * ktmp;                                                   
        dxx_y = kdxx0 * ktmp;                                                 
        
        //C
        ktmp = inv1;
        dx_x = _mm_load_ps(&inputImage[(y + 1) * imageStride + x + 4]);       
        BLEND_ROTATE_LEFT(ktmp, dx_x);
        
        dxx_y += kdxx1 * ktmp;                                                
        
        //D
        ktmp = inv2;
        BLEND_ROTATE_LEFT(ktmp, dxx_x);
        dx_y += kdx2 * ktmp;                                                  
        dxx_y += kdxx0 * ktmp;                                                
        
        //E
        register __m128 kdx = _mm_set_ps(0.0, 1.0, -2.0, 1.0);                  
        ktmp = _mm_setzero_ps();                                                
        dxx_x = _mm_dp_ps(inv1, kdx, 113);                                      
        ROTATE_RIGHT(kdx);                                                     
        dxx_x += _mm_dp_ps(inv1, kdx, 226);                                     
        ROTATE_RIGHT_BLEND(kdx, ktmp);                                         
        inv2 = dx_x;                                                            
        dxx_x += _mm_dp_ps(inv1, kdx, 196) + _mm_dp_ps(inv2, ktmp, 20);         
        ROTATE_RIGHT(ktmp);                                                     
        ROTATE_RIGHT_BLEND(kdx, ktmp);                                         
        dxx_x += _mm_dp_ps(inv1, kdx, 136) + _mm_dp_ps(inv2, ktmp, 56);         
        
        //F
        kdx = _mm_set_ps(0.0, -0.5, 0.0, 0.5);              
        inv2 = dx_x;        PRINT_VECTOR(inv2);
        dx_x = _mm_dp_ps(inv1, kdx, 113);                   
        ROTATE_RIGHT(kdx);                                                     
        dx_x += _mm_dp_ps(inv1, kdx, 226);                  
        ROTATE_RIGHT_BLEND(kdx, ktmp);                                         
        dx_x += _mm_dp_ps(inv1, kdx, 196) + _mm_dp_ps(inv2, ktmp, 20);
        ROTATE_RIGHT(ktmp);                                                     
        ROTATE_RIGHT_BLEND(kdx, ktmp);                                         
        dx_x += _mm_dp_ps(inv1, kdx, 136) + _mm_dp_ps(inv2, ktmp, 56);
            
        //G
        inv0 /*deriv*/ = _mm_set1_ps(2.0) * dx_y * dx_x * dxy;                  
        
        //H
        dx_x = dx_x * dx_x;
        dx_y = dx_y * dx_y;
        
        //I
        inv0 += (dx_x * dxx_x) + (dx_y * dxx_y);                                
        
        //J
        inv1 /*gradMag*/ = _mm_set1_ps(0.0001); 
        inv1 += dx_x + dx_y;                                                    

        //K
        inv0 = inv0 / inv1;                                                     
        
        _mm_storeu_ps(&outputImage[(y + halfKernel) * imageStride + (x + halfKernel)], inv0); 
        
            
    }
  }
  
  Compute2ndDerivativeBondaries(imageStride, imageWidth, 
                                imageHeight, inputImage, inputImage,
                                outputImage);    
}

// Calculate the second derivative
template< class TInputImage, class TOutputImage >
void
OpCannyEdgeDetectionImageFilter< TInputImage, TOutputImage >
::Compute2ndDerivativeBondaries(const int imageStride, const int imageWidth, 
                                const int imageHeight, 
                                const float* gaussianImage, const float* dxImage, 
                                float* outputImage) {
  
        for(int x = 0; x < imageWidth; ++x) {
          if(x == 0){
          
          }  
          else if(x == imageWidth - 1){
          
          }  
          outputImage[x] = 0;
          outputImage[(imageHeight - 1) * imageStride + x] = 0;
        }
        
        for(int y = 1; y < imageHeight - 1; ++y) {
          if(y == 0){
          
          }  
          else if(y == imageHeight - 1){
          
          }  
          outputImage[y * imageStride] = 0;
          outputImage[y * imageStride + (imageWidth - 1)] = 0;
        }
}



template< class TInputImage, class TOutputImage >
void
OpCannyEdgeDetectionImageFilter< TInputImage, TOutputImage >
::Compute2ndDerivativePos()
{

  typename TInputImage::SizeType regionSize = 
        this->GetInput()->GetRequestedRegion().GetSize();
          
    int imageWidth = regionSize[0];
    int imageHeight = regionSize[1];

    float* gaussianInput = m_GaussianBuffer->GetBufferPointer();
    float* dxInput = this->GetOutput()->GetBufferPointer();
    float* outputImage  = m_UpdateBuffer->GetBufferPointer();
    
    const int kernelWidth = 3;
    const int halfKernel = kernelWidth / 2;
    const int imageStride = this->GetInput()->GetOffsetTable()[1];
    
    int startX  = 0;
    int stopX   = imageWidth - 2 * halfKernel;
    int startY  = 0;
    int stopY   = imageHeight - 2 * halfKernel;

      
    #pragma omp parallel for shared (gaussianInput, outputImage) 
    for (int y = startY; y < stopY; ++y) {
      const register __m128 kdx0 = _mm_set_ps(0.5, 0.5, 0.5, 0.5);              
      const register __m128 kdx2 = _mm_set_ps(-0.5, -0.5, -0.5, -0.5);          
      
      for (int x = startX; x < stopX; x += 4) {
        
        __m128 dx2dx_x, dx2dx_y, dxg_x, dxg_y;
        
        __m128 inv0 = _mm_load_ps(&dxInput[y * imageStride + x]);            
        __m128 inv1 = _mm_load_ps(&dxInput[(y + 1) * imageStride + x]);      
        __m128 inv2 = _mm_load_ps(&dxInput[(y + 2) * imageStride + x]);      
        
        __m128 inv0_4 = _mm_load_ps(&dxInput[y * imageStride + x + 4]);      
        __m128 inv1_4 = _mm_load_ps(&dxInput[(y + 1) * imageStride + x + 4]);
        __m128 inv2_4 = _mm_load_ps(&dxInput[(y + 2) * imageStride + x + 4]);
        
        register __m128 ktmp;
        
        ktmp = inv0;
        BLEND_ROTATE_LEFT(ktmp, inv0_4);
        dx2dx_y = kdx0 * ktmp;                                                  
        
        ktmp = inv2;
        BLEND_ROTATE_LEFT(ktmp, inv2_4);
        dx2dx_y += kdx2 * ktmp; 
        
        register __m128 kdx = _mm_set_ps(0.0, -0.5, 0.0, 0.5);
        ktmp = _mm_setzero_ps();
        dx2dx_x = _mm_dp_ps(inv1, kdx, 113);
        ROTATE_RIGHT(kdx);                                                     
        dx2dx_x += _mm_dp_ps(inv1, kdx, 226);
        ROTATE_RIGHT_BLEND(kdx, ktmp);                                         
        dx2dx_x += _mm_dp_ps(inv1, kdx, 196) + _mm_dp_ps(inv1_4, ktmp, 20);
        ROTATE_RIGHT(ktmp);                                                     
        ROTATE_RIGHT_BLEND(kdx, ktmp);                                         
        dx2dx_x += _mm_dp_ps(inv1, kdx, 136) + _mm_dp_ps(inv1_4, ktmp, 56);

        inv0 = _mm_load_ps(&gaussianInput[y * imageStride + x]);            
        inv1 = _mm_load_ps(&gaussianInput[(y + 1) * imageStride + x]);      
        inv2 = _mm_load_ps(&gaussianInput[(y + 2) * imageStride + x]);      
        
        inv0_4 = _mm_load_ps(&gaussianInput[y * imageStride + x + 4]);      
        inv1_4 = _mm_load_ps(&gaussianInput[(y + 1) * imageStride + x + 4]);
        inv2_4 = _mm_load_ps(&gaussianInput[(y + 2) * imageStride + x + 4]);
                                                            
        ktmp = _mm_setzero_ps();
        
        ktmp = inv0;
        BLEND_ROTATE_LEFT(ktmp, inv0_4);
        dxg_y = kdx0 * ktmp;
        
        ktmp = inv2;
        BLEND_ROTATE_LEFT(ktmp, inv2_4);
        dxg_y += kdx2 * ktmp;
        
        kdx = _mm_set_ps(0.0, -0.5, 0.0, 0.5);
        ktmp = _mm_setzero_ps();
        dxg_x = _mm_dp_ps(inv1, kdx, 113);
        ROTATE_RIGHT(kdx);                                                     
        dxg_x += _mm_dp_ps(inv1, kdx, 226);
        ROTATE_RIGHT_BLEND(kdx, ktmp);                                         
        dxg_x += _mm_dp_ps(inv1, kdx, 196) + _mm_dp_ps(inv1_4, ktmp, 20);
        ROTATE_RIGHT(ktmp);                                                     
        ROTATE_RIGHT_BLEND(kdx, ktmp);                                         
        dxg_x += _mm_dp_ps(inv1, kdx, 136) + _mm_dp_ps(inv1_4, ktmp, 56);
 
        register __m128 gradMag = _mm_set1_ps(0.0001);
        register __m128 derivPos = _mm_setzero_ps();
         
        gradMag += dxg_x * dxg_x;
        gradMag += dxg_y * dxg_y;
        
        gradMag = _mm_sqrt_ps (gradMag);
        
        //calculate gradient of 2nd derivative
        derivPos = dx2dx_x * (dxg_x / gradMag /*calculate the directional derivative*/); 
        derivPos += dx2dx_y * (dxg_y / gradMag);
        derivPos = _mm_cmple_ps(derivPos,  _mm_setzero_ps());
        static const __m128 one = _mm_set1_ps(0x00000001);
        derivPos = _mm_and_ps(derivPos, one);
        derivPos = derivPos * gradMag;
        _mm_storeu_ps(&outputImage[(y + halfKernel) * imageStride + (x + halfKernel)], derivPos); 
    }
  }
  
//  Compute2ndDerivativePosBondaries(imageStride, imageWidth, 
//                                   imageHeight, dxInput, gaussianInput,
//                                   outputImage);  
}


template< class TInputImage, class TOutputImage >
void
OpCannyEdgeDetectionImageFilter< TInputImage, TOutputImage >
::ZeroCrossing()
{

  typename TInputImage::SizeType regionSize = 
        this->GetInput()->GetRequestedRegion().GetSize();
          
    int imageWidth = regionSize[0];
    int imageHeight = regionSize[1];

    float* inputImage = this->GetOutput()->GetBufferPointer();
    float* outputImage  = m_GaussianBuffer->GetBufferPointer();

    const int kernelWidth = 3;
    const int halfKernel = kernelWidth / 2;
    const int imageStride = this->GetInput()->GetOffsetTable()[1];
    
    int startX  = 4;
    int stopX   = (imageWidth - 2 * halfKernel) - 4;
    int startY  = 0;
    int stopY   = imageHeight - 2 * halfKernel;

      
    #pragma omp parallel for shared (inputImage, outputImage) 
    for (int y = startY; y < stopY; ++y) {
      for (int x = startX; x < stopX; x += 4) {
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
        
        if (_mm_movemask_ps(out) == 0xf) {
          _mm_storeu_ps(&outputImage[(y + 1) * imageStride + x], _mm_and_ps(out, one)); 
          continue;
        }
        
        that = _mm_load_ps(&inputImage[(y + 2) * imageStride + x]);  
                
        maskThat = _mm_and_ps(that, sign);
        maskThat = _mm_or_ps(maskThat, one);
        
        lessThan = _mm_cmpneq_ps(maskThis, maskThat);
        maskThat = _mm_or_ps(lessThan, out);

        absThat = _mm_and_ps(that, abs);
//        absThis = _mm_and_ps(thisOne, abs);
              
        lessThan = _mm_cmplt_ps(absThis, absThat);
        
        maskThat = _mm_and_ps(lessThan, maskThat);
        
        out = _mm_or_ps(maskThat, out);
        
        if (_mm_movemask_ps(out) == 0xf) {
          _mm_storeu_ps(&outputImage[(y + 1) * imageStride + x], _mm_and_ps(out, one)); 
          continue;
        }

        that = _mm_load_ps(&inputImage[(y + 1) * imageStride + x - 4]);  
        that = _mm_blend_ps(thisOne, that, 8); 
        
        ROTATE_RIGHT(that);
        
        maskThat = _mm_and_ps(that, sign);
        maskThat = _mm_or_ps(maskThat, one);
        
        lessThan = _mm_cmpneq_ps(maskThis, maskThat);
        
        maskThat = _mm_or_ps(lessThan, out);

        absThat = _mm_and_ps(that, abs);
//        absThis = _mm_and_ps(thisOne, abs);
  
              
        lessThan = _mm_cmplt_ps(absThis, absThat);
        
        maskThat = _mm_and_ps(lessThan, maskThat);
        
        out = _mm_or_ps(maskThat, out);
        
        
        if (_mm_movemask_ps(out) == 0xf) {
          _mm_storeu_ps(&outputImage[(y + 1) * imageStride + x], _mm_and_ps(out, one)); 
          continue;
        }
        
        that = _mm_load_ps(&inputImage[(y + 1) * imageStride + x + 4]);  
        that = _mm_blend_ps(thisOne, that, 1); 
        ROTATE_LEFT(that);
        maskThat = _mm_and_ps(that, sign);
        maskThat = _mm_or_ps(maskThat, one);
        lessThan = _mm_cmpneq_ps(maskThis, maskThat);
        maskThat = _mm_or_ps(lessThan, out);

        absThat = _mm_and_ps(that, abs);
//        absThis = _mm_and_ps(thisOne, abs);
              
        lessThan = _mm_cmplt_ps(absThis, absThat);
        
        equal = _mm_cmpeq_ps(absThis, absThat);
        
        lessThan = _mm_or_ps(lessThan, equal);
        
        maskThat = _mm_and_ps(lessThan, maskThat);
        
        out = _mm_or_ps(maskThat, out);
        
        _mm_storeu_ps(&outputImage[(y + 1) * imageStride + x], _mm_and_ps(out, one)); 
        
            
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

  typename TInputImage::SizeType regionSize = 
        this->GetInput()->GetRequestedRegion().GetSize();
          
    int imageWidth = regionSize[0];
    int imageHeight = regionSize[1];

    float* inputImage  = m_GaussianBuffer->GetBufferPointer();
    float* outputImage = this->GetOutput()->GetBufferPointer();
    
    PRINT_LABEL ("HysteresisThresholding");


    const int imageStride = this->GetInput()->GetOffsetTable()[1]; 

    ClearBuffer( outputImage, 
                 imageStride, 
                 imageHeight );
    
    ClearBuffer( m_UpdateBuffer->GetBufferPointer(), 
                 imageStride, 
                 imageHeight );
    
    PRINT(imageWidth);
    
    int startX  = 0;
    int stopX   = imageWidth;
    int startY  = 0;
    int stopY   = imageHeight;

    HysteresisEdgeIndex* buffer = (HysteresisEdgeIndex*)(m_UpdateBuffer->GetBufferPointer());
    int offset = imageHeight / omp_get_num_threads();
    
    HysteresisQueue* queues = new HysteresisQueue[omp_get_max_threads()];
    
    for(int i = 0; i < omp_get_max_threads(); ++i) {
      queues[i] = HysteresisQueue(buffer);
      buffer += offset;
    }
    
    #pragma omp parallel for shared (queues, inputImage, outputImage)  
    for (int y = startY; y < stopY; ++y) {
      HysteresisQueue queue = queues[omp_get_thread_num()];
        const register __m128 upperThreshold = _mm_set1_ps(m_UpperThreshold);
        
      for (int x = startX; x < stopX; x += 4) {
        
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
    }
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

      if(x < imageWidth && y > 0) {
        i = (y - 1) * imageStride + x + 1; //north east
        if(input[i] > m_LowerThreshold && 
           output[i] != 1) {
            queue.Enqueue(x + 1, y - 1);
            output[i] = 1;
        }
      }
                        
      if(x < imageWidth) {
        i = y * imageStride + x + 1; //east
        if(input[i] > m_LowerThreshold && 
           output[i] != 1) {
            queue.Enqueue(x + 1, y);
            output[i] = 1;
        }
      }
                        
      if(x < imageWidth && y < imageHeight) {
        i = (y + 1) * imageStride + x + 1; //south east
        if(input[i] > m_LowerThreshold && 
           output[i] != 1) {
            queue.Enqueue(x + 1, y + 1);
            output[i] = 1;
        }
      }
                        
      if(y < imageHeight) {
        i = (y + 1) * imageStride + x; //south
        if(input[i] > m_LowerThreshold && 
           output[i] != 1) {
            queue.Enqueue(x, y + 1);
            output[i] = 1;
        }
      }
                        
      if(x > 0 && y < imageHeight) {
        i = (y + 1) * imageStride + x - 1; //south west
        if(input[i] > m_LowerThreshold && 
           output[i] != 1) {
            queue.Enqueue(x - 1, y + 1);
            output[i] = 1;
        }
      }
}



template< class TInputImage, class TOutputImage >
int 
OpCannyEdgeDetectionImageFilter< TInputImage, TOutputImage >
::CalculateAlignedChunk ( int height )
{
    int chunk = height / omp_get_num_procs();
    return chunk % 64 == 0 ? chunk : (chunk + 64 - (chunk % 64)) / height;
}

template< class TInputImage, class TOutputImage >
float* 
OpCannyEdgeDetectionImageFilter< TInputImage, TOutputImage >
::AllocateAlignedBuffer ( int width, int height )
{
    int stride = CalculateAlignedStride (width, height, sizeof(float), ALIGMENT_BYTES);
    //aligned in 64 bytes for cache performance in Intel Core architecture
    //since 64 bytes is a multiple of 16 bytes, SSE alignment will be preserved  
#if defined(__GNUC__)  && !defined(__INTEL_COMPILER)  
    float *buffer __attribute__ ((aligned(ALIGMENT_BYTES))) = new float[stride * height];
#elif defined __INTEL_COMPILER  
    __declspec(align(ALIGMENT_BYTES)) float *buffer = new float[stride * height];
#endif
    
    return buffer;
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
    int imageStride = CalculateAlignedStride ( width, height, sizeof(float), ALIGMENT_BYTES ); 
    scGaussian7SSE (imageStride, width, height, 
                    input, output, kernel);
}

template< class TInputImage, class TOutputImage >
void
OpCannyEdgeDetectionImageFilter< TInputImage, TOutputImage >
::ComputeGradient(const float* input, float* output)
{
    PRINT_LABEL ("ComputeGradient");
    typename TInputImage::SizeType regionSize = this->GetInput()->GetRequestedRegion().GetSize();
    int width = regionSize[0];
    int height = regionSize[1];    
    
    int stride = CalculateAlignedStride ( width, height ); 
    
    const float* in = input;
    // create temporary buffers
    float* tempGx = AllocateAlignedBuffer ( width, height );
    float* tempGy = AllocateAlignedBuffer ( width, height );
    
     // kernel
    int radius = m_GaussianKernelWidth / 2 + 1;
    int gaussianRadius = m_GaussianKernelWidth / 2 * 2;
    
    int startX  = 0;
    int stopX   = width - (2 * radius); 
    int startY  = 0;
    int stopY   = height - gaussianRadius;   
    
    // ### Calculates the x gaussian
    __m128i kvGx = _mm_set_epi32 ( 0, -1, 0, 1 );   // kernel gx
    __m128i kvGy = _mm_set_epi32 ( 0, 1, 2, 1 );    // kernel gy
    
    PRINT_LABEL ("Gradient x - row order");
    #ifndef DEBUG
    #pragma omp parallel for num_threads(omp_get_num_procs()) shared (in, tempGx, tempGy, kvGx, kvGy) 
    #endif
    for (int y = startY; y < stopY; ++y)
    {
        PRINT_LINE();
        for (int x = startX; x < stopX; ++x)
        {
        
            int idxT = y * stride + x;
            // gx
            __m128i inv = _mm_setzero_si128(); // create input vector and set to zero
            inv = _mm_loadu_si128 ( ( __m128i *) &in[y * stride + x]); // load groups of four floats
            __m128i r =  _mm_cvtps_epi32 (_mm_dp_ps ( _mm_cvtepi32_ps (kvGx), 
                                                      _mm_cvtepi32_ps (inv), 
                                                      241 )); //calculate dot procuct and store in kvGx[0]
            tempGx[idxT] = _mm_extract_epi32 (r, 0); // extract the r[0]
            PRINT_INLINE ("gx:");
            PRINT_INLINE (tempGx[idxT]);
            
            // gy
            r =  _mm_cvtps_epi32 (_mm_dp_ps ( _mm_cvtepi32_ps (kvGy), 
                                              _mm_cvtepi32_ps (inv), 
                                              241 )); //calculate dot procuct and store in kvGy[0]
            tempGy[idxT] = _mm_extract_epi32 (r, 0); // extract the r[0]           
            PRINT_INLINE ("gy:");
            PRINT_INLINE (tempGy[idxT]);
        }    
    }
                                    
    float* gx = tempGx;
    float* gy = tempGy;
    
    startX  = 0;
    stopX  = width - 2 * radius;  

    startY  = 0;
    stopY   = height - 2 * radius;      

    kvGx = _mm_set_epi32( 0, 1, 2, 1 );
    kvGy = _mm_set_epi32( 0, -1, 0, 1 );

    PRINT_LABEL ("Gradiente y - col order");
    
    // ### Calculates the y gaussian
    #ifndef DEBUG
    #pragma omp parallel for num_threads(omp_get_num_procs()) shared (tempGx, tempGy, kvGx, kvGy) 
    #endif
    for (int y = startY; y < stopY; ++y) {
        PRINT_LINE();
        int m, xx, yy;
        for (int x = startX; x < stopX; ++x)   {
            int idxI = y * stride + x;
            //gx
            __m128i inv = _mm_setzero_si128(); // create input vector and set to zero
            inv = _mm_insert_epi32 (inv, gx[idxI], 0);
            inv = _mm_insert_epi32 (inv, gx[idxI + stride] , 1);
            inv = _mm_insert_epi32 (inv, gx[idxI + 2 * stride], 2);

            __m128i r =  _mm_cvtps_epi32 (_mm_dp_ps ( _mm_cvtepi32_ps (kvGx), 
                                                _mm_cvtepi32_ps (inv), 
                                                241 )); // calculate dot procuct and store in kvGx[0]
                                                
                                                
            //outGx[idxI] = _mm_extract_epi32 (r, 0);
            xx = _mm_extract_epi32 (r, 0);
            PRINT_INLINE ("gx:");
            PRINT_INLINE (xx);
            
            //gy
            inv = _mm_insert_epi32 (inv, gy[idxI], 0);
            inv = _mm_insert_epi32 (inv, gy[idxI + stride] , 1);
            inv = _mm_insert_epi32 (inv, gy[idxI + 2 * stride], 2);
            r =  _mm_cvtps_epi32 (_mm_dp_ps ( _mm_cvtepi32_ps (kvGy), 
                                              _mm_cvtepi32_ps (inv), 
                                              241 )); // calculate dot procuct and store in kvGy[0]
            //outGy[idxI] = _mm_extract_epi32 (r, 0);
            yy = _mm_extract_epi32 (r, 0);
            //OpCannyEdgeDetectionImageFilter< TInputImage, TOutputImage >::OpCannyMagnitudeAndOrientation mo;
            //mo = 
            
            PRINT_INLINE ("gy:");
            PRINT_INLINE (yy);
            
            output[idxI] = *(int*)&GetMagnitudeAndOrientation(xx,yy);
            
        }
    }       
    
    delete [] tempGx;
    delete [] tempGy;
    
}



template< class TInputImage, class TOutputImage >
void
OpCannyEdgeDetectionImageFilter< TInputImage, TOutputImage >
:: ClearBuffer( float* buffer, 
                int stride, 
                int height )
{
    OutputImagePixelType * start  = buffer;
    OutputImagePixelType * end = &buffer[stride * height];
    OutputImagePixelType * p;
    
    //openmp is slower than sse only version 
    //#pragma omp parallel for num_threads(omp_get_num_procs()) shared (start, end) private(p)
    //for (p = start; p < end; ++p) {
    //     *p = 0;
    //}
    
    const __m128 value = _mm_set_ps(0.0f, 0.0f, 0.0f, 0.0f);
    for (p = start; p < end - 32; p += 32) {
        _mm_stream_ps(p, value);
        _mm_stream_ps(p + 4, value);
        _mm_stream_ps(p + 8, value);
        _mm_stream_ps(p + 12, value);
        _mm_stream_ps(p + 16, value);
        _mm_stream_ps(p + 20, value);
        _mm_stream_ps(p + 24, value);
        _mm_stream_ps(p + 28, value);
    }
    
    p -= 32;

    // trailing ones
    while (p < end)
        *p++ = 0;  
    
}


template< class TInputImage, class TOutputImage >
void
OpCannyEdgeDetectionImageFilter< TInputImage, TOutputImage >
:: Multiply( int stride, int height, 
             float* input1, float* input2,  float* output )
{
//    cout << "stride " << stride << endl;
    int startY  = 0;
    int stopY   = height * stride; //shared (input1, input2, output) 
    //#pragma omp parallel for 
    for (int y = startY; y < stopY; y += 4) {
        __m128 inv0 = _mm_load_ps(&input1[y]);   PRINT_VECTOR(inv0);
        __m128 inv1 = _mm_load_ps(&input2[y]);   PRINT_VECTOR(inv1);
        _mm_stream_ps(&output[y], _mm_mul_ps(inv0, inv1));
    }
}


template< class TInputImage, class TOutputImage >
void
OpCannyEdgeDetectionImageFilter< TInputImage, TOutputImage >
:: MaxMin ( const float* buffer, int* max, int* min, int radius )
{

    typename TInputImage::SizeType regionSize = this->GetInput()->GetRequestedRegion().GetSize();
    int width = regionSize[0];
    int height = regionSize[1];    

    *min = std::numeric_limits<int>::max();
    *max = std::numeric_limits<int>::min();


    int stride = CalculateAlignedStride ( width, height );    
    
    //openmp is slower than sse only version 
    //#pragma omp parallel for num_threads(omp_get_num_procs()) shared (start, end) private(p)
    //for (p = start; p < end; ++p) {
    //     *p = 0;
    //}
    
    int startX  = 0;
    int stopX   = width - 2 * radius;
    int startY  = 0;
    int stopY   = height - 2 * radius;   
    int offset = (stride - width + radius * 2);
    int x = 0;
    __m128i maxv = _mm_set1_epi32(*max);
    __m128i minv =  _mm_set1_epi32(*min);
    //cout << offset << endl;
    //cout << radius << endl;
    //cout << width << endl;
    //cout << stride << endl;
    for (int y = startY; y < stopY; ++y)
    {
        for (x = startX; x < stopX - 4; x += 4, buffer += 4)
        {    
            //cout << x << " ";
            //cout << (((long)buffer % 16) == 0) << " ";
            __m128i v = _mm_load_si128 ( ( __m128i *) buffer );
            maxv =  _mm_max_epi32 (maxv, v);
            minv =  _mm_min_epi32 (minv, v);
        }
        // trailing ones
        while (x++ < stopX) {
            if ( *buffer > *max ) 
                *max = *buffer; 
            if ( *buffer < *min ) 
                *min = *buffer; 
            //cout << ((buffer % 16) == 0) << " ";
            buffer++;     
        }         
        buffer += offset;
    }

    int temp[4] __attribute__ ((aligned(16)));
    _mm_store_si128 ( (__m128i * ) temp, maxv); // store result in tempv    
    /*
    cout << endl;
    cout << temp[0] << endl;
    cout << temp[1] << endl;
    cout << temp[2] << endl;
    cout << temp[3] << endl;
    */
    
    for (int i = 0; i < 4; ++i) {
        if ( temp [i] > *max ) 
            *max = temp [i]; 
    }    
    _mm_store_si128 ( (__m128i * ) temp, minv); // store result in tempv    
    for (int i = 0; i < 4; ++i) {
        if ( temp [i] < *min ) 
            *min = temp [i]; 
    } 
       
}


template< class TInputImage, class TOutputImage >
float
OpCannyEdgeDetectionImageFilter< TInputImage, TOutputImage >
:: MaxGradient ( const float* gx, const float* gy )
{

    typename TInputImage::SizeType regionSize = this->GetInput()->GetRequestedRegion().GetSize();
    int width = regionSize[0];
    int height = regionSize[1];    


    int stride = CalculateAlignedStride ( width, height );    
    int radius = (m_GaussianKernelWidth / 2) + 1;
    
    const float*  gxp = gx;
    const float*  gyp = gy;
    
    //openmp is slower than sse only version 
    //#pragma omp parallel for num_threads(omp_get_num_procs()) shared (start, end) private(p)
    //for (p = start; p < end; ++p) {
    //     *p = 0;
    //}
    
    int startX  = 0;
    int stopX   = width - 2 * radius;
    int startY  = 0;
    int stopY   = height - 2 * radius;   
    int offset = (stride - width + radius * 2) - 4;
    
    __m128i result, abs, gxv, gyv;
    result = _mm_setzero_si128();
    abs =  _mm_set1_epi32 (0xFFFFFFFF);
    float maxGradient = 0;
    //float g = 0;
    int x = 0;
    for (int y = startY; y < stopY; ++y)
    {
        for (x = startX; x < stopX - 4; x += 4, gxp += 4, gyp += 4)
        {    
            gxv = _mm_loadu_si128 ( ( __m128i *) gxp );
            gyv = _mm_loadu_si128 ( ( __m128i *) gyp );
            result =  _mm_max_epi32 (result, 
                _mm_add_epi32 (  _mm_and_si128 (gxv, abs),  
                                 _mm_and_si128 (gyv, abs) ));
        }
        /*
        gxp -= 4;
        gyp -= 4;

        int temp[4] __attribute__ ((aligned(16)));
        _mm_storeu_si128 ( (__m128i * ) temp, result); // store result in tempv    
         
        for (int i = 0; i < 4; ++i) {
            if ( temp [i] > maxGradient ) 
                maxGradient = temp [i]; 
        }
        // trailing ones
        while (x++ < stopX) {
            g = fabs( *gxp++ ) + fabs( *gyp++ );  
            if ( g > maxGradient ) 
                maxGradient = g; 
        }        */
        gxp += offset;
        gyp += offset;
    }

    int temp[4] __attribute__ ((aligned(16)));
    _mm_storeu_si128 ( (__m128i * ) temp, result); // store result in tempv    
     
    for (int i = 0; i < 4; ++i) {
        if ( temp [i] > maxGradient ) 
            maxGradient = temp [i]; 
    }    
     /*
    for (gxp = startGx, gyp = startGy; gxp < endGx - 4; gxp += 4, gyp += 4) {
    
        gxv = _mm_loadu_ps ( gxp );
        gyv = _mm_loadu_ps ( gyp );
        result = _mm_max_ps(result, _mm_add_ps ( _mm_and_ps (gxv, abs), _mm_and_ps (gyv, abs) ));
        
        cout << *gxp << ", " << *gyp << "; ";
        
        //gxp += offset;
        //gyp += offset;
        
       
        gxp += 4; gyp += 4;
        gxv = _mm_load_ps ( gxp );
        gyv = _mm_load_ps ( gyp );
        result = _mm_max_ps(result, _mm_add_ps ( _mm_and_ps (gxv, abs), _mm_and_ps (gyv, abs) ));
        
        cout << *gxp << ", " << *gyp << "; ";
        
        gxp += 4; gyp += 4;
        gxv = _mm_load_ps ( gxp );
        gyv = _mm_load_ps ( gyp );
        result = _mm_max_ps(result, _mm_add_ps ( _mm_and_ps (gxv, abs), _mm_and_ps (gyv, abs) ));
        
        cout << *gxp << ", " << *gyp << "; ";
        
        gxp += 4; gyp += 4;
        gxv = _mm_load_ps ( gxp );
        gyv = _mm_load_ps ( gyp );
        result = _mm_max_ps(result, _mm_add_ps ( _mm_and_ps (gxv, abs), _mm_and_ps (gyv, abs) ));
        
        cout << *gxp << ", " << *gyp << "; ";
        
        gxp += 4; gyp += 4;
        gxv = _mm_load_ps ( gxp );
        gyv = _mm_load_ps ( gyp );
        result = _mm_max_ps(result, _mm_add_ps ( _mm_and_ps (gxv, abs), _mm_and_ps (gyv, abs) ));
        
        cout << *gxp << ", " << *gyp << "; ";
        
        gxp += 4; gyp += 4;
        gxv = _mm_load_ps ( gxp );
        gyv = _mm_load_ps ( gyp );
        result = _mm_max_ps(result, _mm_add_ps ( _mm_and_ps (gxv, abs), _mm_and_ps (gyv, abs) ));
        
        cout << *gxp << ", " << *gyp << "; ";
        
        gxp += 4; gyp += 4;
        gxv = _mm_load_ps ( gxp );
        gyv = _mm_load_ps ( gyp );
        result = _mm_max_ps(result, _mm_add_ps ( _mm_and_ps (gxv, abs), _mm_and_ps (gyv, abs) ));
        
        cout << *gxp << ", " << *gyp << "; ";
        
        gxp += 4; gyp += 4;
        gxv = _mm_load_ps ( gxp );
        gyv = _mm_load_ps ( gyp );
        result = _mm_max_ps(result, _mm_add_ps ( _mm_and_ps (gxv, abs), _mm_and_ps (gyv, abs) ));
     
        cout << *gxp << ", " << *gyp << "; ";
        
    }
        */
    
   
    return maxGradient;
}

template< class TInputImage, class TOutputImage >
inline void
OpCannyEdgeDetectionImageFilter< TInputImage, TOutputImage >
::VerifyThreshold( float* out, int g, int x, int y ) {

    ListNodeType * node;
    if (*out != 0) {
        // check if above or equal to high threshold, if so have an edge
        if (g > m_UpperThreshold) {
            *out = 255;
            node = m_NodeStore->Borrow();
            IndexType i = {{x, y}};
            node->m_Value = i;
            m_NodeList->PushFront(node);
        }      
        // check if below low threshold, if so not an edge
        else if (g <= m_LowerThreshold) {
              *out = 0;
        }
        else {
            *out = 1;
        }
    } 
}


template< class TInputImage, class TOutputImage >
void
OpCannyEdgeDetectionImageFilter< TInputImage, TOutputImage >
:: NonMaximumSuppressionBoundaries(float* output, float* om)
{
    PRINT_LABEL ("NonMaximumSuppressionBoundaries");
    //TODO: usar tangente para o theta
    //TODO: Empilhar apenas os inteiros, tirar a estrutura.
    typename TInputImage::SizeType regionSize = 
        this->GetInput()->GetRequestedRegion().GetSize();
    int width = regionSize[0];
    int height = regionSize[1];    
    
    //deletar
    float* gx = om;
    float* gy = om;
    
    float* gxp = gx;
    float* gyp = gy;
    float* out = output;

    //TODO: NonMaximaSupression: Do the magnitude and 
    // orientation calculation in one pass only.
    int stride = CalculateAlignedStride ( width, height );    
    int radius = m_GaussianKernelWidth / 2 + 1;
    
    //ClearEdges(output, stride, height, 0, 0, radius * 2, radius * 2); 

    float t = 0; //theta
    
    // 0 to 22.5 is set to 0 degrees.
    const float q1 = 0.392699082;   
    // 22.5 to 67.5 degrees is set to 45 degrees (0.785398163 radians).
    const float q2 = 1.17809725;    
    // 67.5 to 112.5 degrees is set to 90 degrees (1.57079633 radians).
    const float q3 = 1.96349541;    
    // 112.5 to 157.5 degrees is set to 135 degrees (2.35619449 radians).
    const float q4 = 2.74889357;    
    
    
    int g;
    
    width -= radius * 2;    
    height -= radius * 2;    

    PRINT_INLINE ("\t");
    PRINT_LABEL ("Corner Top-Left");
    PRINT_INLINE ("\t");    
    
    // x . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .        
    gxp = gx;    
    gyp = gy;    
    out = output;    
    
    t = arctan2 (*gyp, *gxp);
    g =  abs(*gxp) + abs(*gyp);  
    
    PRINT_INLINE (*gxp);
    PRINT_INLINE (",");
    PRINT_INLINE (*gyp);
    PRINT_INLINE (";");
    
    if ( t < q1 ) { // 0 degrees = -
        *out = g > abs(*(gxp + 1)) + abs(*(gyp + 1));
    } 
    else if ( t < q2 ) { // 45 degrees = \ -
        *out = g > abs(*(gxp + stride + 1)) + abs(*(gyp + stride + 1));
    }
    else if ( t < q3) { // 90 degress = |
        *out = g > abs(*(gxp + stride)) + abs(*(gyp + stride));
    }
    else { // 0 degrees = -
        *out = g > abs(*(gxp + 1)) + abs(*(gyp + 1));
    }

    VerifyThreshold( out, g, 0, 0 );
    

     
    PRINT_INLINE ("\t");
    PRINT_LABEL ("Corner Top-Right");
    PRINT_INLINE ("\t");    
    // . . . . . . . . . x
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .        
    gxp = gx + width - 1;
    gyp = gy + width - 1;
    out = output + width - 1;  
    
    t = arctan2 (*gyp, *gxp);
    g =  abs(*gxp) + abs(*gyp);  
    
    PRINT_INLINE (*gxp);
    PRINT_INLINE (",");
    PRINT_INLINE (*gyp);
    PRINT_INLINE (";");
    
    if ( t < q1 ) { // 0 degrees = -
        *out = g > abs(*(gxp - 1)) + abs(*(gyp - 1));
    } 
    else if ( t < q3) { // 90 degress = |
        *out = g > abs(*(gxp + stride)) + abs(*(gyp + stride));
    }
    else if ( t < q4 ) { // 135 degress = /
        *out = g > abs(*(gxp + stride - 1)) + abs(*(gyp + stride - 1));
    }            
    else { // 0 degrees = -
        *out = g > abs(*(gxp - 1)) + abs(*(gyp - 1));
    }

    VerifyThreshold( out, g, width - 1, 0 );
        
    PRINT_INLINE ("\t");
    PRINT_LABEL ("Corner Bottom-Right");
    PRINT_INLINE ("\t");    
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . x        
    gxp = gx + stride * (height - 1) + width - 1;
    gyp = gy + stride * (height - 1) + width - 1;
    out = output + stride * (height - 1) + width - 1;  
    
    t = arctan2 (*gyp, *gxp);
    g =  abs(*gxp) + abs(*gyp);  
    
    PRINT_INLINE (*gxp);
    PRINT_INLINE (",");
    PRINT_INLINE (*gyp);
    PRINT_INLINE (";");
    
    if ( t < q1 ) { // 0 degrees = -
        *out = g > abs(*(gxp - 1)) + abs(*(gyp - 1));
    } 
    else if ( t < q3) { // 90 degress = |
        *out = g > abs(*(gxp - stride)) + abs(*(gyp - stride));
    }
    else if ( t < q2 ) { // 45 degrees = \ -
        *out = g > abs(*(gxp - stride - 1)) + abs(*(gyp - stride - 1));
    }       
    else { // 0 degrees = -
        *out = g > abs(*(gxp - 1)) + abs(*(gyp - 1));
    }

    VerifyThreshold( out, g, width - 1, height - 1 );
        
        
    PRINT_INLINE ("\t");
    PRINT_LABEL ("Corner Bottom-Left");
    PRINT_INLINE ("\t");    
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // x . . . . . . . . .        
    gxp = gx + stride * (height - 1);
    gyp = gy + stride * (height - 1);
    out = output + stride * (height - 1);  
    
    t = arctan2 (*gyp, *gxp);
    g =  abs(*gxp) + abs(*gyp);  
    
    PRINT_INLINE (*gxp);
    PRINT_INLINE (",");
    PRINT_INLINE (*gyp);
    PRINT_INLINE (";");
    
    if ( t < q1 ) { // 0 degrees = -
        *out = g > abs(*(gxp + 1)) + abs(*(gyp + 1));
    } 
    else if ( t < q3) { // 90 degress = |
        *out = g > abs(*(gxp - stride)) + abs(*(gyp - stride));
    }
    else if ( t < q4 ) { // 135 degress = /
        *out = g > abs(*(gxp - stride + 1)) + abs(*(gyp - stride + 1));
    }            
    else { // 0 degrees = -
        *out = g > abs(*(gxp + 1)) + abs(*(gyp + 1));
    }

    VerifyThreshold( out, g, 0, height - 1 );          
    
    gxp = gx + stride;    
    gyp = gy + stride;    
    out = output + stride;    
            
    PRINT_INLINE ("\t");
    PRINT_LABEL ("Left");
    PRINT_INLINE ("\t");
    // . . . . . . . . . .
    // x . . . . . . . . .
    // x . . . . . . . . .
    // x . . . . . . . . .
    // x . . . . . . . . .
    // x . . . . . . . . .
    // x . . . . . . . . .
    // x . . . . . . . . .
    // x . . . . . . . . .
    // . . . . . . . . . .    
    for (int y = 1; y < height - 1; y++) //the corner pixels are not evaluated
    {
        t = arctan2 (*gyp, *gxp);
        g =  abs(*gxp) + abs(*gyp);  
        
        PRINT_INLINE (*gxp);
        PRINT_INLINE (",");
        PRINT_INLINE (*gyp);
        PRINT_INLINE (";");
        
        if ( t < q1 ) { // 0 degrees = -
            *out = g > abs(*(gxp + 1)) + abs(*(gyp + 1));
        } 
        else if ( t < q2 ) { // 45 degrees = \ -
            *out = g > abs(*(gxp + stride + 1)) + abs(*(gyp + stride + 1));
        }
        else if ( t < q3) { // 90 degress = |
            *out = g > abs(*(gxp - stride)) + abs(*(gyp - stride));
            if (*out)
                *out = g > abs(*(gxp + stride)) + abs(*(gyp + stride));
        }
        else if ( t < q4 ) { // 135 degress = /
            *out = g > abs(*(gxp - stride + 1)) + abs(*(gyp - stride + 1));
        }            
        else { // 0 degrees = -
            *out = g > abs(*(gxp + 1)) + abs(*(gyp + 1));
        }

        VerifyThreshold( out, g, 0, y );

        out += stride; 
        gyp += stride; 
        gxp += stride; 
    }   

    gxp = gx + width - 1 + stride;
    gyp = gy + width - 1 + stride;
    out = output + width - 1 + stride;
    
    PRINT_INLINE ("\t");
    PRINT_LABEL ("Right");
    PRINT_INLINE ("\t");
    // . . . . . . . . . .
    // . . . . . . . . . x
    // . . . . . . . . . x
    // . . . . . . . . . x
    // . . . . . . . . . x
    // . . . . . . . . . x
    // . . . . . . . . . x
    // . . . . . . . . . x
    // . . . . . . . . . x
    // . . . . . . . . . .    
    for (int y = 1; y < height - 1; y++)
    {
        t = arctan2 (*gyp, *gxp);
        g =  abs(*gxp) + abs(*gyp);  
        
        PRINT_INLINE (*gxp);
        PRINT_INLINE (",");
        PRINT_INLINE (*gyp);
        PRINT_INLINE (";");        
        
        if ( t < q1 ) { // 0 degrees = -
            *out = g > abs(*(gxp - 1)) + abs(*(gyp - 1));
        } 
        else if ( t < q2 ) { // 45 degrees - \ -
            *out = g > abs(*(gxp - stride - 1)) + abs(*(gyp - stride - 1));
        }
        else if ( t < q3) { // 90 degress - |
            *out = g > abs(*(gxp - stride)) + abs(*(gyp - stride));
            if (*out)
                *out = g > abs(*(gxp + stride)) + abs(*(gyp + stride));
        }
        else if ( t < q4 ) { // 135 degress - /
            *out = g > abs(*(gxp + stride - 1)) + abs(*(gyp + stride - 1));
        }            
        else {  // 0 degrees = -
            *out = g > abs(*(gxp - 1)) + abs(*(gyp - 1));
        }

        VerifyThreshold( out, g, width - 1, y );

        out += stride; 
        gyp += stride; 
        gxp += stride; 
    }   
                        

    PRINT_INLINE ("\t");
    PRINT_LABEL ("Top");
    PRINT_INLINE ("\t");
    gxp = gx + 1;
    gyp = gy + 1;
    out = output + 1;
    // . x x x x x x x x .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .    
    for (int x = 1; x < width - 1; x++)
    {
        t = arctan2 (*gyp, *gxp);
        g =  abs(*gxp) + abs(*gyp);  
        
        PRINT_INLINE (*gxp);
        PRINT_INLINE (",");
        PRINT_INLINE (*gyp);
        PRINT_INLINE (";");               
        
        if ( t < q1 ) { // 0 degrees = -
            *out = g > abs(*(gxp - 1)) + abs(*(gyp - 1));
            if (*out)
                *out = g > abs(*(gxp + 1)) + abs(*(gyp + 1));
        } 
        else if ( t < q2 ) { // 45 degrees - \ -
            *out = g > abs(*(gxp + stride + 1)) + abs(*(gyp + stride + 1));
        }
        else if ( t < q3) { // 90 degress - |
            *out = g > abs(*(gxp + stride)) + abs(*(gyp + stride));
        }
        else if ( t < q4 ) { // 135 degress - /
            *out = g > abs(*(gxp + stride - 1)) + abs(*(gyp + stride - 1));
        }            
        else { // 0 degrees = -
            *out = g > abs(*(gxp - 1)) + abs(*(gyp - 1));
            if (*out)
                *out = g > abs(*(gxp + 1)) + abs(*(gyp + 1));
        }
        
        VerifyThreshold( out, g, x, 0 );

        out++; 
        gyp++; 
        gxp++; 
    }   

    gxp = gx + (stride * (height - 1)) + 1;
    gyp = gy + (stride * (height - 1)) + 1;
    out = output + (stride * (height - 1)) + 1;

    PRINT_INLINE ("\t");
    PRINT_LABEL ("Bottom");        
    PRINT_INLINE ("\t");
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .    
    // . x x x x x x x x .
    for (int x = 1; x < width - 1; x++)
    {
        t = arctan2 (*gyp, *gxp);
        g =  abs(*gxp) + abs(*gyp);  
        
        PRINT_INLINE (*gxp);
        PRINT_INLINE (",");
        PRINT_INLINE (*gyp);
        PRINT_INLINE (";");             
        
        if ( t < q1 ) { // 0 degrees = -
            *out = g > abs(*(gxp - 1)) + abs(*(gyp - 1));
            if (*out)
                *out = g > abs(*(gxp + 1)) + abs(*(gyp + 1));
        } 
        else if ( t < q2 ) { // 45 degrees - \ -
            *out = g > abs(*(gxp + stride + 1)) + abs(*(gyp + stride + 1));
        }
        else if ( t < q3) { // 90 degress - |
            *out = g > abs(*(gxp + stride)) + abs(*(gyp + stride));
        }
        else if ( t < q4 ) { // 135 degress - /
            *out = g > abs(*(gxp - stride + 1)) + abs(*(gyp - stride + 1));
        }            
        else { // 0 degrees = -
            *out = g > abs(*(gxp - 1)) + abs(*(gyp - 1));
            if (*out)
                *out = g > abs(*(gxp + 1)) + abs(*(gyp + 1));
        }

        VerifyThreshold( out, g, x, height - 1 );

        out++; 
        gyp++; 
        gxp++; 
    }   
}


template< class TInputImage, class TOutputImage >
void
OpCannyEdgeDetectionImageFilter< TInputImage, TOutputImage >
:: NonMaximumSuppression(float* output, float* om)
{
    PRINT_LABEL ("NonMaximumSuppression");
    //TODO: usar tangente para o theta
    //TODO: Empilhar apenas os inteiros, tirar a estrutura.
    typename TInputImage::SizeType regionSize = 
        this->GetInput()->GetRequestedRegion().GetSize();
    int width = regionSize[0];
    int height = regionSize[1];    
    
    //deletar
    float* gx = om;
    float* gy = om;
    
    float* gxp = gx;
    float* gyp = gy;
    float* out = output;
    
    NonMaximumSuppressionBoundaries(output, om);
    
    //TODO: NonMaximaSupression: Do the magnitude and 
    // orientation calculation in one pass only.
    int stride = CalculateAlignedStride ( width, height );    
    int radius = m_GaussianKernelWidth / 2 + 1;
    
    //ClearEdges(output, stride, height, 0, 0, radius * 2, radius * 2); 

    float t = 0; //theta
    
    // 0 to 22.5 is set to 0 degrees.
    const float q1 = 0.392699082;   
    // 22.5 to 67.5 degrees is set to 45 degrees (0.785398163 radians).
    const float q2 = 1.17809725;    
    // 67.5 to 112.5 degrees is set to 90 degrees (1.57079633 radians).
    const float q3 = 1.96349541;    
    // 112.5 to 157.5 degrees is set to 135 degrees (2.35619449 radians).
    const float q4 = 2.74889357;    
    
    
    int g;
    
    width -= radius * 2;    
    height -= radius * 2;    

    int top = 0;
    int bottom = height - radius * 2;
    int right = width - radius * 2;
    int left = 0;
        
    gxp = gx + stride + 1;    
    gyp = gy + stride + 1;    
    out = output + stride + 1;        
        
    top = 1;
    bottom = height - 1;
    right = width - 1;
    left = 1;
    
    int startY  = top;
    int startX  = left;
    
    int stopY   = bottom;  
    int stopX  = right;   
    
    int offset = stride - width + 2;    
    PRINT_INLINE ("\t");
    PRINT_LABEL ("Middle");        
    PRINT_INLINE ("\t");
    
    // . . . . . . . . . .
    // . x x x x x x x x .
    // . x x x x x x x x .
    // . x x x x x x x x .
    // . x x x x x x x x .
    // . x x x x x x x x .
    // . x x x x x x x x .
    // . x x x x x x x x .
    // . x x x x x x x x .    
    // . . . . . . . . . .    
    for (int y = startY; y < stopY; y++)
    {
        for (int x = startX; x < stopX; x++, out++, gxp++, gyp++)
        {
            t = arctan2 (*gyp, *gxp);
            g =  abs(*gxp) + abs(*gyp);  
            //cout << *gxp / (fabs(*gyp) + 1e-10) << " ";
            
            PRINT_INLINE (*gxp);
            PRINT_INLINE (",");
            PRINT_INLINE (*gyp);
            PRINT_INLINE (";");               
            //if (t < 0 ) {
            //    t += M_PI;
            //}
           
            // ## Non-maximum suppression ##
            // The same binary map shown on the left after non-maxima 
            // suppression. The edges are still coloured to indicate direction.
        
            // Given estimates of the image gradients, a search is then carried 
            // out to determine if the gradient magnitude assumes a local 
            // maximum in the gradient direction. So, for example,
        
            // if the rounded angle is zero degrees the point will be 
            // considered to be on the edge if its intensity is greater than 
            // the intensities in the north and south directions,
            
            // if the rounded angle is 45 degrees the point will be considered 
            // to be on the edge if its intensity is greater than the 
            // intensities in the north west and south east directions.
            
            // if the rounded angle is 90 degrees the point will be considered 
            // to be on the edge if its intensity is greater than 
            // the intensities in the west and east directions,
            
            // if the rounded angle is 135 degrees the point will be considered 
            // to be on the edge if its intensity is greater than the 
            // intensities in the north east and south west directions,
                       

            if ( t < q1 ) {
                *out = g > abs(*(gxp - 1)) + abs(*(gyp - 1));
                if (*out)
                    *out = g > abs(*(gxp + 1)) + abs(*(gyp + 1));
            } 
            else if ( t < q2 ) { // 45 degrees
                *out = g > abs(*(gxp - stride - 1)) + abs(*(gyp - stride - 1));
                if (*out)
                    *out = g > abs(*(gxp + stride + 1)) + abs(*(gyp + stride + 1));
            }
            else if ( t < q3) { // 90 degress
                *out = g > abs(*(gxp - stride)) + abs(*(gyp - stride));
                if (*out)
                    *out = g > abs(*(gxp + stride)) + abs(*(gyp + stride));
            }
            else if ( t < q4 ) { // 135 degress
                *out = g > abs(*(gxp - stride + 1)) + abs(*(gyp - stride + 1));
                if (*out)
                    *out = g > abs(*(gxp + stride - 1)) + abs(*(gyp + stride - 1));
            }            
            else {
                *out = g > abs(*(gxp - 1)) + abs(*(gyp - 1));
                if (*out)
                    *out = g > abs(*(gxp + 1)) + abs(*(gyp + 1));
            }
            
            VerifyThreshold( out, g, x, y );
            
        }    
        out += offset; 
        gyp += offset; 
        gxp += offset; 
    }   
}


template< class TInputImage, class TOutputImage >
void
OpCannyEdgeDetectionImageFilter< TInputImage, TOutputImage >
::HysteresisThresholding( float* input )
{

    PRINT_LABEL ("HysteresisThresholding");        
    typename TInputImage::SizeType regionSize = this->GetInput()->GetRequestedRegion().GetSize();
    int width = regionSize[0];
    int height = regionSize[1];    
   
    int stride = CalculateAlignedStride ( width, height );    
    
    OutputImagePixelType *output = this->GetOutput(0)->GetBufferPointer(); //outputBuffer
    OutputImagePixelType *out = output; //outputBuffer
    
    int offset = m_GaussianKernelWidth / 2 + 1;

    //ClearBuffer (out, stride, height);
    
    IndexType cIndex;
    ListNodeType *node;
    ListNodeType *child_node;
    int indexIn = 0;
    int indexOut = 0;
  
    //int h = 255;

    float*  north;
    float* south;
    float* west;
    float* east;
    float* north_east;
    float* south_east;
    float* north_west;
    float* south_west;
    
    float* in;
    
                
    while(!m_NodeList->Empty()) {
    
        // Pop the front node from the list and read its index value.
        node = m_NodeList->Front(); // get a pointer to the first node
        cIndex = node->m_Value;    // read the value of the first node
        m_NodeList->PopFront();    // unlink the front node
        m_NodeStore->Return(node); // return the memory for reuse
        indexIn = cIndex[1] * stride + cIndex[0];
        indexOut = (cIndex[1] + offset) * stride + (cIndex[0] + offset);

        PRINT (cIndex);
        
        out = output + indexOut; //outputBuffer
        in = input + indexIn;

        *out = *in;
        
        north_west = in - stride - 1;
        north = in - stride;
        north_east = in - stride + 1;
        west = in - 1;
        east = in + 1;
        south_west = in + stride - 1;
        south = in + stride;
        south_east = in + stride + 1;
        
        if ( *north_west == 1 ) {
            *(out - stride - 1) = 255;
            //*north_west = h;
            child_node = m_NodeStore->Borrow();
            IndexType i = {{cIndex[0] - 1, cIndex[1] - 1}};
            child_node->m_Value = i;
            m_NodeList->PushFront(child_node);        
        }                           
        if ( *north == 1  ) {
            *(out - stride) = 255;
            //*north = h;
            child_node = m_NodeStore->Borrow();
            IndexType i = {{cIndex[0], cIndex[1] - 1}};
            child_node->m_Value = i;
            m_NodeList->PushFront(child_node);        
        }                           
        if ( *west == 1 ) {
            *(out - 1) = 255;
            //*west = h;
            child_node = m_NodeStore->Borrow();
            IndexType i = {{cIndex[0] - 1, cIndex[1]}};
            child_node->m_Value = i;
            m_NodeList->PushFront(child_node);        
        }                           
        if ( *east == 1 ) {
            *(out + 1) = 255;
            //*east = h;
            child_node = m_NodeStore->Borrow();
            IndexType i = {{cIndex[0] + 1, cIndex[1]}};
            child_node->m_Value = i;
            m_NodeList->PushFront(child_node);        
        }                           
        if ( *south_west == 1 ) {
            *(out - stride - 1) = 255;
            //*south_west = h;
            child_node = m_NodeStore->Borrow();
            IndexType i = {{cIndex[0] - 1, cIndex[1] + 1}};
            child_node->m_Value = i;
            m_NodeList->PushFront(child_node);        
        }                           
        if ( *south == 1 ) {
            *(out + stride) = 255;
            //*south = h;
            child_node = m_NodeStore->Borrow();
            IndexType i = {{cIndex[0], cIndex[1] + 1}};
            child_node->m_Value = i;
            m_NodeList->PushFront(child_node);        
        }                           
        if ( *south_east == 1 ) {
            *(out + stride + 1) = 255;
            //*south_east = h;
            child_node = m_NodeStore->Borrow();
            IndexType i = {{cIndex[0] + 1, cIndex[1] + 1}};
            child_node->m_Value = i;
            m_NodeList->PushFront(child_node);        
        }                           
    }
        
}


template< class TInputImage, class TOutputImage >
void
OpCannyEdgeDetectionImageFilter< TInputImage, TOutputImage >
:: ClearEdges( float*  buffer, 
               int stride, int height,  int top, int left, int bottom, int right )
{

    //cout << "stride " << stride << endl;
    //cout << "height " << height << endl;
    //cout << "top " << top << endl;
    //cout << "left " << left << endl;
    //cout << "bottom " << bottom << endl;
    //cout << "right " << right << endl;


    // x x x x x x x x x x
    // x x x x x x x x x x
    // x x x x x x x x x x
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    OutputImagePixelType * p = buffer;
    for (int y = 0; y < top; y++) {
        for (int x = 0; x < stride; x++, p++) {
            *p = 0;
        }
    }

    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // . . . . . . . . . .
    // x x x x x x x x x x
    // x x x x x x x x x x
    // x x x x x x x x x x
    p = buffer;
    p += (height - bottom) * stride;
    for (int y = 0; y < bottom; y++) {
        for (int x = 0; x < stride; x++, p++) {
            *p = 0;
        }
    }

    // x x x . . . . . . .
    // x x x . . . . . . .
    // x x x . . . . . . .
    // x x x . . . . . . .
    // x x x . . . . . . .
    // x x x . . . . . . .
    // x x x . . . . . . .
    // x x x . . . . . . .
    // x x x . . . . . . .
    // x x x . . . . . . .
    p = buffer;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < left; x++, p++) {
            *p = 0;
        }
        p += stride - left;
    }

    // . . . . . . . x x x 
    // . . . . . . . x x x 
    // . . . . . . . x x x 
    // . . . . . . . x x x 
    // . . . . . . . x x x 
    // . . . . . . . x x x 
    // . . . . . . . x x x 
    // . . . . . . . x x x 
    // . . . . . . . x x x 
    // . . . . . . . x x x 
    p = buffer;
    p += stride - right;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < right; x++, p++) {
            *p = 0;
        }
        p += stride - right;
    }
    
}

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
