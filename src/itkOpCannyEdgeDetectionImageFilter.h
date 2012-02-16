/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkOpCannyEdgeDetectionImageFilter.h,v $
  Language:  C++
  Date:      $Date: 2009-04-25 12:27:15 $
  Version:   $Revision: 1.28 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkOpCannyEdgeDetectionImageFilter_h
#define __itkOpCannyEdgeDetectionImageFilter_h

#include "itkImageToImageFilter.h"
#include "itkImage.h"
#include "itkFixedArray.h"
#include "itkConstNeighborhoodIterator.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkMultiplyImageFilter.h"
#include "itkZeroFluxNeumannBoundaryCondition.h"
#include "itkMultiThreader.h"
#include "itkDerivativeOperator.h"
#include "itkSparseFieldLayer.h"
#include "itkObjectStore.h"
#include "itkCannyEdgeDetectionImageFilter.h"
#include "itkOpGaussianOperator.h"

#include "opConvolutionFilter.h"

#include <omp.h>
#ifdef __SSE4_1__
#include <smmintrin.h>  
#endif 

#include <emmintrin.h>
#include <xmmintrin.h>

#include <iostream>
#include <iomanip>

//#define DEBUG

using namespace std;

using std::cout;
using std::cerr;  
using std::endl;
using std::setw;
using std::string;
using std::ifstream;


namespace itk
{
    

struct HysteresisEdgeIndex
{
public:
  unsigned short X;
  unsigned short Y;
  HysteresisEdgeIndex(int x, int y) {
   this->X = x;
   this->Y = y;
  }
};
 
class HysteresisQueue
{
public:
  int Begin;
  int End;
  int MaxSize;
  int Count;
//  unsigned long Count;
  HysteresisEdgeIndex* Buffer;
  
  HysteresisQueue() {  }
  
  HysteresisQueue(HysteresisEdgeIndex* buffer, int maxSize) {
   this->MaxSize = maxSize;
   this->Begin = -1;
   this->End = -1;
   this->Buffer = buffer;
   this->Count = 0;
  }
  
  HysteresisEdgeIndex* Enqueue(int x, int y) {
//    #pragma omp critical
//    cout << "Enqueue\tx: " << x << "\ty: " << y << "\tnt: " << omp_get_thread_num() << 
//            "\tIsFull: " << this->IsFull() << "\tIsEmpty: " << this->IsEmpty() << "\tBegin: " << Begin << "\tEnd: " << End << "\tCount: " << Count << endl;
    if (this->IsFull()) return NULL;
    if (this->IsEmpty()) this->Begin++;
    ++this->End;
    if (this->End == this->MaxSize) {
      this->End = 0;
    }
    ++this->Count;
    Buffer[this->End].X = x;
    Buffer[this->End].Y = y;
    return &Buffer[this->End];
  }
  
  HysteresisEdgeIndex* Dequeue() {
//    #pragma omp critical
//    cout << "Dequeue\t\t\tnt: " << omp_get_thread_num() << 
//            "\tIsFull: " << this->IsFull() << "\tIsEmpty: " << this->IsEmpty() << "\tBegin: " << Begin << "\tEnd: " << End << "\tCount: " << Count << endl;
    if(this->IsEmpty()) return NULL;
    int begin = this->Begin;
    ++this->Begin;
    --this->Count;
    if(this->IsEmpty()) {
     this->Begin = -1;
     this->End = -1;
    }
    else {
      if (this->Begin == this->MaxSize) {
        this->Begin = 0;
      }
    }
    return &Buffer[begin];
  }
  
//  __m128i* DequeueVec() {
//    if(this->IsEmpty() || this->Count <= 4) return NULL;
//    this->Begin += 4;
//    this->Count -= 4;
//    return (__m128i*)&Buffer[this->Begin - 4];
//  }
  
  
  HysteresisEdgeIndex* Pick() {
   return &Buffer[this->Begin];
  }
  
  inline int GetCount() {
    return this->Count;
  }
  
  inline bool IsEmpty() {
   return this->Count == 0;
  }
  
  inline bool IsFull() {
   return this->Count == this->MaxSize;
  }
  
};
   
/** \class OpCannyEdgeDetectionImageFilter
 *
 * This filter is an implementation of a Canny edge detector for scalar-valued
 * images.  Based on John Canny's paper "A Computational Approach to Edge 
 * Detection"(IEEE Transactions on Pattern Analysis and Machine Intelligence, 
 * Vol. PAMI-8, No.6, November 1986),  there are four major steps used in the 
 * edge-detection scheme:
 * (1) Smooth the input image with Gaussian filter.
 * (2) Calculate the second directional derivatives of the smoothed image. 
 * (3) Non-Maximum Suppression: the zero-crossings of 2nd derivative are found,
 *     and the sign of third derivative is used to find the correct extrema. 
 * (4) The hysteresis thresholding is applied to the gradient magnitude
 *      (multiplied with zero-crossings) of the smoothed image to find and 
 *      link edges.
 *
 * \par Inputs and Outputs
 * The input to this filter should be a scalar, real-valued Itk image of
 * arbitrary dimension.  The output should also be a scalar, real-value Itk
 * image of the same dimensionality.
 *
 * \par Parameters
 * There are four parameters for this filter that control the sub-filters used
 * by the algorithm.
 *
 * \par 
 * Variance and Maximum error are used in the Gaussian smoothing of the input
 * image.  See  itkDiscreteGaussianImageFilter for information on these
 * parameters.
 *
 * \par
 * Threshold is the lowest allowed value in the output image.  Its data type is 
 * the same as the data type of the output image. Any values below the
 * Threshold level will be replaced with the OutsideValue parameter value, whose
 * default is zero.
 * 
 * \todo Edge-linking will be added when an itk connected component labeling
 * algorithm is available.
 *
 * \sa DiscreteGaussianImageFilter
 * \sa ZeroCrossingImageFilter
 * \sa ThresholdImageFilter */
template<class TInputImage, class TOutputImage>
class ITK_EXPORT OpCannyEdgeDetectionImageFilter
  : public ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard "Self" & Superclass typedef.  */
  typedef OpCannyEdgeDetectionImageFilter                 Self;
  typedef ImageToImageFilter<TInputImage, TOutputImage> Superclass;
   
  /** Image typedef support   */
  typedef TInputImage  InputImageType;
  typedef TOutputImage OutputImageType;
      
  /** SmartPointer typedef support  */
  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** Define pixel types. */
  typedef typename TInputImage::PixelType   InputImagePixelType;
  typedef typename TOutputImage::PixelType  OutputImagePixelType;
  typedef typename TInputImage::IndexType   IndexType;

  /** The default boundary condition is used unless overridden 
   *in the Evaluate() method. */
  typedef ZeroFluxNeumannBoundaryCondition<OutputImageType>
  DefaultBoundaryConditionType;

  /** The type of data structure that is passed to this function object
   * to evaluate at a pixel that does not lie on a data set boundary.
   */
  typedef ConstNeighborhoodIterator<OutputImageType,
                                    DefaultBoundaryConditionType> NeighborhoodType;

  typedef itk::ListNode<IndexType>            ListNodeType;
  typedef ObjectStore<ListNodeType>      ListNodeStorageType;
  typedef SparseFieldLayer<ListNodeType> ListType;
  typedef typename ListType::Pointer     ListPointerType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);  
    
  /** Typedef to describe the output image region type. */
  typedef typename TOutputImage::RegionType OutputImageRegionType;
  typedef typename TInputImage::RegionType  InputImageRegionType;

  /** Run-time type information (and related methods). */
  itkTypeMacro(OpCannyEdgeDetectionImageFilter, ImageToImageFilter);
  
  /** ImageDimension constant    */
  itkStaticConstMacro(ImageDimension, unsigned int,
                      TInputImage::ImageDimension);
  itkStaticConstMacro(OutputImageDimension, unsigned int,
                      TOutputImage::ImageDimension);
                              
  /** Typedef of double containers */
  typedef FixedArray<double, itkGetStaticConstMacro(ImageDimension)> ArrayType;

  /** Standard get/set macros for filter parameters. */
  itkSetMacro(Variance, ArrayType);
  itkGetConstMacro(Variance, const ArrayType);
  itkSetMacro(MaximumError, ArrayType);
  itkGetConstMacro(MaximumError, const ArrayType);

  /** Set/Get the Variance parameter used by the Gaussian smoothing
      filter in this algorithm */
  void SetVariance(const typename ArrayType::ValueType v)
    {
    for (unsigned int i=0; i < TInputImage::ImageDimension; i++)
      {
      if (m_Variance[i] != v)
        {
        m_Variance.Fill(v);
        this->Modified();
        break;
        }
      }
    }
  
  /** Set/Get the MaximumError paramter used by the Gaussian smoothing filter
      in this algorithm */
  void SetMaximumError(const typename ArrayType::ValueType v)
    {
    for (unsigned int i=0; i < TInputImage::ImageDimension; i++)
      {
      if (m_MaximumError[i] != v)
        {
        m_MaximumError.Fill(v);
        this->Modified();
        break;
        }
      }
    }
  
  /* Set the Threshold value for detected edges. */
  void SetThreshold(const OutputImagePixelType th)
    {
    this->m_Threshold = th;
    this->m_UpperThreshold = m_Threshold;
    this->m_LowerThreshold = m_Threshold/2.0;
    itkLegacyReplaceBodyMacro(SetThreshold, 2.2, SetUpperThreshold);
    }
  
  OutputImagePixelType GetThreshold(OutputImagePixelType th) 
    {
    itkLegacyReplaceBodyMacro(GetThreshold, 2.2, GetUpperThreshold);
    return this->m_Threshold; 
    }

  /* Set the Sigma value for detected edges. */
  void SetSigma(const float value)
    {
    this->m_Sigma = value;
    }
  
  float GetSigma() 
    {
    return this->m_Sigma; 
    }

  ///* Set the Threshold value for detected edges. */
  itkSetMacro(UpperThreshold, OutputImagePixelType );
  itkGetConstMacro(UpperThreshold, OutputImagePixelType);

  itkSetMacro(LowerThreshold, OutputImagePixelType );
  itkGetConstMacro(LowerThreshold, OutputImagePixelType);

  /* Set the Thresholdvalue for detected edges. */
  itkSetMacro(OutsideValue, OutputImagePixelType);
  itkGetConstMacro(OutsideValue, OutputImagePixelType);
  
  OutputImageType * GetNonMaximumSuppressionImage()
    {
    return this->m_MultiplyImageFilter->GetOutput();
    }

  /** OpCannyEdgeDetectionImageFilter needs a larger input requested
   * region than the output requested region ( derivative operators, etc).  
   * As such, OpCannyEdgeDetectionImageFilter needs to provide an implementation
   * for GenerateInputRequestedRegion() in order to inform the 
   * pipeline execution model.
   *
   * \sa ImageToImageFilter::GenerateInputRequestedRegion()  */  
  virtual void GenerateInputRequestedRegion() throw(InvalidRequestedRegionError);
      
 #ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(InputHasNumericTraitsCheck,
    (Concept::HasNumericTraits<InputImagePixelType>));
  itkConceptMacro(OutputHasNumericTraitsCheck,
    (Concept::HasNumericTraits<OutputImagePixelType>));
  itkConceptMacro(SameDimensionCheck,
    (Concept::SameDimension<ImageDimension, OutputImageDimension>));
  itkConceptMacro(InputIsFloatCheck,
    (Concept::IsFloatingPoint<InputImagePixelType>));
  itkConceptMacro(OutputIsFloatCheck,
    (Concept::IsFloatingPoint<OutputImagePixelType>));
  /** End concept checking */
#endif

protected:
  OpCannyEdgeDetectionImageFilter();
  OpCannyEdgeDetectionImageFilter(const Self&) {}
  void PrintSelf(std::ostream& os, Indent indent) const;

  void GenerateData();


private: 
  virtual ~OpCannyEdgeDetectionImageFilter(){};

  void ClearEdges( float* buffer, 
                   int stride, int height, int top, 
                   int left, int bottom, int right );

  void ClearBuffer( float* buffer, 
                    int stride, int height );
  
  void GaussianBlur ( const InputImagePixelType *input, float* output);
  
  /** Calculate the second derivative of the smoothed image, it writes the 
   *  result to m_UpdateBuffer using the ThreadedCompute2ndDerivative() method
   *  and multithreading mechanism.   */
  void Compute2ndDerivative();
  
//  void Compute2ndDerivative(const int imageStride, const int imageWidth, 
//                            const int imageHeight, 
//                            const float* inputImage, float* lvv, float* lvvv, const int outStride);
                            
  void Compute2ndDerivative(const int imageStride, const int imageWidth, 
                       const int imageHeight, 
                       const float* inputImage, float* outputImage, const int outStride);                            

  void Compute2ndDerivativePos();
  
  void Compute2ndDerivativePos(const int imageStride, const int imageWidth, 
                       const int imageHeight, 
                       const float* gaussianInput, const float* dxInput,
                       float* outputImage, const int outStride);

  void ZeroCrossing();
  
  void ZeroCrossing(const int imageStride, const int imageWidth, 
                    const int imageHeight, 
                    const float* inputImage, float* outputImage, 
                    const int outStride, const int startX, const int stopX);

  void HysteresisThresholding();
  
  /** Edge linking funciton */
  void FollowEdge(int imageStride, int imageWidth, int imageHeight, HysteresisQueue& queue,
                  float* input, float* output);  
  
  void VerifyEdge(int x, int y, int imageStride, int imageWidth, int imageHeight, HysteresisQueue& queue,
                  float* input, float* output);  
  
  void Multiply( int stride, int height, 
             const float* __restrict input1, 
             const float* __restrict input2,  
             float* __restrict output );
             
  
  /** This allocate storage for m_UpdateBuffer, m_UpdateBuffer1 */
  void AllocateUpdateBuffer();

    int GetGaussianKernelWidth(){
        OpGaussianOperator<float, 1> oper;
        oper.SetVariance(m_Variance[0]);
        return oper.GetCoefficients().size();        
    }
    
    float* gaussianKernel1D ( )
    {
        PRINT_LABEL ("CreateDiscreteGaussianKernel1D");
        
        
        
        OpGaussianOperator<float, 1> oper;
        oper.SetVariance(m_Variance[0]);
        
        std::vector<double> coeff = oper.GetCoefficients();
        std::vector<double>::iterator it;
              
        int kernelWidth = oper.GetCoefficients().size();
        float* kernel __attribute__ ((aligned(ALIGMENT_BYTES))) = new float[kernelWidth];
        
         int i = 0;
        for (it = coeff.begin(); it < coeff.end(); ++it)
          {
           kernel[i++] = *it;
          }
        return kernel;
    }
        
  /** The variance of the Gaussian Filter used in this filter */
  ArrayType m_Variance;

  /** The maximum error of the gaussian blurring kernel in each dimensional
   * direction.  */
  ArrayType m_MaximumError;

  /** Upper threshold value for identifying edges. */
  OutputImagePixelType m_UpperThreshold;  //should be float here?

    /** Lower threshold value for identifying edges. */
  OutputImagePixelType m_LowerThreshold; //should be float here?

  /** Threshold value for identifying edges. */
  OutputImagePixelType m_Threshold;

  /** Sigma value for identifying edges. */
//  float m_Sigma;

  /** Gaussian kernel width */
  int m_GaussianKernelWidth;  
    

  /** "Background" value for use in thresholding. */
  OutputImagePixelType m_OutsideValue;

  /** Update buffers used during calculation of multiple steps */
  typename OutputImageType::Pointer  m_GaussianBuffer;
  typename OutputImageType::Pointer  m_UpdateBuffer;
  
  typename OutputImageType::Pointer  m_BoundaryBuffer1;
  typename OutputImageType::Pointer  m_BoundaryBuffer2;

  unsigned long m_Stride[ImageDimension];
  unsigned long m_Center;

  typename ListNodeStorageType::Pointer m_NodeStore;
  ListPointerType                       m_NodeList;
  
	
};

} //end of namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkOpCannyEdgeDetectionImageFilter.txx"
#endif
  
#endif