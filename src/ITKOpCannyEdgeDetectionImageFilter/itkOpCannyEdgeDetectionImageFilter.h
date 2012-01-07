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

#include "itkStopWatch.h"



#include <omp.h>
#include <smmintrin.h>
#include <emmintrin.h>
#include <xmmintrin.h>

#include <iostream>
#include <iomanip>

//#define DEBUG
#ifndef DEBUG
 #define PRINT_IMAGE_TO_FILE(file, image, width, height, stride)
    #define PRINT_IMAGE(width, height, stride, kernelWidth, out)
    #define PRINT_LABEL(x)
    #define PRINT(x)
    #define PRINT_INLINE(x)
    #define PRINT_LINE()
    #define PRINT_VECTOR(x)
    #define PRINT_POSITION(x, y)
   #define PRINT_VECTOR_RELEASE(x) \
   cout << #x << " { " << __builtin_ia32_vec_ext_v4sf(x, 0) << " " \
                       << __builtin_ia32_vec_ext_v4sf(x, 1) << " " \
                       << __builtin_ia32_vec_ext_v4sf(x, 2) << " " \
                       << __builtin_ia32_vec_ext_v4sf(x, 3) << " } " << endl;
#else

 #define PRINT_IMAGE_TO_FILE(file, width, height, stride, out) \
     printImageToFile(file, width, height, stride, out);
    #define PRINT_IMAGE(width, height, stride, kernelWidth, out) \
     printImage(width, height, stride, kernelWidth, out)
 #define PRINT_VECTOR(x) \
 cout << #x << " { " << __builtin_ia32_vec_ext_v4sf(x, 0) << " " \
                     << __builtin_ia32_vec_ext_v4sf(x, 1) << " " \
                     << __builtin_ia32_vec_ext_v4sf(x, 2) << " " \
                     << __builtin_ia32_vec_ext_v4sf(x, 3) << " } " << endl;
 #define PRINT_POSITION(x, y) \
     cout << " { " << __builtin_ia32_vec_ext_v4sf(x, y) << " } ";
    #define PRINT_LABEL(x) \
     cout << endl << " ### " << x << " ### " << endl;
    #define PRINT(x) \
     cout << #x << ":\t" << x << endl;
    #define PRINT_INLINE(x) \
     cout << x << " ";
    #define PRINT_LINE() \
     cout << endl;   
 
#endif                       
                     
#define ROTATE_LEFT(vector) \
 vector = _mm_shuffle_ps(vector, vector, _MM_SHUFFLE(0,3,2,1)); PRINT_VECTOR(vector); 

#define ROTATE_RIGHT(vector) \
 vector = _mm_shuffle_ps(vector, vector, _MM_SHUFFLE(2, 1, 0, 3)); PRINT_VECTOR(vector); 
                     
#define ROTATE_RIGHT_BLEND(vector1, vector2) \
 vector1 = _mm_shuffle_ps(vector1, vector1, _MM_SHUFFLE(2, 1, 0, 3)); PRINT_VECTOR(vector1); \
 vector2 = _mm_blend_ps(vector2, vector1, 1); PRINT_VECTOR(vector2); 
                     
#define BLEND_ROTATE_LEFT(vector0, vector1) \
 vector0 = _mm_blend_ps(vector0, vector1, 1); PRINT_VECTOR(vector0); \
 ROTATE_LEFT(vector0);

#define BLEND_ROTATE1_LEFT(vector0, vector1) \
    BLEND_ROTATE_LEFT(vector0, vector1) \
    ROTATE_LEFT(vector1)

#define BLEND_ROTATE2_LEFT(vector0, vector1, vector2) \
    BLEND_ROTATE_LEFT(vector0, vector1) \
    BLEND_ROTATE_LEFT(vector1, vector2) \
    ROTATE_LEFT(vector2)

#define BLEND_ROTATE3_LEFT(vector0, vector1, vector2, vector3) \
    BLEND_ROTATE_LEFT(vector0, vector1) \
    BLEND_ROTATE_LEFT(vector1, vector2) \
    BLEND_ROTATE_LEFT(vector2, vector3) \
    ROTATE_LEFT(vector3)

#define BLEND_ROTATE4_LEFT(vector0, vector1, vector2, vector3, vector4) \
    BLEND_ROTATE_LEFT(vector0, vector1) \
    BLEND_ROTATE_LEFT(vector1, vector2) \
    BLEND_ROTATE_LEFT(vector2, vector3) \
    BLEND_ROTATE_LEFT(vector3, vector4) \
    ROTATE_LEFT(vector4)

#define BLEND_ROTATE5_LEFT(vector0, vector1, vector2, vector3, vector4, vector5) \
    BLEND_ROTATE_LEFT(vector0, vector1) \
    BLEND_ROTATE_LEFT(vector1, vector2) \
    BLEND_ROTATE_LEFT(vector2, vector3) \
    BLEND_ROTATE_LEFT(vector3, vector4) \
    BLEND_ROTATE_LEFT(vector4, vector5) \
    ROTATE_LEFT(vector5)

#define BLEND_ROTATE6_LEFT(vector0, vector1, vector2, vector3, vector4, vector5, vector6) \
    BLEND_ROTATE_LEFT(vector0, vector1) \
    BLEND_ROTATE_LEFT(vector1, vector2) \
    BLEND_ROTATE_LEFT(vector2, vector3) \
    BLEND_ROTATE_LEFT(vector3, vector4) \
    BLEND_ROTATE_LEFT(vector4, vector5) \
    BLEND_ROTATE_LEFT(vector5, vector6) \
    ROTATE_LEFT(vector6)

#define BLEND_ROTATE7_LEFT(vector0, vector1, vector2, vector3, vector4, vector5, vector6, vector7) \
    BLEND_ROTATE_LEFT(vector0, vector1) \
    BLEND_ROTATE_LEFT(vector1, vector2) \
    BLEND_ROTATE_LEFT(vector2, vector3) \
    BLEND_ROTATE_LEFT(vector3, vector4) \
    BLEND_ROTATE_LEFT(vector4, vector5) \
    BLEND_ROTATE_LEFT(vector5, vector6) \
    BLEND_ROTATE_LEFT(vector6, vector7) \
    ROTATE_LEFT(vector7)

#define ALIGMENT_BYTES 64

using namespace std;

using std::cout;
using std::cerr;  
using std::endl;
using std::setw;
using std::string;
using std::ifstream;


namespace itk
{
    

class HysteresisEdgeIndex
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
    if (this->IsFull()) return NULL;
    if (this->IsEmpty()) this->Begin++;
    ++this->End;
    if (this->End == this->MaxSize) {
      this->End = 0;
    }
    ++this->Count;
    return &(Buffer[this->End] = HysteresisEdgeIndex(x,y));
  }
  
//  void Enqueue(HysteresisEdgeIndex index) {
//   if(IsEmpty()) this->Begin++;
//   this->End++;
//   Buffer[this->End] = HysteresisEdgeIndex(x,y);
//  }
  
  HysteresisEdgeIndex* Dequeue() {
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
// 
//   
//class HysteresisEdgeIndex
//{
//public:
//  unsigned short X;
//  unsigned short Y;
//  HysteresisEdgeIndex(int x, int y) {
//   this->X = x;
//   this->Y = y;
//  }
//};
// 
//class HysteresisQueue
//{
//public:
//  int Begin;
//  int End;
////  unsigned long Count;
//  HysteresisEdgeIndex* Buffer;
//  
//  HysteresisQueue() {  }
//  
//  HysteresisQueue(HysteresisEdgeIndex* buffer) {
//   this->Begin = -1;
//   this->End = -1;
//   this->Buffer = buffer;
//  }
//  
//  HysteresisEdgeIndex* Enqueue(int x, int y) {
//   if(IsEmpty()) this->Begin++;
//   //this->End++;
////   Count++;
//   return &(Buffer[++this->End] = HysteresisEdgeIndex(x,y));
//  }
//  
////  void Enqueue(HysteresisEdgeIndex index) {
////   if(IsEmpty()) this->Begin++;
////   this->End++;
////   Buffer[this->End] = HysteresisEdgeIndex(x,y);
////  }
//  
//  HysteresisEdgeIndex* Dequeue() {
//    if(IsEmpty()) return NULL;
//    int begin = this->Begin;
//    if(this->Begin == this->End) {
//      this->Begin = -1;
//      this->End = -1;
//    } 
//    else {
//      this->Begin++;
//    }
////    Count--;
//    return &Buffer[begin];
//  }
//  
//  __m128i* DequeueVec() {
//    if(IsEmpty() || this->Count() <= 4) return NULL;
//    this->Begin += 4;
////    Count--;
//    return (__m128i*)&Buffer[this->Begin - 4];
//  }
//  
//  
//  HysteresisEdgeIndex* Pick() {
//   return &Buffer[this->Begin];
//  }
//  
//  int Count() {
//    return this->End - this->Begin + 1;
//  }
//  
//  bool IsEmpty() {
//   return this->Begin == -1;
//  }
//  
//};
//    
   
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
    
  StopWatch& GetStopWatch()
    {
    return this->m_Timer;    
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

  /** Thread-Data Structure   */
  struct CannyThreadStruct
    {
    OpCannyEdgeDetectionImageFilter *Filter;
    };


    struct OpCannyIndex
    {
      int x;
      int y;
    };
    
    //__attribute__ ((aligned(32)))
    struct OpCannyMagnitudeAndOrientation
    {
      short Magnitude;
      short Orientation;
    };
    
inline
OpCannyMagnitudeAndOrientation
GetMagnitudeAndOrientation (int gx, int gy)
{
    short g;  // magnitude
    short m;  // slope
    
    //const float q1 = 0.392699082;   // 0 to 22.5 is set to 0 degrees.
    //const float q2 = 1.17809725;    // 22.5 to 67.5 degrees is set to 45 degrees (0.785398163 radians).
    //const float q3 = 1.96349541;    // 67.5 to 112.5 degrees is set to 90 degrees (1.57079633 radians).
    //const float q4 = 2.74889357;    // 112.5 to 157.5 degrees is set to 135 degrees (2.35619449 radians).


    const float q1 = 0.414213562;   // 0 to 22.5 is set to 0 degrees.
    const float q2 = 2.41421356;    // 22.5 to 67.5 degrees is set to 45 degrees (0.785398163 radians).
    const float q3 = -2.41421356;    // 67.5 to 112.5 degrees is set to 90 degrees (1.57079633 radians).
    const float q4 = -0.414213562;    // 112.5 to 157.5 degrees is set to 135 degrees (2.35619449 radians).


    float t = gy / (gx + 1); // (gx + 1) for preventing division by zero
    g =  abs(gx) + abs(gy);  
    
    std::cout << "t: " << t << " ";
    
   
    if ( t < q1 ) {
        m = 0;
    } 
    else if ( t < q2 ) { // 45 degrees
        m = 1;
    }
    else if ( t < q3) { // 90 degress
        m = 2;
    }
    else if ( t < q4 ) { // 135 degress
        m = 3;
    }            
    else {
        m = 0;
    }
    
    return OpCannyMagnitudeAndOrientation{g, m};
/*
    
    int top = 0;
    int bottom = height - radius * 2;
    int right = width - radius * 2;
    int left = 0;
        
    int startY  = top;
    int startX  = left;
    
    int stopY   = bottom;  
    int stopX  = right;   
    
    
    int offset = stride - width + 2 * radius;
    
    
    // adjacent pixels
    int p1 = 0; 
    int p2 = 0;    
    
    int g;
        
    for (int y = startY; y < stopY; y++)
    {
        for (int x = startX; x < stopX; x++, om++, gx++, gy++)
        {
            //t = atan2 (*gradientGy, *gradientGx);
            t = arctan2 (*gy, *gx);
            g =  abs(*gx) + abs(*gy);  
            
            if (t < 0 ) {
                t += M_PI;
            }
           
            if ( t < q1 ) {
                p1 = abs(*(gx - 1)) + abs(*(gy - 1));
                p2 = abs(*(gx + 1)) + abs(*(gy + 1));
            } 
            else if ( t < q2 ) { // 45 degrees
                p1 = abs(*(gx - stride - 1)) + abs(*(gy - stride - 1));
                p2 = abs(*(gx + stride + 1)) + abs(*(gy + stride + 1));
            }
            else if ( t < q3) { // 90 degress
                p1 = abs(*(gx - stride)) + abs(*(gy - stride));
                p2 = abs(*(gx + stride)) + abs(*(gy + stride));
            }
            else if ( t < q4 ) { // 135 degress
                p1 = abs(*(gx - stride + 1)) + abs(*(gy - stride + 1));
                p2 = abs(*(gx + stride - 1)) + abs(*(gy + stride - 1));
            }            
            else {
                p1 = abs(*(gx - 1)) + abs(*(gy - 1));
                p2 = abs(*(gx + 1)) + abs(*(gy + 1));
            }
        }    
        om += offset; 
        gy += offset; 
        gx += offset; 
    }   
    
        float  angle, r;
        const float coeff_1 = 0.7853; //M_PI/4;
        const float coeff_2 = 2.3561; //3*coeff_1;
        float abs_y = fabsf(y) + 1e-10;      // kludge to prevent 0/0 condition
        if (x>=0)
        {
            r = (x - abs_y) / (x + abs_y);
            angle = coeff_1 - coeff_1 * r;
        }
        else
        {
            r = (x + abs_y) / (abs_y - x);
            angle = coeff_2 - coeff_1 * r;
        }
        return (y < 0) ? - angle : angle ; // negate if in quad III or IV
        
        */
        
}    

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

  void Compute2ndDerivativePos();

  void ZeroCrossing();

  void HysteresisThresholding();
  
  /** Edge linking funciton */
  void FollowEdge(int imageStride, int imageWidth, int imageHeight, HysteresisQueue& queue,
                  float* input, float* output);  
  
  void VerifyEdge(int x, int y, int imageStride, int imageWidth, int imageHeight, HysteresisQueue& queue,
                  float* input, float* output);  
  
  void Multiply( int stride, int height, 
             float* input1, float* input2,  float* output );
             
  void Compute2ndDerivativeBondaries(const int imageStride, const int imageWidth, 
                                const int imageHeight, 
                                const float* gaussianImage, const float* dxImage, 
                                float* outputImage);
  
  /** This methos is used to calculate the 2nd derivative for 
   * non-boundary pixels. It is called by the ThreadedCompute2ndDerivative 
   * method. */  
  OutputImagePixelType ComputeCannyEdge(const NeighborhoodType &it,
                                        void *globalData );

   
  void ComputeGradient (const float* input, float* output);
                                                     
  float* AllocateAlignedBuffer ( int width, int height );                                                                                                                                         
  int  CalculateAlignedStride (int width, int height, int typeSize, int alignInBytes);                                                                                                                                         
  int  CalculateAlignedChunk ( int height );                                                                                                                                         
                                             
  //void MoveToAlignedBuffer ( const OutputImagePixelType * input, OutputImagePixelType ** output,
  //                            int width, int height );                                                                                                                                         
                                             
  float MaxGradient ( const float* gx, const float* gy );                                                                                  
  void MaxMin ( const float* buffer, int* max, int* min, int radius );
  
  void VerifyThreshold( float* out, int g, int x, int y );
  
  void NonMaximumSuppression(float* output, float* om);  
  void NonMaximumSuppressionBoundaries(float* output, float* om);  
   //TODO as

                          
  void HysteresisThresholding( float* input ); 
                                                                              
  /** This allocate storage for m_UpdateBuffer, m_UpdateBuffer1 */
  void AllocateUpdateBuffer();


    inline static float arctan2(int y, int x)
    {
        float  angle, r;
        const float coeff_1 = 0.7853; //M_PI/4;
        const float coeff_2 = 2.3561; //3*coeff_1;
        int abs_y = abs(y) + 1;      // kludge to prevent 0/0 condition
        if (x >=0)
        {
            r = (x - abs_y) / (x + abs_y);
            angle = coeff_1 - coeff_1 * r;
        }
        else
        {
            r = (x + abs_y) / (abs_y - x);
            angle = coeff_2 - coeff_1 * r;
        }
        return (y < 0) ? M_PI - angle : angle; // negate if in quad III or IV
    }                       
    
    int GetGaussianKernelWidth(){
        if (m_GaussianKernelWidth == 0) {
            m_GaussianKernelWidth = (int) (m_Variance[0] * 6);
            m_GaussianKernelWidth = m_GaussianKernelWidth % 2 == 0 ? 
                                    m_GaussianKernelWidth - 1 : 
                                    m_GaussianKernelWidth;
        }
        return m_GaussianKernelWidth;        
    }
    
//    
//    int* CreateDiscreteGaussianKernel1D ( )
//    {
//        PRINT_LABEL ("CreateDiscreteGaussianKernel1D");
//    
//        int *kernel __attribute__ ((aligned(16))) = new int[GetGaussianKernelWidth()]; //****alocar dinamicamente
//        int radius = GetGaussianKernelWidth() / 2;
//        double value = 0;
//        double min = std::numeric_limits<double>::max();
//        PRINT (GetGaussianKernelWidth());
//        PRINT_LABEL ("float gaussian");
//        
//        for ( int x = -radius, i = 0; i < GetGaussianKernelWidth(); x++, i++ )
//        {
//           // kernel[i] = (int) ( ( 1 / ( sqrt( 2 * M_PI * pow ( sigma, 2 ) ) ) * 
//           //                   exp ( pow ( x, 2 ) / ( -2 * pow ( sigma, 2 ) ) ) / 
//           //                   ( sqrt( 2 * M_PI ) * sigma ) ) );
//           
//             value = exp ( pow ( x, 2 ) / ( -2 * pow ( m_Variance[0], 2 ) ) ) / ( sqrt( 2 * M_PI ) * m_Variance[0] );
//             if (value < min) min = value;
//             PRINT_INLINE(value);
//             //cout <<  min << " ";
//        }   
//        
//        double ratio = (1 / min);
//        PRINT_LABEL ("discrete gaussian");
//    
//        for ( int x = -radius, i = 0; i < GetGaussianKernelWidth(); x++, i++ )
//        {
//           // kernel[i] = (int) ( ( 1 / ( sqrt( 2 * M_PI * pow ( sigma, 2 ) ) ) * 
//           //                   exp ( pow ( x, 2 ) / ( -2 * pow ( sigma, 2 ) ) ) / 
//           //                   ( sqrt( 2 * M_PI ) * sigma ) ) );
//           
//             value = exp ( pow ( x, 2 ) / ( -2 * pow ( m_Variance[0], 2 ) ) ) / ( sqrt( 2 * M_PI ) * m_Variance[0] );
//             kernel[i] = (int)(value * ratio);
//             PRINT_INLINE(kernel[i]);
//        } 
//        
//                        
//        return kernel;
//    
//        //( 1 / ( sqrt(2*pi*sigma^2) )
//    
//    
//    /*
//    double g = 0;
//            for (double ySubPixel = y - 0.5; ySubPixel < y + 0.6; ySubPixel += 0.1)
//            {
//                for (double xSubPixel = x - 0.5; xSubPixel < x + 0.6; xSubPixel += 0.1)
//                {
//                    g = g + ((1 / (2 * Math.PI * theta * theta)) * Math.pow(Math.E, -(xSubPixel * 
//                    xSubPixel + ySubPixel * ySubPixel) / (2 * theta * theta)));
//                }
//            }
//            g = g / 121;
//            // System.out.println(g);
//            return g;
//    */
//    }
//    

    
    float* gaussianKernel1D ( )
    {
        PRINT_LABEL ("CreateDiscreteGaussianKernel1D");
        
        int kernelWidth = GetGaussianKernelWidth();
        float* kernel __attribute__ ((aligned(16))) = new float[kernelWidth];
        
        
        OpGaussianOperator<float, 1> oper;
        oper.SetVariance(m_Variance[0]);
        oper.SetMaximumKernelWidth(kernelWidth);
        //oper.SetMaximumError(m_MaximumError[i]);        
    
        
        std::vector<double> coeff = oper.GetCoefficients();
        std::vector<double>::iterator it;
              
//         cout <<  "size " << coeff.size() << endl;
         int i = 0;
        for (it = coeff.begin(); it < coeff.end(); ++it)
          {
           kernel[i++] = *it;
//            cout <<  *it << " ";
          }
//          cout <<  endl;        
    
//        float *kernel __attribute__ ((aligned(16))) = new float[GetGaussianKernelWidth()];
//        int radius = GetGaussianKernelWidth() / 2;
//        double sum = 0;
//        double min = std::numeric_limits<double>::max();
//        PRINT (GetGaussianKernelWidth());
//        PRINT_LABEL ("float gaussian");
//        
//        for ( int x = -radius, i = 0; i < GetGaussianKernelWidth(); x++, i++ )
//        {
//             double value = exp ( pow ( x, 2 ) / ( -2 * pow ( m_Variance[0], 2 ) ) ) / ( sqrt( 2 * M_PI ) * m_Variance[0] );
//             kernel[i] = value;
//             sum += value;
//             PRINT_INLINE(value);
//        }   
//        
//        PRINT_LABEL ("discrete gaussian");
//    
//        for ( int x = -radius, i = 0; i < GetGaussianKernelWidth(); x++, i++ )
//        {
//             kernel[i] /= sum;
//             cout << kernel[i] << " ";
//             PRINT_INLINE(kernel[i]);
//        } 
                        
        return kernel;
    }
        
    
    

float* gaussianKernel2D() {
 
    
    int kernelWidth = GetGaussianKernelWidth();
    
    const int radius =  kernelWidth / 2;
    
 
    float* kernel = AllocateAlignedBuffer(kernelWidth, kernelWidth);
    int kernelStride = CalculateAlignedStride(kernelWidth, kernelWidth, sizeof(float), ALIGMENT_BYTES);

    for (int i = 0; i < kernelStride * kernelWidth; i++) {
         kernel[i] = 0;
    }       
    
    #ifdef DEBUG 
    cout << endl;
    cout << "Gaussian2D kernel" << endl;
    cout << "kernelWidth " << kernelWidth << endl;
    #endif
    
    //http://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm
    for(int y = -radius; y < radius + 1; ++y) {
        for(int x = -radius; x < radius + 1; ++x) {
            float value = exp( (pow(x,2) + pow(y,2)) / (-2 * pow(m_Variance[0], 2))) / (2 * M_PI * pow(m_Variance[0], 2));       
            kernel[(y + radius) * kernelStride + (x + radius)] = value;
//            #ifdef DEBUG 
            cout << value << " ";
//            #endif
        }
//        #ifdef DEBUG 
        cout << endl;
//        #endif
    }
    return kernel;
}





void scGaussian7SSE (const int imageStride, const int imageWidth, const int imageHeight, 
                     const float* inputImage, float* outputImage, const float* kernel) {

    const int kernelWidth = 7;
    const int halfKernel = kernelWidth / 2;
    const int yBlock = kernelWidth;
    
    int startX  = 0;
    int stopX   = imageWidth;
    int startY  = 0;
    int stopY   = imageHeight - 2 * (kernelWidth / 2);
    const int kernelOffset = kernelWidth -  1;

      
    #pragma omp parallel for shared (inputImage, outputImage) 
    for (int y = startY; y < stopY; ++y) {
     
        //Load kernel lines. Make it const, so we don't have to load every time
        register __m128 inv0 = _mm_load_ps(kernel);                        PRINT_VECTOR(inv0);
        const register __m128 kvy0 = _mm_shuffle_ps(inv0, inv0, 0);         PRINT_VECTOR(kvy0);
        const register __m128 kvy1 = _mm_shuffle_ps(inv0, inv0, 85);        PRINT_VECTOR(kvy1);
        const register __m128 kvy2 = _mm_shuffle_ps(inv0, inv0, 170);       PRINT_VECTOR(kvy2);
        const register __m128 kvy3 = _mm_shuffle_ps(inv0, inv0, 255);       PRINT_VECTOR(kvy3);

        register __m128 kvx0 = _mm_load_ps(kernel);                        PRINT_VECTOR(kvx0);
        register __m128 kvx1 = _mm_load_ps(kernel + 4);                    PRINT_VECTOR(kvx1);
        register __m128 kvx2 = _mm_setzero_ps();                           PRINT_VECTOR(kvx2);


         //vectors that will hold y dot product results
        __m128 sum0, sum1, sum2;
        sum0 = sum1 = sum2 = _mm_setzero_ps();
        
        PRINT_LABEL("inv"); 
        PRINT(y); 
        
        //calculate y dot products
        
        //x
        inv0 = _mm_load_ps(&inputImage[y * imageStride]);                            PRINT_VECTOR(inv0);
        register __m128 inv1 = _mm_load_ps(&inputImage[(y + 1) * imageStride]);      PRINT_VECTOR(inv1);
        register __m128 inv2 = _mm_load_ps(&inputImage[(y + 2) * imageStride]);      PRINT_VECTOR(inv2);
        sum0 += kvy0 * inv0;    PRINT_VECTOR(sum0);
        sum0 += kvy1 * inv1;    PRINT_VECTOR(sum0);
        sum0 += kvy2 * inv2;    PRINT_VECTOR(sum0);
        
        inv0 = _mm_load_ps(&inputImage[(y + 3) * imageStride]);      PRINT_VECTOR(inv0);
        inv1 = _mm_load_ps(&inputImage[(y + 4) * imageStride]);      PRINT_VECTOR(inv1);
        inv2 = _mm_load_ps(&inputImage[(y + 5) * imageStride]);      PRINT_VECTOR(inv2);
        sum0 += kvy3 * inv0;    PRINT_VECTOR(sum0);
        sum0 += kvy2 * inv1;    PRINT_VECTOR(sum0);
        sum0 += kvy1 * inv2;    PRINT_VECTOR(sum0);
        
        inv0 = _mm_load_ps(&inputImage[(y + 6) * imageStride]);      PRINT_VECTOR(inv0);
        sum0 += kvy0 * inv0;    PRINT_VECTOR(sum0);
        
        //x + 4
        inv0 = _mm_load_ps(&inputImage[y * imageStride + 4]);            PRINT_VECTOR(inv0);
        inv1 = _mm_load_ps(&inputImage[(y + 1) * imageStride + 4]);      PRINT_VECTOR(inv1);
        inv2 = _mm_load_ps(&inputImage[(y + 2) * imageStride + 4]);      PRINT_VECTOR(inv2);
        sum1 += kvy0 * inv0;    PRINT_VECTOR(sum1);
        sum1 += kvy1 * inv1;    PRINT_VECTOR(sum1);
        sum1 += kvy2 * inv2;    PRINT_VECTOR(sum1);
        
        inv0 = _mm_load_ps(&inputImage[(y + 3) * imageStride + 4]);      PRINT_VECTOR(inv0);
        inv1 = _mm_load_ps(&inputImage[(y + 4) * imageStride + 4]);      PRINT_VECTOR(inv1);
        inv2 = _mm_load_ps(&inputImage[(y + 5) * imageStride + 4]);      PRINT_VECTOR(inv2);
        sum1 += kvy3 * inv0;    PRINT_VECTOR(sum1);
        sum1 += kvy2 * inv1;    PRINT_VECTOR(sum1);
        sum1 += kvy1 * inv2;    PRINT_VECTOR(sum1);
        
        inv0 = _mm_load_ps(&inputImage[(y + 6) * imageStride + 4]);      PRINT_VECTOR(inv0);
        sum1 += kvy0 * inv0;    PRINT_VECTOR(sum1);
        
        for (int x = 0; x < stopX; x += 4) {
            
            PRINT_LINE(); 
            PRINT(x); 
            PRINT_VECTOR(sum0)
            PRINT_VECTOR(sum1)
            
            inv0 = _mm_load_ps(&inputImage[y * imageStride + x + 8]);            PRINT_VECTOR(inv0);
            inv1 = _mm_load_ps(&inputImage[(y + 1) * imageStride + x + 8]);      PRINT_VECTOR(inv1);
            inv2 = _mm_load_ps(&inputImage[(y + 2) * imageStride + x + 8]);      PRINT_VECTOR(inv2);
            sum2 += kvy0 * inv0;                                                 PRINT_VECTOR(sum2);
            sum2 += kvy1 * inv1;                                                 PRINT_VECTOR(sum2);
            sum2 += kvy2 * inv2;                                                 PRINT_VECTOR(sum2);
            
            inv0 = _mm_load_ps(&inputImage[(y + 3) * imageStride + x + 8]);      PRINT_VECTOR(inv0);
            inv1 = _mm_load_ps(&inputImage[(y + 4) * imageStride + x + 8]);      PRINT_VECTOR(inv1);
            inv2 = _mm_load_ps(&inputImage[(y + 5) * imageStride + x + 8]);      PRINT_VECTOR(inv2);
            sum2 += kvy3 * inv0;                                                 PRINT_VECTOR(sum2);
            sum2 += kvy2 * inv1;                                                 PRINT_VECTOR(sum2);
            sum2 += kvy1 * inv2;                                                 PRINT_VECTOR(sum2);
            
            inv0 = _mm_load_ps(&inputImage[(y + 6) * imageStride + x + 8]);      PRINT_VECTOR(inv0);
            sum2 += kvy0 * inv0;                                                 PRINT_VECTOR(sum2);
            
            // ---------   ---------
            // |0|1|2|3|   |4|5|6|-|
            // ---------   ---------
            inv0 = _mm_dp_ps(sum0, kvx0, 241) + 
                   _mm_dp_ps(sum1, kvx1 , 113);          PRINT_VECTOR(inv0);
                   
            // ---------   ---------
            // |3|0|1|2|   |3|4|5|6|
            // ---------   ---------
            ROTATE_RIGHT(kvx1);                                                     
            ROTATE_RIGHT_BLEND(kvx0, kvx1);   
            inv0 += _mm_dp_ps(sum0, kvx0, 226) + 
                   _mm_dp_ps(sum1, kvx1, 242);          PRINT_VECTOR(inv0);
            
            PRINT_LABEL("Terceiro"); 
            // ---------   ---------   ---------
            // |2|3|0|1|   |2|3|4|5|   |6|-|-|-|
            // ---------   ---------   ---------
            ROTATE_RIGHT_BLEND(kvx1, kvx2); 
            ROTATE_RIGHT_BLEND(kvx0, kvx1); 
            inv0 += _mm_dp_ps(sum0, kvx0, 196) + 
                   _mm_dp_ps(sum1, kvx1, 244) + 
                   _mm_dp_ps(sum2, kvx2 , 20);          PRINT_VECTOR(inv0);
            
            PRINT_LABEL("Quarto"); 
            PRINT_VECTOR(sum0);
            PRINT_VECTOR(sum1);
            PRINT_VECTOR(sum2);
            
            // ---------   ---------   ---------
            // |1|2|3|0|   |1|2|3|4|   |5|6|-|-|
            // ---------   ---------   ---------
            ROTATE_RIGHT(kvx2);
            ROTATE_RIGHT_BLEND(kvx1, kvx2);
            ROTATE_RIGHT_BLEND(kvx0, kvx1);                                         
            inv0 += _mm_dp_ps(sum0, kvx0, 136) + 
                   _mm_dp_ps(sum1, kvx1, 248) + 
                   _mm_dp_ps(sum2, kvx2, 56);          PRINT_VECTOR(inv0);
            
            PRINT_LABEL("Rotate"); 
            PRINT_VECTOR(kvx0);
            PRINT_VECTOR(kvx1);
            PRINT_VECTOR(kvx2);

            ROTATE_RIGHT(kvx0);
            //FIXME O erro está aqui. kvx1 não está carregado de forma correta. Dar um shuffle com inv1 para pegar de volta os elementos.
            kvx1 = _mm_blend_ps(kvx1, kvx2, 7);
            ROTATE_RIGHT(kvx1);
            
            
            PRINT_LABEL("sum"); 
            PRINT((y + halfKernel) * imageStride + (x + halfKernel));
            _mm_storeu_ps(&outputImage[(y + halfKernel) * imageStride + (x + halfKernel)], inv0);                 PRINT_VECTOR(inv0);
            PRINT(outputImage[(y + halfKernel) * imageStride + (x + halfKernel)]);
            PRINT(outputImage[(y + halfKernel) * imageStride + (x + halfKernel) + 1]);
            PRINT(outputImage[(y + halfKernel) * imageStride + (x + halfKernel) + 2]);
            PRINT(outputImage[(y + halfKernel) * imageStride + (x + halfKernel) + 3]);            
            
            sum0 = sum1;
            sum1 = sum2;
            sum2 = _mm_setzero_ps();
        }
     
    }
}


//
//float* gaussianKernel2D() {
// 
//    
//    const int radius = GetGaussianKernelWidth() / 2;
//    
//    int kernelWidth = GetGaussianKernelWidth();
// 
//    float* kernel = AllocateAlignedBuffer(kernelWidth, kernelWidth);
//    int kernelStride = CalculateAlignedStride(kernelWidth, kernelWidth, sizeof(float), ALIGMENT_BYTES);
//
//    for (int i = 0; i < kernelStride * kernelWidth; i++) {
//         kernel[i] = 0;
//    }       
//    
//    #ifdef DEBUG 
//    cout << endl;
//    cout << "Gaussian2D kernel" << endl;
//    cout << "kernelWidth " << kernelWidth << endl;
//    #endif
//    
//    double sum = 0;
//    
//    //http://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm
//    for(int y = -radius; y < radius + 1; ++y) {
//        for(int x = -radius; x < radius + 1; ++x) {
//            double value = exp( (pow(x,2) + pow(y,2)) / (-2 * pow(m_Variance[0], 2))) / (2 * M_PI * pow(m_Variance[0], 2));       
//            kernel[(y + radius) * kernelStride + (x + radius)] = value;
//            sum += value;
//        }
//    }
//    
//    for(int y = -radius; y < radius + 1; ++y) {
//        for(int x = -radius; x < radius + 1; ++x) {
//            kernel[(y + radius) * kernelStride + (x + radius)] /= sum;
////            #ifdef DEBUG 
//            cout << kernel[(y + radius) * kernelStride + (x + radius)] << " ";
////            #endif
//        }
////        #ifdef DEBUG 
//        cout << endl;
////        #endif
//    }    
//    return kernel;
//}

        
    
    
//ok
void processTopLeftCorner(const int imageStride, const int imageWidth, const int imageHeight, 
                          const int kernelStride, const int kernelWidth, const float* kernel, const float* inputImage, float* outputImage) {


//    cout << endl << "processTopLeftCorner" << endl;
    int halfKernel = kernelWidth / 2;

    for(int y = 0; y < halfKernel; ++y) {
        for(int x = 0; x < halfKernel; ++x) {

            float value = 0;

            
            int left = halfKernel - x;
            int top = halfKernel - y;
          
            //traverse corner ok
            float v = inputImage[0];
            for (int i = 0; i < top + 1; ++i) {
                for (int j = 0; j < left + 1; ++j) {
                    value += kernel[i * kernelStride + j] * v;
                }
            }
            
            //traverse top ok
            for (int i = 0; i < top + 1; ++i) {
                for (int j = left + 1, jj = 1; j < kernelWidth; ++j, ++jj) {
                    value += kernel[i * kernelStride + j] * inputImage[jj];
                }
            }
            
            //traverse left ok
            for (int i = top + 1, ii = imageStride; i < kernelWidth; ++i, ii += imageStride) {
                for (int j = 0; j < left + 1; ++j) {
                    value += kernel[i * kernelStride + j] * inputImage[ii];
                }
            }
            
            //traverse middle ok
            for (int i = top + 1, ii = 1; i < kernelWidth; ++i, ++ii) {
                for (int j = left + 1, jj = 1; j < kernelWidth; ++j, ++jj) {
                    value += kernel[i * kernelStride + j] * inputImage[ii * imageStride + jj];
                }
            }            
            
            outputImage[y * imageStride + x] = value;
                
        }                 
    }                 
}

//ok
void processTopMiddle(const int imageStride, const int imageWidth, const int imageHeight, 
                          const int kernelStride, const int kernelWidth, const float* kernel, const float* inputImage, float* outputImage) {
                           
//    cout << endl << "processTopMiddle" << endl;
    int halfKernel = kernelWidth / 2;

    for(int y = 0; y < halfKernel; ++y) {
        for(int x = halfKernel; x < imageWidth - halfKernel; ++x) {

            float value = 0;

            
            int top = halfKernel - y;
          
            //traverse top ok
            for (int i = 0; i < top + 1; ++i) {
                for (int j = 0, jj = 0; j < kernelWidth; ++j, ++jj) {
                    value += kernel[i * kernelStride + j] * inputImage[(x - halfKernel) + jj];
                }
            }
            
            //traverse middle
            for (int i = top + 1, ii = 1; i < kernelWidth; ++i, ++ii) {
                for (int j = 0, jj = 0; j < kernelWidth; ++j, ++jj) {
                    value += kernel[i * kernelStride + j] * inputImage[ii * imageStride + (x - halfKernel) + jj];
                }
            }            
            
            outputImage[y * imageStride + x] = value;
                                    
            
        }                 
    }                 
}

//ok
void processTopRightCorner(const int imageStride, const int imageWidth, const int imageHeight, 
                          const int kernelStride, const int kernelWidth, const float* kernel, const float* inputImage, float* outputImage) {
                           
//    cout << endl << "processTopRightCorner" << endl;
    int halfKernel = kernelWidth / 2;

    for(int y = 0; y < halfKernel; ++y) {
        for(int x = imageWidth - 1; x > imageWidth - halfKernel - 1; --x) {

            float value = 0;

            
            int top = halfKernel - y;
            int right = halfKernel - (imageWidth - x - 1);
          
            //traverse corner ok 
            float v = inputImage[imageWidth - 1];
            for (int i = 0; i < top + 1; ++i) {
                for (int j = kernelWidth - 1; j >= kernelWidth - right - 1; --j) {
                    value += kernel[i * kernelStride + j] * v;
                }
            }

            //traverse top ok 
            for (int i = 0; i < top + 1; ++i) {
                for (int j = kernelWidth - right - 2, jj = imageWidth - 2; j >= 0; --j, --jj) {
                    value += kernel[i * kernelStride + j] * inputImage[jj];
                }
            }
            
            //traverse right ok 
            for (int i = top + 1, ii = imageStride + imageWidth - 1; i < kernelWidth; ++i, ii += imageStride) {
                for (int j = kernelWidth - 1; j >=  kernelWidth - right - 1; --j) {
                    value += kernel[i * kernelStride + j] * inputImage[ii];
                }
            }
            
            //traverse middle ok
            for (int i = top + 1, ii = 1; i < kernelWidth; ++i, ++ii) {
                for (int j = kernelWidth - right - 2, jj = imageWidth - 2; j >= 0; --j, --jj) {
                    value += kernel[i * kernelStride + j] * inputImage[ii * imageStride + jj];
                }
            }           
            
            outputImage[y * imageStride + x] = value;

            
        }                 
    }                 
}                           
                           

void processRightMiddle(const int imageStride, const int imageWidth, const int imageHeight, 
                          const int kernelStride, const int kernelWidth, const float* kernel, const float* inputImage, float* outputImage) {
    
//    cout << endl << "processRightMiddle" << endl;
    int halfKernel = kernelWidth / 2;
    
    for(int x = imageWidth - 1; x > imageWidth - halfKernel - 1; --x) {
        for(int y = halfKernel; y < imageHeight - halfKernel; ++y) {
    
            float value = 0;
    
    
            int right = halfKernel - (imageWidth - x - 1);
          
            // traverse right ok
            for (int j = kernelWidth - 1; j >= kernelWidth - right - 1; --j) {
                for (int i = 0, ii = 0; i < kernelWidth; ++i, ++ii) {
                    value += kernel[i * kernelStride + j] * inputImage[((y - halfKernel) + ii) * imageStride + imageWidth - 1];
                }
            }
            
            
            // traverse middle ok
            for (int j = kernelWidth - right - 2, jj = 0; j >= 0; --j, ++jj) {
                for (int i = 0, ii = 0; i < kernelWidth; ++i, ++ii) {
                    value += kernel[i * kernelStride + j] * inputImage[(((y - halfKernel) + ii) * imageStride + imageWidth - 2) - jj];
                }
            }
            
            outputImage[y * imageStride + x] = value;
            
        }                 
    }                 
}

void processBottomRightCorner(const int imageStride, const int imageWidth, const int imageHeight, 
                          const int kernelStride, const int kernelWidth, const float* kernel, const float* inputImage, float* outputImage) {
                           
//    cout << endl << "processBottomRightCorner" << endl;
    int halfKernel = kernelWidth / 2;
    
    
    for(int y = imageHeight - 1; y > imageHeight - halfKernel - 1; --y) {
        for(int x = imageWidth - 1; x > imageWidth - halfKernel - 1; --x) {
           
            float value = 0;
           
            int right = halfKernel - (imageWidth - x - 1);
            int bottom = halfKernel - (imageHeight - y - 1);
           
            //traverse corner ok
            float v = inputImage[(imageStride * (imageHeight - 1)) + imageWidth - 1];
            for (int i = kernelWidth - 1; i >= kernelWidth - bottom - 1; --i) {
                for (int j = kernelWidth - 1; j >= kernelWidth - right - 1; --j) {
                    value += kernel[i * kernelStride + j] * v;
                }
            }          
            
            //traverse bottom ok
            for (int j = kernelWidth - right - 2, jj = 2; j >= 0; --j, ++jj) {
                int v = inputImage[(imageStride * (imageHeight - 1)) + imageWidth - jj];
                for (int i = kernelWidth - 1; i >= kernelWidth - bottom - 1; --i) {
                    value += kernel[i * kernelStride + j] * v;
                }
            }
            
            //traverse right ok
            for (int i = kernelWidth - bottom - 2, ii = imageHeight - 2; i >= 0; --i, --ii) {
                int v = inputImage[(ii * imageStride + imageWidth) - 1];
                for (int j = kernelWidth - 1; j >= kernelWidth - right - 1; --j) {
                    value += kernel[i * kernelStride + j] * v;
                }
            }
            
            //traverse middle ok
            for (int i = kernelWidth - bottom - 2, ii = imageHeight - 2; i >= 0; --i, --ii) {
                for (int j = kernelWidth - right - 2, jj = 2; j >= 0; --j, ++jj) {
                    value += kernel[i * kernelStride + j] * inputImage[ii * imageStride + imageWidth - jj];
                }
            }                        
            outputImage[y * imageStride + x] = value;
            
        }                 
    }                 
}

//ok
void processBottomMiddle(const int imageStride, const int imageWidth, const int imageHeight, 
                          const int kernelStride, const int kernelWidth, const float* kernel, const float* inputImage, float* outputImage) {
                           
//    cout << endl << "processBottomMiddle" << endl;
    int halfKernel = kernelWidth / 2;
    
    for(int y = imageHeight - 1; y > imageHeight - halfKernel - 1; --y) {
        for(int x = halfKernel; x < imageWidth - halfKernel; ++x) {
    
            float value = 0;
           
            int bottom = halfKernel - (imageHeight - y - 1);
       
            //traverse bottom
            for (int i = kernelWidth - 1; i > kernelWidth - bottom - 1; --i) {
                int jj = ((imageHeight - 1) * imageStride) + (x - halfKernel);
                for (int j = 0; j < kernelWidth; ++j) {
                    value += kernel[i * kernelStride + j] * inputImage[j + jj];
                }
            }
            
            //traverse middle
            for (int i = kernelWidth - bottom - 1, ii = imageHeight - 1; i >= 0; --i, --ii) {
                int jj = (ii * imageStride) + (x - halfKernel);
                for (int j = 0; j < kernelWidth; ++j) {
                    value += kernel[i * kernelStride + j] * inputImage[j + jj];
                }
            }
            outputImage[y * imageStride + x] = value;
                                    
        }                 
    }                 
}

//ok

void processBottomLeftCorner(const int imageStride, const int imageWidth, const int imageHeight, 
                          const int kernelStride, const int kernelWidth, const float* kernel, const float* inputImage, float* outputImage) {

                           
//    cout << endl << "processBottomLeftCorner" << endl;
    int halfKernel = kernelWidth / 2;
    
    
    for(int y = imageHeight - 1; y > imageHeight - halfKernel - 1; --y) {
        for(int x = 0; x < halfKernel; ++x) {
    
            float value = 0;
    
            int left = halfKernel - x;
            int bottom = halfKernel - (imageHeight - y - 1);
           
            //traverse corner
            float v = inputImage[imageStride * (imageHeight - 1)];
            for (int i = kernelWidth - 1; i >= kernelWidth - bottom - 1; --i) {
                for (int j = 0; j < left + 1; ++j) {
                    value += kernel[i * kernelStride + j] * v;
                }
            }          
            
            //traverse bottom
            for (int j = left + 1, jj = 1; j < kernelWidth; ++j, ++jj) {
                int v = inputImage[(imageHeight - 1) * imageStride + jj];
                for (int i = kernelWidth - 1; i >= kernelWidth - bottom - 1; --i) {
                    value += kernel[i * kernelStride + j] * v;
                }
            }
            
            //traverse left
            for (int i = kernelWidth - bottom - 2, ii = imageHeight - 2; i >= 0; --i, --ii) {
                int v = inputImage[ii * imageStride];
                for (int j = 0; j < left + 1; ++j) {
                    value += kernel[i * kernelStride + j] * v;
                }
            }
            
            for (int i = kernelWidth - bottom - 2, ii = imageHeight - 2; i >= 0; --i, --ii) {
                for (int j = left + 1, jj = 1; j < kernelWidth; ++j, ++jj) {
                    value += kernel[i * kernelStride + j] * inputImage[ii * imageStride + jj];
                }
            }                        
            
            outputImage[y * imageStride + x] = value;
            
        }                 
    }                 
}

//ok

void processLeftMiddle(const int imageStride, const int imageWidth, const int imageHeight, 
                          const int kernelStride, const int kernelWidth, const float* kernel, const float* inputImage, float* outputImage) {
                           
//    cout << endl << "processLeftMiddle" << endl;
    int halfKernel = kernelWidth / 2;
    
    for(int x = 0; x < halfKernel; ++x) {
        for(int y = halfKernel; y < imageHeight - halfKernel; ++y) {
    
            float value = 0;
           
            int left = halfKernel - x;
          
       
            //traverse right
            for (int j = 0; j < left; ++j) {
                for (int i = 0; i < kernelWidth; ++i) {
                    value += kernel[i * kernelStride + j] * inputImage[((y - halfKernel) + i) * imageStride];
                }
            }
            
            //traverse middle
            for (int j = left, jj = 0; j < kernelWidth; ++j, ++jj) {
                for (int i = 0; i < kernelWidth; ++i) {
                    value += kernel[i * kernelStride + j] * inputImage[((y - halfKernel) + i) * imageStride + jj];
                }
            }
                     
            outputImage[y * imageStride + x] = value;
            
        }                 
    }                 
}


void processBoundaries(const int imageStride, const int imageWidth, const int imageHeight, 
                       const int kernelStride, int kernelWidth, 
                       const float* inputImage, float* outputImage, const float* kernel) {
    
        
    processTopLeftCorner (imageStride, imageWidth, imageHeight, kernelStride, kernelWidth, kernel, inputImage, outputImage);
    processTopMiddle (imageStride, imageWidth, imageHeight, kernelStride, kernelWidth, kernel, inputImage, outputImage);
    processTopRightCorner (imageStride, imageWidth, imageHeight, kernelStride, kernelWidth, kernel, inputImage, outputImage);
    processRightMiddle (imageStride, imageWidth, imageHeight, kernelStride, kernelWidth, kernel, inputImage, outputImage);
    processBottomRightCorner (imageStride, imageWidth, imageHeight, kernelStride, kernelWidth, kernel, inputImage, outputImage);
    processBottomMiddle (imageStride, imageWidth, imageHeight, kernelStride, kernelWidth, kernel, inputImage, outputImage);
    processBottomLeftCorner (imageStride, imageWidth, imageHeight, kernelStride, kernelWidth, kernel, inputImage, outputImage);
    processLeftMiddle (imageStride, imageWidth, imageHeight, kernelStride, kernelWidth, kernel, inputImage, outputImage);

     
}                             
    


//no image vector reuse
void alignedSSENoReuse4SumsConvolve (const int imageStride, const int imageWidth, const int imageHeight, 
                                     const int kernelStride, int kernelWidth, 
                                     const float* inputImage, float* outputImage, const float* kernel) {

    #ifdef DEBUG
        cout << endl;
    #endif
    PRINT(kernelStride);
    int halfKernel = kernelWidth / 2;
    int startX  = 0;
    int stopX   = imageWidth - halfKernel * 2;
    int startY  = 0;
    int stopY   = imageHeight - halfKernel * 2;
                         
    #pragma omp parallel for shared (inputImage, outputImage) 
    for (int y = startY; y < stopY; ++y) {
        for (int x = startX; x < stopX; x += 16) { 
                    //cout << "x: " << x << " y: " << y << endl;
            int yy = y * imageStride;
            register __m128 sum0, sum1, sum2, sum3; 
            sum0 = sum1 = sum2 = sum3 = _mm_setzero_ps();
            for (int r = 0; r < kernelWidth; ++r) {
                const int idxFtmp = r * kernelStride;
                const int idxIntmp = (y + r) * imageStride + x; 
                for (int c = 0; c < kernelWidth; c += 4) {
                    __m128 iv0, iv1, iv2, iv3, iv4;
                    register const __m128 kv = _mm_load_ps(&kernel[idxFtmp + c]);                     PRINT_VECTOR(kv);
                    //cout << "aqui 1" << flush << endl;
                    iv0 = _mm_load_ps(&inputImage[idxIntmp + c]);               PRINT_VECTOR(iv0);
                    iv1 = _mm_load_ps(&inputImage[idxIntmp + c + 4]);           PRINT_VECTOR(iv1);
                    iv2 = _mm_load_ps(&inputImage[idxIntmp + c + 8]);           PRINT_VECTOR(iv2);
                    iv3 = _mm_load_ps(&inputImage[idxIntmp + c + 12]);          PRINT_VECTOR(iv3);
                    iv4 = _mm_load_ps(&inputImage[idxIntmp + c + 16]);          PRINT_VECTOR(iv4);
                    
                    //cout << "aqui 2" << flush << endl;
                    PRINT_LABEL("sum0"); 
                    sum0 += _mm_dp_ps(kv, iv0, 241);    PRINT_VECTOR(sum0); 
                    sum1 += _mm_dp_ps(kv, iv1, 241);    PRINT_VECTOR(sum1);
                    sum2 += _mm_dp_ps(kv, iv2, 241);    PRINT_VECTOR(sum2);
                    sum3 += _mm_dp_ps(kv, iv3, 241);    PRINT_VECTOR(sum3);
                     
                    //cout << "aqui 3" << flush << endl;
                     
                    BLEND_ROTATE4_LEFT(iv0, iv1, iv2, iv3, iv4);

                    PRINT_LABEL("sum1"); 
                    sum0 += _mm_dp_ps(kv, iv0, 242);    PRINT_VECTOR(sum0);
                    sum1 += _mm_dp_ps(kv, iv1, 242);    PRINT_VECTOR(sum1);
                    sum2 += _mm_dp_ps(kv, iv2, 242);    PRINT_VECTOR(sum2);
                    sum3 += _mm_dp_ps(kv, iv3, 242);    PRINT_VECTOR(sum3);
                    
                    //cout << "aqui 4" << flush << endl;
                    
                    BLEND_ROTATE4_LEFT(iv0, iv1, iv2, iv3, iv4);

                    PRINT_LABEL("sum2"); 
                    sum0 += _mm_dp_ps(kv, iv0, 244);    PRINT_VECTOR(sum0);
                    sum1 += _mm_dp_ps(kv, iv1, 244);    PRINT_VECTOR(sum1);
                    sum2 += _mm_dp_ps(kv, iv2, 244);    PRINT_VECTOR(sum2);
                    sum3 += _mm_dp_ps(kv, iv3, 244);    PRINT_VECTOR(sum3);
                    
                    //cout << "aqui 5" << flush << endl;
                    
                    BLEND_ROTATE4_LEFT(iv0, iv1, iv2, iv3, iv4);

                    PRINT_LABEL("sum3"); 
                    sum0 += _mm_dp_ps(kv, iv0, 248);    PRINT_VECTOR(sum0);
                    sum1 += _mm_dp_ps(kv, iv1, 248);    PRINT_VECTOR(sum1);
                    sum2 += _mm_dp_ps(kv, iv2, 248);    PRINT_VECTOR(sum2);
                    sum3 += _mm_dp_ps(kv, iv3, 248);    PRINT_VECTOR(sum3);
                    
                }
            } //for (int r = 0...
            _mm_storeu_ps(&outputImage[(y + halfKernel) * imageStride + (x + halfKernel)], sum0);         PRINT_VECTOR(sum0);
            _mm_storeu_ps(&outputImage[(y + halfKernel) * imageStride + (x + halfKernel) + 4], sum1);     PRINT_VECTOR(sum1);
            _mm_storeu_ps(&outputImage[(y + halfKernel) * imageStride + (x + halfKernel) + 8], sum2);     PRINT_VECTOR(sum2);
            _mm_storeu_ps(&outputImage[(y + halfKernel) * imageStride + (x + halfKernel) + 12], sum3);    PRINT_VECTOR(sum3);
        } //for (int x = 0...
    } //for (int y = 0...
    processBoundaries (imageStride, imageWidth, imageHeight, 
                       kernelStride, kernelWidth, 
                       inputImage, outputImage, kernel);
}


    
inline static void printImage(int imageWidth, int imageHeight, int stride, const float* out) {
    cout << endl;
    for (int y = 0; y < imageHeight; ++y) {
        for (int x = 0; x < imageWidth; ++x) {
            cout << left << setw(9) << out[y * stride + x]; 
        }
        cout << endl;
    }   
}

void sc3SSE (const int imageStride, const int imageWidth, const int imageHeight, 
             const float* inputImage, float* outputImage, const float* kernelX, const float* kernelY) {

    const int kernelWidth = 3;
    const int halfKernel = kernelWidth / 2;
    const int yBlock = kernelWidth;
    
    int startX  = 0;
    int stopX   = imageWidth;
    int startY  = 0;
    int stopY   = imageHeight - 2 * (kernelWidth / 2);
    const int kernelOffset = kernelWidth -  1;

      
    #pragma omp parallel for shared (inputImage, outputImage) 
    for (int y = startY; y < stopY; ++y) {
        const __m128 kv = _mm_load_ps(kernelY);             PRINT_VECTOR(kv);
        const register __m128 kvy0 = _mm_shuffle_ps(kv, kv, 0);            PRINT_VECTOR(kvy0);
        const register __m128 kvy1 = _mm_shuffle_ps(kv, kv, 85);           PRINT_VECTOR(kvy1);
        const register __m128 kvy2 = _mm_shuffle_ps(kv, kv, 170);          PRINT_VECTOR(kvy2);
        register __m128 kvx0 = _mm_load_ps(kernelX);                 PRINT_VECTOR(kvx0);

        __m128 kvx1 = _mm_setzero_ps();                     PRINT_VECTOR(kvx1);
        __m128 sum0, sum1;
        sum0 = sum1 = _mm_setzero_ps();
        PRINT_LABEL("inv"); 
        
        __m128 inv0 = _mm_load_ps(&inputImage[y * imageStride]);            PRINT_VECTOR(inv0);
        __m128 inv1 = _mm_load_ps(&inputImage[(y + 1) * imageStride]);      PRINT_VECTOR(inv1);
        __m128 inv2 = _mm_load_ps(&inputImage[(y + 2) * imageStride]);      PRINT_VECTOR(inv2);
        
        PRINT(y); 
                    
        sum0 += kvy0 * inv0;    PRINT_VECTOR(sum0);
        sum0 += kvy1 * inv1;    PRINT_VECTOR(sum0);
        sum0 += kvy2 * inv2;    PRINT_VECTOR(sum0);
        
        for (int x = 0; x < stopX; x += 4) {
            
            PRINT_LINE(); 
            PRINT(x); 
            PRINT_VECTOR(sum0)
            PRINT_VECTOR(sum1)
            
            inv0 = _mm_load_ps(&inputImage[y * imageStride + x + 4]);            PRINT_VECTOR(inv0);
            inv1 = _mm_load_ps(&inputImage[(y + 1) * imageStride + x + 4]);      PRINT_VECTOR(inv1);
            inv2 = _mm_load_ps(&inputImage[(y + 2) * imageStride + x + 4]);      PRINT_VECTOR(inv2);
            
            sum1 += kvy0 * inv0;    PRINT_VECTOR(sum1);
            sum1 += kvy1 * inv1;    PRINT_VECTOR(sum1);
            sum1 += kvy2 * inv2;    PRINT_VECTOR(sum1);
            
            kvx1 = _mm_setzero_ps();
            
            inv0 = _mm_dp_ps(sum0, kvx0, 113);                                      PRINT_VECTOR(inv0);
            ROTATE_RIGHT(kvx0);                                                     
            inv0 += _mm_dp_ps(sum0, kvx0, 226);                                     PRINT_VECTOR(inv0);
            ROTATE_RIGHT_BLEND(kvx0, kvx1);                                         
            inv0 += _mm_dp_ps(sum0, kvx0, 196) + _mm_dp_ps(sum1, kvx1, 20);         PRINT_VECTOR(inv0);
            ROTATE_RIGHT(kvx1);                                                     
            ROTATE_RIGHT_BLEND(kvx0, kvx1);                                         
            inv0 += _mm_dp_ps(sum0, kvx0, 136) + _mm_dp_ps(sum1, kvx1, 56);         PRINT_VECTOR(inv0);
            ROTATE_RIGHT(kvx0);                                                     
            
            PRINT_LABEL("sum"); 
            PRINT(y * imageStride + x);
            _mm_stream_ps(&outputImage[y * imageStride + x], inv0); PRINT_VECTOR(inv0);
            PRINT(outputImage[y * imageStride + x]);
            PRINT(outputImage[y * imageStride + x + 1]);
            PRINT(outputImage[y * imageStride + x + 2]);
            PRINT(outputImage[y * imageStride + x + 3]);            
            
            sum0 = sum1;
            sum1 = _mm_setzero_ps();
        }
     
    }
}



                   
//    
//    int GetDiscreteGaussianKernel1DSum()
//    {
//        int *kernel = CreateDiscreteGaussianKernel1D();
//        int sum = 0;
//        for (int i = 0; i < GetGaussianKernelWidth(); i++)
//        {
//            sum += kernel[i];
//        } 
//        return sum;        
//    }       
//    
  /** Timer*/  

  StopWatch m_Timer;  

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