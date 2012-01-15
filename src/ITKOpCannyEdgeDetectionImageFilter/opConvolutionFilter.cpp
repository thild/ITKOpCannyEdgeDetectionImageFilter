// 
// opConvutionFilter.cpp
//  
// Author:
//       Tony Alexander Hild <tony_hild@yahoo.com>
// 
// Copyright (c) 2011 
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.


#include "opConvolutionFilter.h"
#include <iostream>
#include <fstream>
#include <iomanip>

//sse and omp
#include <omp.h>
#include <xmmintrin.h>  // SSE  (Required to use the __m128, and __m128d type)
#include <emmintrin.h>  // SSE2 (Required to use the __m128i type)
#include <pmmintrin.h>  // SSE3
 
#ifdef __SSE4_1__
#include <smmintrin.h>  
#endif 
#include <cmath>

using namespace std;


void opConvolve (const int s, const int w, const int h, 
                        const int ks, const int kw, 
                        const float* input, float* output, const float* kernel) {
    
    if(kw < 2) return;
    
    if(kw > w || kw > h) return;
                  
    if(w < ALIGMENT_BYTES || h < ALIGMENT_BYTES){
        alignedConvolve (s, w, h, ks, kw, 
                       input, output, kernel);  
        return;
     
    }                
                  
    switch (kw) {
        case 3:
            sse3Convolve (s, w, h, ks, 
                          input, output, kernel);
            break;
        case 5:
            sse5Convolve (s, w, h, ks, 
                          input, output, kernel);
            break;
        case 7:
            sseNoReuse4Convolve (s, w, h, ks, kw, 
                         input, output, kernel);
            break;
        case 9:
            sse9Convolve (s, w, h, ks,
                          input, output, kernel);
            break;
        case 11:
            sseReuse4Convolve (s, w, h, ks, kw, 
                               input, output, kernel);
            break;
        default:
                sseNoReuse4Convolve (s, w, h, ks, kw, 
                             input, output, kernel);
            break;
    }                  
}
 
void opSeparableConvolve (const int s, const int w, const int h, const int kw, 
                          const float* __restrict input, float* __restrict output, 
                          const float* __restrict kernelX, 
                          const float* __restrict kernelY) {
    if(kw < 2) return;
    
    if(kw > w || kw > h) return;
                  
    if(w < ALIGMENT_BYTES || h < ALIGMENT_BYTES){
        separableConvolve (s, w, h, kw, 
                           input, output, kernelX, kernelY, true);  
        return;
     
    }                
                  
    switch (kw) {
        case 3:
            sc3SSE (s, w, h,
                    input, output, 
                    kernelX, kernelY);
            break;
        case 5:
            sc5SSE (s, w, h,
                    input, output, 
                    kernelX, kernelY);
            break;
        case 7:
            sc7SSE (s, w, h,
                    input, output, 
                    kernelX, kernelY);
            break;
        case 9:
            sc9SSE (s, w, h,
                    input, output, 
                    kernelX, kernelY);
            break;
        default:
            scSSE (s, w, h, kw,
                    input, output, 
                    kernelX, kernelY);
            break;
    }                                             
}
 


void naiveConvolve (const int s, const int w, const int h, 
                    const int ks, int kw, 
                    const float* input, float* output, const float* kernel) {

    int hk = kw / 2;                       
    int startX  = 0;
    int stopX   = w - hk * 2;
    int startY  = 0;
    int stopY   = h - hk * 2;
                         
    #pragma omp parallel for shared (input, output) 
    for (int y = startY; y < stopY; ++y) {
        for (int x = startX; x < stopX; ++x) {
            float sum = 0;
            for (int r = 0; r < kw; ++r) {
                int idxFtmp = r * kw;
                int idxIntmp = (y + r) * w + x;
                for (int c = 0; c < kw; ++c) {
                    sum += kernel[idxFtmp + c] * input[idxIntmp + c];
                }
            } //for (int r = 0...
            output[(y + hk) * w + (x + hk)] = sum;
        } //for (int x = 0...
    } //for (int y = 0...
    
    processBoundaries2D (s, w, h, 
                       ks, kw, 
                       input, output, kernel);
//    printImage(w, h, s, output);
}

void alignedConvolve (const int s, const int w, const int h, 
                      const int ks, const int kw, 
                      const float* input, float* output, const float* kernel) {
                       
    int hk = kw / 2;                       
    int startX  = 0;
    int stopX   = w - hk * 2;
    int startY  = 0;
    int stopY   = h - hk * 2;
         
    #pragma omp parallel for shared (input, output) 
    for (int y = startY; y < stopY; ++y) {
        for (int x = startX; x < stopX; ++x) {
            float sum = 0;
            for (int r = 0; r < kw; ++r) {
                int idxFtmp = r * ks;
                int idxIntmp = (y + r) * s + x;
                for (int c = 0; c < kw; c++) {
                    sum += kernel[idxFtmp + c] * input[idxIntmp + c];
                }
            } //for (int r = 0...
            output[(y + hk) * s + (x + hk)] = sum;
        } //for (int x = 0...
    } //for (int y = 0...
    processBoundaries2D (s, w, h, 
                       ks, kw, 
                       input, output, kernel);
//    printImage(w, h, s, output);
}


//no image vector reuse
void sseNoReuse1Convolve (const int s, const int w, const int h, 
                         const int ks, int kw, 
                         const float* input, float* output, const float* kernel) {
    int hk = kw / 2;                       
    int startX  = 0;
    int stopX   = w - 2 * (kw / 2);
    int startY  = 0;
    int stopY   = h - 2 * (kw / 2);
                         
    #pragma omp parallel for shared (input, output) 
    for (int y = startY; y < stopY; ++y) {
        for (int x = startX; x < stopX; x += 4) {
            register __m128 sum0;
            sum0 = _mm_setzero_ps();
            for (int r = 0; r < kw; ++r) {
                const int idxFtmp = r * ks;
                const int idxIntmp = (y + r) * s + x;
                for (int c = 0; c < kw; c += 4) {
                    __m128 iv0, iv1;
                    register const __m128 kv = _mm_load_ps(&kernel[idxFtmp + c]);                     PRINT_VECTOR(kv);
                    iv0 = _mm_load_ps(&input[idxIntmp + c]);               PRINT_VECTOR(iv0);
                    iv1 = _mm_load_ps(&input[idxIntmp + c + 4]);           PRINT_VECTOR(iv1);
                    
                    PRINT_LABEL("sum0"); 
                    sum0 += _mm_dp_ps(kv, iv0, 241);    PRINT_VECTOR(sum0); 
                     
                    BLEND_ROTATE1_LEFT(iv0, iv1);

                    PRINT_LABEL("sum1"); 
                    sum0 += _mm_dp_ps(kv, iv0, 242);    PRINT_VECTOR(sum0);
                    
                    BLEND_ROTATE1_LEFT(iv0, iv1);

                    PRINT_LABEL("sum2"); 
                    sum0 += _mm_dp_ps(kv, iv0, 244);    PRINT_VECTOR(sum0);
                    
                    BLEND_ROTATE1_LEFT(iv0, iv1);

                    PRINT_LABEL("sum3"); 
                    sum0 += _mm_dp_ps(kv, iv0, 248);    PRINT_VECTOR(sum0);
                    
                }
            } //for (int r = 0...
            
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk)], sum0);         PRINT_VECTOR(sum0);
        } //for (int x = 0...
    } //for (int y = 0...
    processBoundaries2D (s, w, h, 
                       ks, kw, 
                       input, output, kernel);
    //printImage(w, h, s, output);
    
}


//no image vector reuse
void sseNoReuse2Convolve (const int s, const int w, const int h, 
                         const int ks, int kw, 
                         const float* input, float* output, const float* kernel) {


    int hk = kw / 2;                       
    int startX  = 0;
    int stopX   = w - 2 * (kw / 2);
    int startY  = 0;
    int stopY   = h - 2 * (kw / 2);
                         
    #pragma omp parallel for shared (input, output) 
    for (int y = startY; y < stopY; ++y) {
        for (int x = startX; x < stopX; x += 8) {
            register __m128 sum0, sum1;
            sum0 = sum1 = _mm_setzero_ps();
            for (int r = 0; r < kw; ++r) {
                const int idxFtmp = r * ks;
                const int idxIntmp = (y + r) * s + x;
                for (int c = 0; c < kw; c += 4) {
                    __m128 iv0, iv1, iv2;
                    register const __m128 kv = _mm_load_ps(&kernel[idxFtmp + c]);                     PRINT_VECTOR(kv);
                    iv0 = _mm_load_ps(&input[idxIntmp + c]);               PRINT_VECTOR(iv0);
                    iv1 = _mm_load_ps(&input[idxIntmp + c + 4]);           PRINT_VECTOR(iv1);
                    iv2 = _mm_load_ps(&input[idxIntmp + c + 8]);           PRINT_VECTOR(iv2);
                    
                    PRINT_LABEL("sum0"); 
                    sum0 += _mm_dp_ps(kv, iv0, 241);    PRINT_VECTOR(sum0); 
                    sum1 += _mm_dp_ps(kv, iv1, 241);    PRINT_VECTOR(sum1);
                     
                    BLEND_ROTATE2_LEFT(iv0, iv1, iv2);

                    PRINT_LABEL("sum1"); 
                    sum0 += _mm_dp_ps(kv, iv0, 242);    PRINT_VECTOR(sum0);
                    sum1 += _mm_dp_ps(kv, iv1, 242);    PRINT_VECTOR(sum1);
                    
                    BLEND_ROTATE2_LEFT(iv0, iv1, iv2);

                    PRINT_LABEL("sum2"); 
                    sum0 += _mm_dp_ps(kv, iv0, 244);    PRINT_VECTOR(sum0);
                    sum1 += _mm_dp_ps(kv, iv1, 244);    PRINT_VECTOR(sum1);
                    
                    BLEND_ROTATE2_LEFT(iv0, iv1, iv2);

                    PRINT_LABEL("sum3"); 
                    sum0 += _mm_dp_ps(kv, iv0, 248);    PRINT_VECTOR(sum0);
                    sum1 += _mm_dp_ps(kv, iv1, 248);    PRINT_VECTOR(sum1);
                    
                }
            } //for (int r = 0...
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk)], sum0);         PRINT_VECTOR(sum0);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 4], sum1);     PRINT_VECTOR(sum1);
        } //for (int x = 0...
    } //for (int y = 0...
    processBoundaries2D (s, w, h, 
                       ks, kw, 
                       input, output, kernel);
    //printImage(w, h, s, output);
    
}


//no image vector reuse
void sseNoReuse3Convolve (const int s, const int w, const int h, 
                         const int ks, int kw, 
                         const float* input, float* output, const float* kernel) {


    int hk = kw / 2;                       
    int startX  = 0;
    int stopX   = w - 2 * (kw / 2);
    int startY  = 0;
    int stopY   = h - 2 * (kw / 2);
                         
    #pragma omp parallel for shared (input, output) 
    for (int y = startY; y < stopY; ++y) {
        for (int x = startX; x < stopX; x += 12) {
            register __m128 sum0, sum1, sum2;
            sum0 = sum1 = sum2 = _mm_setzero_ps();
            for (int r = 0; r < kw; ++r) {
                const int idxFtmp = r * ks;
                const int idxIntmp = (y + r) * s + x;  
                for (int c = 0; c < kw; c += 4) {
                    __m128 iv0, iv1, iv2, iv3;
                    register const __m128 kv = _mm_load_ps(&kernel[idxFtmp + c]);                     PRINT_VECTOR(kv);
                    iv0 = _mm_load_ps(&input[idxIntmp + c]);               PRINT_VECTOR(iv0);
                    iv1 = _mm_load_ps(&input[idxIntmp + c + 4]);           PRINT_VECTOR(iv1);
                    iv2 = _mm_load_ps(&input[idxIntmp + c + 8]);           PRINT_VECTOR(iv2);
                    iv3 = _mm_load_ps(&input[idxIntmp + c + 12]);          PRINT_VECTOR(iv3);
                    
                    PRINT_LABEL("sum0"); 
                    sum0 += _mm_dp_ps(kv, iv0, 241);    PRINT_VECTOR(sum0); 
                    sum1 += _mm_dp_ps(kv, iv1, 241);    PRINT_VECTOR(sum1);
                    sum2 += _mm_dp_ps(kv, iv2, 241);    PRINT_VECTOR(sum2);
                     
                    BLEND_ROTATE3_LEFT(iv0, iv1, iv2, iv3);

                    PRINT_LABEL("sum1"); 
                    sum0 += _mm_dp_ps(kv, iv0, 242);    PRINT_VECTOR(sum0);
                    sum1 += _mm_dp_ps(kv, iv1, 242);    PRINT_VECTOR(sum1);
                    sum2 += _mm_dp_ps(kv, iv2, 242);    PRINT_VECTOR(sum2);
                    
                    BLEND_ROTATE3_LEFT(iv0, iv1, iv2, iv3);

                    PRINT_LABEL("sum2"); 
                    sum0 += _mm_dp_ps(kv, iv0, 244);    PRINT_VECTOR(sum0);
                    sum1 += _mm_dp_ps(kv, iv1, 244);    PRINT_VECTOR(sum1);
                    sum2 += _mm_dp_ps(kv, iv2, 244);    PRINT_VECTOR(sum2);
                    
                    BLEND_ROTATE3_LEFT(iv0, iv1, iv2, iv3);

                    PRINT_LABEL("sum3"); 
                    sum0 += _mm_dp_ps(kv, iv0, 248);    PRINT_VECTOR(sum0);
                    sum1 += _mm_dp_ps(kv, iv1, 248);    PRINT_VECTOR(sum1);
                    sum2 += _mm_dp_ps(kv, iv2, 248);    PRINT_VECTOR(sum2);
                    
                }
            } //for (int r = 0...
            
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk)], sum0);         PRINT_VECTOR(sum0);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 4], sum1);     PRINT_VECTOR(sum1);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 8], sum2);     PRINT_VECTOR(sum2);
        } //for (int x = 0...
    } //for (int y = 0...
    processBoundaries2D (s, w, h, 
                       ks, kw, 
                       input, output, kernel);
    //printImage(w, h, s, output);
    
}

//no image vector reuse
void sseNoReuse4Convolve (const int s, const int w, const int h, 
                          const int ks, int kw, 
                          const float* __restrict input, float* __restrict output, 
                          const float* __restrict kernel) {


    int hk = kw / 2;
    int startX  = 0;
    int stopX   = w - hk * 2;
    int startY  = 0;
    int stopY   = h - hk * 2;
                         
    #pragma omp parallel for shared (input, output) 
    for (int y = startY; y < stopY; ++y) {
        for (int x = startX; x < stopX; x += 16) { 
            register __m128 sum0, sum1, sum2, sum3; 
            sum0 = sum1 = sum2 = sum3 = _mm_setzero_ps();
            for (int r = 0; r < kw; ++r) {
                const int idxFtmp = r * ks;
                const int idxIntmp = (y + r) * s + x; 
                
                for (int c = 0; c < kw; c += 4) {
                    __m128 iv0, iv1, iv2, iv3, iv4;
                    register const __m128 kv = _mm_load_ps(&kernel[idxFtmp + c]);                     PRINT_VECTOR(kv);
                    iv0 = _mm_load_ps(&input[idxIntmp + c]);               PRINT_VECTOR(iv0);
                    iv1 = _mm_load_ps(&input[idxIntmp + c + 4]);           PRINT_VECTOR(iv1);
                    iv2 = _mm_load_ps(&input[idxIntmp + c + 8]);           PRINT_VECTOR(iv2);
                    iv3 = _mm_load_ps(&input[idxIntmp + c + 12]);          PRINT_VECTOR(iv3);
                    iv4 = _mm_load_ps(&input[idxIntmp + c + 16]);          PRINT_VECTOR(iv4);
                    
                    PRINT_LABEL("sum0"); 
                    sum0 += _mm_dp_ps(kv, iv0, 241);    PRINT_VECTOR(sum0); 
                    sum1 += _mm_dp_ps(kv, iv1, 241);    PRINT_VECTOR(sum1);
                    sum2 += _mm_dp_ps(kv, iv2, 241);    PRINT_VECTOR(sum2);
                    sum3 += _mm_dp_ps(kv, iv3, 241);    PRINT_VECTOR(sum3);
                     
                    BLEND_ROTATE4_LEFT(iv0, iv1, iv2, iv3, iv4);

                    PRINT_LABEL("sum1"); 
                    sum0 += _mm_dp_ps(kv, iv0, 242);    PRINT_VECTOR(sum0);
                    sum1 += _mm_dp_ps(kv, iv1, 242);    PRINT_VECTOR(sum1);
                    sum2 += _mm_dp_ps(kv, iv2, 242);    PRINT_VECTOR(sum2);
                    sum3 += _mm_dp_ps(kv, iv3, 242);    PRINT_VECTOR(sum3);
                    
                    BLEND_ROTATE4_LEFT(iv0, iv1, iv2, iv3, iv4);

                    PRINT_LABEL("sum2"); 
                    sum0 += _mm_dp_ps(kv, iv0, 244);    PRINT_VECTOR(sum0);
                    sum1 += _mm_dp_ps(kv, iv1, 244);    PRINT_VECTOR(sum1);
                    sum2 += _mm_dp_ps(kv, iv2, 244);    PRINT_VECTOR(sum2);
                    sum3 += _mm_dp_ps(kv, iv3, 244);    PRINT_VECTOR(sum3);
                    
                    BLEND_ROTATE4_LEFT(iv0, iv1, iv2, iv3, iv4);

                    PRINT_LABEL("sum3"); 
                    sum0 += _mm_dp_ps(kv, iv0, 248);    PRINT_VECTOR(sum0);
                    sum1 += _mm_dp_ps(kv, iv1, 248);    PRINT_VECTOR(sum1);
                    sum2 += _mm_dp_ps(kv, iv2, 248);    PRINT_VECTOR(sum2);
                    sum3 += _mm_dp_ps(kv, iv3, 248);    PRINT_VECTOR(sum3);
                    
                }
            } //for (int r = 0...
            
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk)], sum0);         PRINT_VECTOR(sum0);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 4], sum1);     PRINT_VECTOR(sum1);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 8], sum2);     PRINT_VECTOR(sum2);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 12], sum3);    PRINT_VECTOR(sum3);
        } //for (int x = 0...
    } //for (int y = 0...
    
    processBoundaries2D (s, w, h, 
                       ks, kw, 
                       input, output, kernel);
    
//    printImage(w, h, s, output);
    
}


//no image vector reuse
void sseNoReuse5Convolve (const int s, const int w, const int h, 
                         const int ks, int kw, 
                         const float* input, float* output, const float* kernel) {

    int hk = kw / 2;                       
    int startX  = 0;
    int stopX   = w - 2 * (kw / 2);
    int startY  = 0;
    int stopY   = h - 2 * (kw / 2);
                         
    #pragma omp parallel for shared (input, output) 
    for (int y = startY; y < stopY; ++y) {
        for (int x = startX; x < stopX; x += 20) {
            register __m128 sum0, sum1, sum2, sum3, sum4;
            sum0 = sum1 = sum2 = sum3 = sum4 = _mm_setzero_ps();
            for (int r = 0; r < kw; ++r) {
                const int idxFtmp = r * ks;
                const int idxIntmp = (y + r) * s + x;
                for (int c = 0; c < kw; c += 4) {
                    __m128 iv0, iv1, iv2, iv3, iv4, iv5;
                    register const __m128 kv = _mm_load_ps(&kernel[idxFtmp + c]);                     PRINT_VECTOR(kv);
                    iv0 = _mm_load_ps(&input[idxIntmp + c]);               PRINT_VECTOR(iv0);
                    iv1 = _mm_load_ps(&input[idxIntmp + c + 4]);           PRINT_VECTOR(iv1);
                    iv2 = _mm_load_ps(&input[idxIntmp + c + 8]);           PRINT_VECTOR(iv2);
                    iv3 = _mm_load_ps(&input[idxIntmp + c + 12]);          PRINT_VECTOR(iv3);
                    iv4 = _mm_load_ps(&input[idxIntmp + c + 16]);          PRINT_VECTOR(iv4);
                    iv5 = _mm_load_ps(&input[idxIntmp + c + 20]);          PRINT_VECTOR(iv5);
                    
                    PRINT_LABEL("sum0"); 
                    sum0 += _mm_dp_ps(kv, iv0, 241);    PRINT_VECTOR(sum0); 
                    sum1 += _mm_dp_ps(kv, iv1, 241);    PRINT_VECTOR(sum1);
                    sum2 += _mm_dp_ps(kv, iv2, 241);    PRINT_VECTOR(sum2);
                    sum3 += _mm_dp_ps(kv, iv3, 241);    PRINT_VECTOR(sum3);
                    sum4 += _mm_dp_ps(kv, iv4, 241);    PRINT_VECTOR(sum4);
                     
                    BLEND_ROTATE5_LEFT(iv0, iv1, iv2, iv3, iv4, iv5);

                    PRINT_LABEL("sum1"); 
                    sum0 += _mm_dp_ps(kv, iv0, 242);    PRINT_VECTOR(sum0);
                    sum1 += _mm_dp_ps(kv, iv1, 242);    PRINT_VECTOR(sum1);
                    sum2 += _mm_dp_ps(kv, iv2, 242);    PRINT_VECTOR(sum2);
                    sum3 += _mm_dp_ps(kv, iv3, 242);    PRINT_VECTOR(sum3);
                    sum4 += _mm_dp_ps(kv, iv4, 242);    PRINT_VECTOR(sum4);
                    
                    BLEND_ROTATE5_LEFT(iv0, iv1, iv2, iv3, iv4, iv5);

                    PRINT_LABEL("sum2"); 
                    sum0 += _mm_dp_ps(kv, iv0, 244);    PRINT_VECTOR(sum0);
                    sum1 += _mm_dp_ps(kv, iv1, 244);    PRINT_VECTOR(sum1);
                    sum2 += _mm_dp_ps(kv, iv2, 244);    PRINT_VECTOR(sum2);
                    sum3 += _mm_dp_ps(kv, iv3, 244);    PRINT_VECTOR(sum3);
                    sum4 += _mm_dp_ps(kv, iv4, 244);    PRINT_VECTOR(sum4);
                    
                    BLEND_ROTATE5_LEFT(iv0, iv1, iv2, iv3, iv4, iv5);

                    PRINT_LABEL("sum3"); 
                    sum0 += _mm_dp_ps(kv, iv0, 248);    PRINT_VECTOR(sum0);
                    sum1 += _mm_dp_ps(kv, iv1, 248);    PRINT_VECTOR(sum1);
                    sum2 += _mm_dp_ps(kv, iv2, 248);    PRINT_VECTOR(sum2);
                    sum3 += _mm_dp_ps(kv, iv3, 248);    PRINT_VECTOR(sum3);
                    sum4 += _mm_dp_ps(kv, iv4, 248);    PRINT_VECTOR(sum4);
                    
                }
            } //for (int r = 0...
            
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk)], sum0);         PRINT_VECTOR(sum0);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 4], sum1);     PRINT_VECTOR(sum1);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 8], sum2);     PRINT_VECTOR(sum2);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 12], sum3);    PRINT_VECTOR(sum3);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 16], sum4);    PRINT_VECTOR(sum4);
        } //for (int x = 0...
    } //for (int y = 0...
    processBoundaries2D (s, w, h, 
                       ks, kw, 
                       input, output, kernel);    
    //printImage(w, h, s, output);
}



//no image vector reuse
void sseNoReuse6Convolve (const int s, const int w, const int h, 
                         const int ks, int kw, 
                         const float* input, float* output, const float* kernel) {

    int hk = kw / 2;                       
    int startX  = 0;
    int stopX   = w - 2 * (kw / 2);
    int startY  = 0;
    int stopY   = h - 2 * (kw / 2);
                         
    #pragma omp parallel for shared (input, output) 
    for (int y = startY; y < stopY; ++y) {
        for (int x = startX; x < stopX; x += 24) {
            register __m128 sum0, sum1, sum2, sum3, sum4, sum5;
            sum0 = sum1 = sum2 = sum3 = sum4 = sum5 = _mm_setzero_ps();
            for (int r = 0; r < kw; ++r) {
                const int idxFtmp = r * ks;
                const int idxIntmp = (y + r) * s + x;
                for (int c = 0; c < kw; c += 4) {
                    __m128 iv0, iv1, iv2, iv3, iv4, iv5, iv6;
                    register const __m128 kv = _mm_load_ps(&kernel[idxFtmp + c]);                     PRINT_VECTOR(kv);
                    iv0 = _mm_load_ps(&input[idxIntmp + c]);               PRINT_VECTOR(iv0);
                    iv1 = _mm_load_ps(&input[idxIntmp + c + 4]);           PRINT_VECTOR(iv1);
                    iv2 = _mm_load_ps(&input[idxIntmp + c + 8]);           PRINT_VECTOR(iv2);
                    iv3 = _mm_load_ps(&input[idxIntmp + c + 12]);          PRINT_VECTOR(iv3);
                    iv4 = _mm_load_ps(&input[idxIntmp + c + 16]);          PRINT_VECTOR(iv4);
                    iv5 = _mm_load_ps(&input[idxIntmp + c + 20]);          PRINT_VECTOR(iv5);
                    iv6 = _mm_load_ps(&input[idxIntmp + c + 24]);          PRINT_VECTOR(iv6);
                    
                    PRINT_LABEL("sum0"); 
                    sum0 += _mm_dp_ps(kv, iv0, 241);    PRINT_VECTOR(sum0); 
                    sum1 += _mm_dp_ps(kv, iv1, 241);    PRINT_VECTOR(sum1);
                    sum2 += _mm_dp_ps(kv, iv2, 241);    PRINT_VECTOR(sum2);
                    sum3 += _mm_dp_ps(kv, iv3, 241);    PRINT_VECTOR(sum3);
                    sum4 += _mm_dp_ps(kv, iv4, 241);    PRINT_VECTOR(sum4);
                    sum5 += _mm_dp_ps(kv, iv5, 241);    PRINT_VECTOR(sum5);
                     
                    BLEND_ROTATE6_LEFT(iv0, iv1, iv2, iv3, iv4, iv5, iv6);

                    PRINT_LABEL("sum1"); 
                    sum0 += _mm_dp_ps(kv, iv0, 242);    PRINT_VECTOR(sum0);
                    sum1 += _mm_dp_ps(kv, iv1, 242);    PRINT_VECTOR(sum1);
                    sum2 += _mm_dp_ps(kv, iv2, 242);    PRINT_VECTOR(sum2);
                    sum3 += _mm_dp_ps(kv, iv3, 242);    PRINT_VECTOR(sum3);
                    sum4 += _mm_dp_ps(kv, iv4, 242);    PRINT_VECTOR(sum4);
                    sum5 += _mm_dp_ps(kv, iv5, 242);    PRINT_VECTOR(sum5);
                    
                    BLEND_ROTATE6_LEFT(iv0, iv1, iv2, iv3, iv4, iv5, iv6);

                    PRINT_LABEL("sum2"); 
                    sum0 += _mm_dp_ps(kv, iv0, 244);    PRINT_VECTOR(sum0);
                    sum1 += _mm_dp_ps(kv, iv1, 244);    PRINT_VECTOR(sum1);
                    sum2 += _mm_dp_ps(kv, iv2, 244);    PRINT_VECTOR(sum2);
                    sum3 += _mm_dp_ps(kv, iv3, 244);    PRINT_VECTOR(sum3);
                    sum4 += _mm_dp_ps(kv, iv4, 244);    PRINT_VECTOR(sum4);
                    sum5 += _mm_dp_ps(kv, iv5, 244);    PRINT_VECTOR(sum5);
                    
                    BLEND_ROTATE6_LEFT(iv0, iv1, iv2, iv3, iv4, iv5, iv6);

                    PRINT_LABEL("sum3"); 
                    sum0 += _mm_dp_ps(kv, iv0, 248);    PRINT_VECTOR(sum0);
                    sum1 += _mm_dp_ps(kv, iv1, 248);    PRINT_VECTOR(sum1);
                    sum2 += _mm_dp_ps(kv, iv2, 248);    PRINT_VECTOR(sum2);
                    sum3 += _mm_dp_ps(kv, iv3, 248);    PRINT_VECTOR(sum3);
                    sum4 += _mm_dp_ps(kv, iv4, 248);    PRINT_VECTOR(sum4);
                    sum5 += _mm_dp_ps(kv, iv5, 248);    PRINT_VECTOR(sum5);
                    
                }
            } //for (int r = 0...
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk)], sum0);         PRINT_VECTOR(sum0);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 4], sum1);     PRINT_VECTOR(sum1);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 8], sum2);     PRINT_VECTOR(sum2);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 12], sum3);    PRINT_VECTOR(sum3);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 16], sum4);    PRINT_VECTOR(sum4);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 20], sum5);    PRINT_VECTOR(sum5);
        } //for (int x = 0...
    } //for (int y = 0...
    processBoundaries2D (s, w, h, 
                       ks, kw, 
                       input, output, kernel);    
    //printImage(w, h, s, output);
}

//no image vector reuse
void sseNoReuse7Convolve (const int s, const int w, const int h, 
                         const int ks, int kw, 
                         const float* input, float* output, const float* kernel) {

    int hk = kw / 2;                       
    int startX  = 0;
    int stopX   = w - 2 * (kw / 2);
    int startY  = 0;
    int stopY   = h - 2 * (kw / 2);
                         
    #pragma omp parallel for shared (input, output) 
    for (int y = startY; y < stopY; ++y) {
        for (int x = startX; x < stopX; x += 28) {
            register __m128 sum0, sum1, sum2, sum3, sum4, sum5, sum6;
            sum0 = sum1 = sum2 = sum3 = sum4 = sum5 = sum6 = _mm_setzero_ps();
            for (int r = 0; r < kw; ++r) {
                const int idxFtmp = r * ks;
                const int idxIntmp = (y + r) * s + x;
                for (int c = 0; c < kw; c += 4) {
                    __m128 iv0, iv1, iv2, iv3, iv4, iv5, iv6, iv7;
                    register const __m128 kv = _mm_load_ps(&kernel[idxFtmp + c]);                     PRINT_VECTOR(kv);
                    iv0 = _mm_load_ps(&input[idxIntmp + c]);               PRINT_VECTOR(iv0);
                    iv1 = _mm_load_ps(&input[idxIntmp + c + 4]);           PRINT_VECTOR(iv1);
                    iv2 = _mm_load_ps(&input[idxIntmp + c + 8]);           PRINT_VECTOR(iv2);
                    iv3 = _mm_load_ps(&input[idxIntmp + c + 12]);          PRINT_VECTOR(iv3);
                    iv4 = _mm_load_ps(&input[idxIntmp + c + 16]);          PRINT_VECTOR(iv4);
                    iv5 = _mm_load_ps(&input[idxIntmp + c + 20]);          PRINT_VECTOR(iv5);
                    iv6 = _mm_load_ps(&input[idxIntmp + c + 24]);          PRINT_VECTOR(iv6);
                    iv7 = _mm_load_ps(&input[idxIntmp + c + 28]);          PRINT_VECTOR(iv7);
                    
                    PRINT_LABEL("sum0"); 
                    sum0 += _mm_dp_ps(kv, iv0, 241);    PRINT_VECTOR(sum0); 
                    sum1 += _mm_dp_ps(kv, iv1, 241);    PRINT_VECTOR(sum1);
                    sum2 += _mm_dp_ps(kv, iv2, 241);    PRINT_VECTOR(sum2);
                    sum3 += _mm_dp_ps(kv, iv3, 241);    PRINT_VECTOR(sum3);
                    sum4 += _mm_dp_ps(kv, iv4, 241);    PRINT_VECTOR(sum4);
                    sum5 += _mm_dp_ps(kv, iv5, 241);    PRINT_VECTOR(sum5);
                    sum6 += _mm_dp_ps(kv, iv6, 241);    PRINT_VECTOR(sum6);
                     
                    BLEND_ROTATE7_LEFT(iv0, iv1, iv2, iv3, iv4, iv5, iv6, iv7);

                    PRINT_LABEL("sum1"); 
                    sum0 += _mm_dp_ps(kv, iv0, 242);    PRINT_VECTOR(sum0);
                    sum1 += _mm_dp_ps(kv, iv1, 242);    PRINT_VECTOR(sum1);
                    sum2 += _mm_dp_ps(kv, iv2, 242);    PRINT_VECTOR(sum2);
                    sum3 += _mm_dp_ps(kv, iv3, 242);    PRINT_VECTOR(sum3);
                    sum4 += _mm_dp_ps(kv, iv4, 242);    PRINT_VECTOR(sum4);
                    sum5 += _mm_dp_ps(kv, iv5, 242);    PRINT_VECTOR(sum5);
                    sum6 += _mm_dp_ps(kv, iv6, 242);    PRINT_VECTOR(sum6);
                    
                    BLEND_ROTATE7_LEFT(iv0, iv1, iv2, iv3, iv4, iv5, iv6, iv7);

                    PRINT_LABEL("sum2"); 
                    sum0 += _mm_dp_ps(kv, iv0, 244);    PRINT_VECTOR(sum0);
                    sum1 += _mm_dp_ps(kv, iv1, 244);    PRINT_VECTOR(sum1);
                    sum2 += _mm_dp_ps(kv, iv2, 244);    PRINT_VECTOR(sum2);
                    sum3 += _mm_dp_ps(kv, iv3, 244);    PRINT_VECTOR(sum3);
                    sum4 += _mm_dp_ps(kv, iv4, 244);    PRINT_VECTOR(sum4);
                    sum5 += _mm_dp_ps(kv, iv5, 244);    PRINT_VECTOR(sum5);
                    sum6 += _mm_dp_ps(kv, iv6, 244);    PRINT_VECTOR(sum6);
                    
                    BLEND_ROTATE7_LEFT(iv0, iv1, iv2, iv3, iv4, iv5, iv6, iv7);

                    PRINT_LABEL("sum3"); 
                    sum0 += _mm_dp_ps(kv, iv0, 248);    PRINT_VECTOR(sum0);
                    sum1 += _mm_dp_ps(kv, iv1, 248);    PRINT_VECTOR(sum1);
                    sum2 += _mm_dp_ps(kv, iv2, 248);    PRINT_VECTOR(sum2);
                    sum3 += _mm_dp_ps(kv, iv3, 248);    PRINT_VECTOR(sum3);
                    sum4 += _mm_dp_ps(kv, iv4, 248);    PRINT_VECTOR(sum4);
                    sum5 += _mm_dp_ps(kv, iv5, 248);    PRINT_VECTOR(sum5);
                    sum6 += _mm_dp_ps(kv, iv6, 248);    PRINT_VECTOR(sum6);
                    
                }
            } //for (int r = 0...
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk)], sum0);         PRINT_VECTOR(sum0);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 4], sum1);     PRINT_VECTOR(sum1);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 8], sum2);     PRINT_VECTOR(sum2);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 12], sum3);    PRINT_VECTOR(sum3);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 16], sum4);    PRINT_VECTOR(sum4);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 20], sum5);    PRINT_VECTOR(sum5);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 24], sum6);    PRINT_VECTOR(sum6);
        } //for (int x = 0...
    } //for (int y = 0...
    processBoundaries2D (s, w, h, 
                       ks, kw, 
                       input, output, kernel);    
    //printImage(w, h, s, output);
}

//with image vector reuse
void sseReuse1Convolve (const int s, const int w, const int h, 
                           const int ks, int kw, 
                           const float* input, float* output, const float* kernel) {

    int hk = kw / 2;                       
    int startX  = 0;
    int stopX   = w - 2 * (kw / 2);
    int startY  = 0;
    int stopY   = h - 2 * (kw / 2);
                         
    #pragma omp parallel for shared (input, output) 
    for (int y = startY; y < stopY; ++y) {
        for (int x = startX; x < stopX; x += 4) {
            register __m128 sum0;
            sum0 = _mm_setzero_ps();
            for (int r = 0; r < kw; ++r) {
                const int idxFtmp = r * ks;
                const int idxIntmp = (y + r) * s + x;
                __m128 iv0, iv1;
                iv0 = _mm_load_ps(&input[idxIntmp]);               PRINT_VECTOR(iv0);
                
                for (int c = 0; c < kw; c += 4) {
                    PRINT_LABEL("carregando"); 
                    register const __m128 kv = _mm_load_ps(&kernel[idxFtmp + c]);   PRINT_VECTOR(kv);
                    iv1 = _mm_load_ps(&input[idxIntmp + c + 4]);              PRINT_VECTOR(iv1);
                    PRINT_LABEL("sum0"); 
                    sum0 += _mm_dp_ps(kv, iv0, 241);    PRINT_VECTOR(sum0); 
                     
                    PRINT_LABEL("sum1"); 
                    BLEND_ROTATE1_LEFT(iv0, iv1);
                    sum0 += _mm_dp_ps(kv, iv0, 242);    PRINT_VECTOR(sum0);
                    
                    PRINT_LABEL("sum2"); 
                    BLEND_ROTATE1_LEFT(iv0, iv1);
                    sum0 += _mm_dp_ps(kv, iv0, 244);    PRINT_VECTOR(sum0);
                    
                    PRINT_LABEL("sum3"); 
                    BLEND_ROTATE1_LEFT(iv0, iv1);
                    sum0 += _mm_dp_ps(kv, iv0, 248);    PRINT_VECTOR(sum0);
                    
                    PRINT_LABEL("trocando"); 
                    BLEND_ROTATE1_LEFT(iv0, iv1);
                    
                }
            } //for (int r = 0...
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk)], sum0);         PRINT_VECTOR(sum0);
        } //for (int x = 0...
    } //for (int y = 0...
    processBoundaries2D (s, w, h, 
                       ks, kw, 
                       input, output, kernel);    
    //printImage(w, h, s, output);
}

//with image vector reuse
void sseReuse2Convolve (const int s, const int w, const int h, 
                           const int ks, int kw, 
                           const float* input, float* output, const float* kernel) {
    int hk = kw / 2;                       
    int startX  = 0;
    int stopX   = w - 2 * hk;
    int startY  = 0;
    int stopY   = h - 2 * hk;
                         
    #pragma omp parallel for shared (input, output) 
    for (int y = startY; y < stopY; ++y) {
        for (int x = startX; x < stopX; x += 8) {
            register __m128 sum0, sum1;
            sum0 = sum1 = _mm_setzero_ps();
            for (int r = 0; r < kw; ++r) {
                const int idxFtmp = r * ks;
                const int idxIntmp = (y + r) * s + x;
                __m128 iv0, iv1, iv2;
                iv0 = _mm_load_ps(&input[idxIntmp]);               PRINT_VECTOR(iv0);
                iv1 = _mm_load_ps(&input[idxIntmp + 4]);           PRINT_VECTOR(iv1);
                
                for (int c = 0; c < kw; c += 4) {
                    PRINT_LABEL("carregando"); 
                    register const __m128 kv = _mm_load_ps(&kernel[idxFtmp + c]);   PRINT_VECTOR(kv);
                    iv2 = _mm_load_ps(&input[idxIntmp + c + 8]);              PRINT_VECTOR(iv2);
                    PRINT_LABEL("sum0"); 
                    sum0 += _mm_dp_ps(kv, iv0, 241);    PRINT_VECTOR(sum0); 
                    sum1 += _mm_dp_ps(kv, iv1, 241);    PRINT_VECTOR(sum1);
                     
                    PRINT_LABEL("sum1"); 
                    BLEND_ROTATE2_LEFT(iv0, iv1, iv2);
                    sum0 += _mm_dp_ps(kv, iv0, 242);    PRINT_VECTOR(sum0);
                    sum1 += _mm_dp_ps(kv, iv1, 242);    PRINT_VECTOR(sum1);
                    
                    PRINT_LABEL("sum2"); 
                    BLEND_ROTATE2_LEFT(iv0, iv1, iv2);
                    sum0 += _mm_dp_ps(kv, iv0, 244);    PRINT_VECTOR(sum0);
                    sum1 += _mm_dp_ps(kv, iv1, 244);    PRINT_VECTOR(sum1);
                    
                    PRINT_LABEL("sum3"); 
                    BLEND_ROTATE2_LEFT(iv0, iv1, iv2);
                    sum0 += _mm_dp_ps(kv, iv0, 248);    PRINT_VECTOR(sum0);
                    sum1 += _mm_dp_ps(kv, iv1, 248);    PRINT_VECTOR(sum1);
                    
                    PRINT_LABEL("trocando"); 
                    BLEND_ROTATE2_LEFT(iv0, iv1, iv2);
                    
                }
            } //for (int r = 0...
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk)], sum0);         PRINT_VECTOR(sum0);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 4], sum1);     PRINT_VECTOR(sum1);
        } //for (int x = 0...
    } //for (int y = 0...
    processBoundaries2D (s, w, h, 
                       ks, kw, 
                       input, output, kernel);    
    
    //printImage(w, h, s, output);
}

//with image vector reuse
void sseReuse3Convolve (const int s, const int w, const int h, 
                           const int ks, int kw, 
                           const float* input, float* output, const float* kernel) {
    int hk = kw / 2;                       
    int startX  = 0;
    int stopX   = w - 2 * (kw / 2);
    int startY  = 0;
    int stopY   = h - 2 * (kw / 2);
                         
    #pragma omp parallel for shared (input, output) 
    for (int y = startY; y < stopY; ++y) {
        for (int x = startX; x < stopX; x += 12) {
            register __m128 sum0, sum1, sum2;
            sum0 = sum1 = sum2 = _mm_setzero_ps();
            for (int r = 0; r < kw; ++r) {
                const int idxFtmp = r * ks;
                const int idxIntmp = (y + r) * s + x;
                __m128 iv0, iv1, iv2, iv3;
                iv0 = _mm_load_ps(&input[idxIntmp]);               PRINT_VECTOR(iv0);
                iv1 = _mm_load_ps(&input[idxIntmp + 4]);           PRINT_VECTOR(iv1);
                iv2 = _mm_load_ps(&input[idxIntmp + 8]);           PRINT_VECTOR(iv2);
                
                for (int c = 0; c < kw; c += 4) {
                    PRINT_LABEL("carregando"); 
                    register const __m128 kv = _mm_load_ps(&kernel[idxFtmp + c]);   PRINT_VECTOR(kv);
                    iv3 = _mm_load_ps(&input[idxIntmp + c + 12]);              PRINT_VECTOR(iv3);
                    PRINT_LABEL("sum0"); 
                    sum0 += _mm_dp_ps(kv, iv0, 241);    PRINT_VECTOR(sum0); 
                    sum1 += _mm_dp_ps(kv, iv1, 241);    PRINT_VECTOR(sum1);
                    sum2 += _mm_dp_ps(kv, iv2, 241);    PRINT_VECTOR(sum2);
                     
                    PRINT_LABEL("sum1"); 
                    BLEND_ROTATE3_LEFT(iv0, iv1, iv2, iv3);
                    sum0 += _mm_dp_ps(kv, iv0, 242);    PRINT_VECTOR(sum0);
                    sum1 += _mm_dp_ps(kv, iv1, 242);    PRINT_VECTOR(sum1);
                    sum2 += _mm_dp_ps(kv, iv2, 242);    PRINT_VECTOR(sum2);
                    
                    PRINT_LABEL("sum2"); 
                    BLEND_ROTATE3_LEFT(iv0, iv1, iv2, iv3);
                    sum0 += _mm_dp_ps(kv, iv0, 244);    PRINT_VECTOR(sum0);
                    sum1 += _mm_dp_ps(kv, iv1, 244);    PRINT_VECTOR(sum1);
                    sum2 += _mm_dp_ps(kv, iv2, 244);    PRINT_VECTOR(sum2);
                    
                    PRINT_LABEL("sum3"); 
                    BLEND_ROTATE3_LEFT(iv0, iv1, iv2, iv3);
                    sum0 += _mm_dp_ps(kv, iv0, 248);    PRINT_VECTOR(sum0);
                    sum1 += _mm_dp_ps(kv, iv1, 248);    PRINT_VECTOR(sum1);
                    sum2 += _mm_dp_ps(kv, iv2, 248);    PRINT_VECTOR(sum2);
                    
                    PRINT_LABEL("trocando"); 
                    BLEND_ROTATE3_LEFT(iv0, iv1, iv2, iv3);
                }
            } //for (int r = 0...
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk)], sum0);         PRINT_VECTOR(sum0);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 4], sum1);     PRINT_VECTOR(sum1);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 8], sum2);     PRINT_VECTOR(sum2);
        } //for (int x = 0...
    } //for (int y = 0...
    processBoundaries2D (s, w, h, 
                       ks, kw, 
                       input, output, kernel);    
    //printImage(w, h, s, output);
}

//with image vector reuse
void sseReuse4Convolve (const int s, const int w, const int h, 
                           const int ks, int kw, 
                           const float* input, float* output, const float* kernel) {

    int hk = kw / 2;                       
    int startX  = 0;
    int stopX   = w - 2 * hk;
    int startY  = 0;
    int stopY   = h - 2 * hk;
                         
    #pragma omp parallel for shared (input, output) 
    for (int y = startY; y < stopY; ++y) {
        for (int x = startX; x < stopX; x += 16) {
            register __m128 sum0, sum1, sum2, sum3;
            sum0 = sum1 = sum2 = sum3 = _mm_setzero_ps();
            for (int r = 0; r < kw; ++r) {
                const int idxFtmp = r * ks;
                const int idxIntmp = (y + r) * s + x;
                __m128 iv0, iv1, iv2, iv3, iv4;
                iv0 = _mm_load_ps(&input[idxIntmp]);               PRINT_VECTOR(iv0);
                iv1 = _mm_load_ps(&input[idxIntmp + 4]);           PRINT_VECTOR(iv1);
                iv2 = _mm_load_ps(&input[idxIntmp + 8]);           PRINT_VECTOR(iv2);
                iv3 = _mm_load_ps(&input[idxIntmp + 12]);          PRINT_VECTOR(iv3);
                
                for (int c = 0; c < kw; c += 4) {
                    PRINT_LABEL("carregando"); 
                    register const __m128 kv = _mm_load_ps(&kernel[idxFtmp + c]);   PRINT_VECTOR(kv);
                    iv4 = _mm_load_ps(&input[idxIntmp + c + 16]);              PRINT_VECTOR(iv4);
                    PRINT_LABEL("sum0"); 
                    sum0 += _mm_dp_ps(kv, iv0, 241);    PRINT_VECTOR(sum0); 
                    sum1 += _mm_dp_ps(kv, iv1, 241);    PRINT_VECTOR(sum1);
                    sum2 += _mm_dp_ps(kv, iv2, 241);    PRINT_VECTOR(sum2);
                    sum3 += _mm_dp_ps(kv, iv3, 241);    PRINT_VECTOR(sum3);
                     
                    PRINT_LABEL("sum1"); 
                    BLEND_ROTATE4_LEFT(iv0, iv1, iv2, iv3, iv4);
                    sum0 += _mm_dp_ps(kv, iv0, 242);    PRINT_VECTOR(sum0);
                    sum1 += _mm_dp_ps(kv, iv1, 242);    PRINT_VECTOR(sum1);
                    sum2 += _mm_dp_ps(kv, iv2, 242);    PRINT_VECTOR(sum2);
                    sum3 += _mm_dp_ps(kv, iv3, 242);    PRINT_VECTOR(sum3);
                    
                    PRINT_LABEL("sum2"); 
                    BLEND_ROTATE4_LEFT(iv0, iv1, iv2, iv3, iv4);
                    sum0 += _mm_dp_ps(kv, iv0, 244);    PRINT_VECTOR(sum0);
                    sum1 += _mm_dp_ps(kv, iv1, 244);    PRINT_VECTOR(sum1);
                    sum2 += _mm_dp_ps(kv, iv2, 244);    PRINT_VECTOR(sum2);
                    sum3 += _mm_dp_ps(kv, iv3, 244);    PRINT_VECTOR(sum3);
                    
                    PRINT_LABEL("sum3"); 
                    BLEND_ROTATE4_LEFT(iv0, iv1, iv2, iv3, iv4);
                    sum0 += _mm_dp_ps(kv, iv0, 248);    PRINT_VECTOR(sum0);
                    sum1 += _mm_dp_ps(kv, iv1, 248);    PRINT_VECTOR(sum1);
                    sum2 += _mm_dp_ps(kv, iv2, 248);    PRINT_VECTOR(sum2);
                    sum3 += _mm_dp_ps(kv, iv3, 248);    PRINT_VECTOR(sum3);
                    
                    PRINT_LABEL("trocando"); 
                    BLEND_ROTATE4_LEFT(iv0, iv1, iv2, iv3, iv4);
                }
            } //for (int r = 0...
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk)], sum0);         PRINT_VECTOR(sum0);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 4], sum1);     PRINT_VECTOR(sum1);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 8], sum2);     PRINT_VECTOR(sum2);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 12], sum3);    PRINT_VECTOR(sum3);
        } //for (int x = 0...
    } //for (int y = 0...
    processBoundaries2D (s, w, h, 
                       ks, kw, 
                       input, output, kernel);    
    //printImage(w, h, s, output);
}


//with image vector reuse
void sseReuse5Convolve (const int s, const int w, const int h, 
                           const int ks, int kw, 
                           const float* input, float* output, const float* kernel) {
    int hk = kw / 2;                       
    int startX  = 0;
    int stopX   = w - 2 * (kw / 2);
    int startY  = 0;
    int stopY   = h - 2 * (kw / 2);
                         
    #pragma omp parallel for shared (input, output) 
    for (int y = startY; y < stopY; ++y) {
        for (int x = startX; x < stopX; x += 20) {
            register __m128 sum0, sum1, sum2, sum3, sum4;
            sum0 = sum1 = sum2 = sum3 = sum4 = _mm_setzero_ps();
            for (int r = 0; r < kw; ++r) {
                const int idxFtmp = r * ks;
                const int idxIntmp = (y + r) * s + x;
                __m128 iv0, iv1, iv2, iv3, iv4, iv5;
                iv0 = _mm_load_ps(&input[idxIntmp]);               PRINT_VECTOR(iv0);
                iv1 = _mm_load_ps(&input[idxIntmp + 4]);           PRINT_VECTOR(iv1);
                iv2 = _mm_load_ps(&input[idxIntmp + 8]);           PRINT_VECTOR(iv2);
                iv3 = _mm_load_ps(&input[idxIntmp + 12]);          PRINT_VECTOR(iv3);
                iv4 = _mm_load_ps(&input[idxIntmp + 16]);          PRINT_VECTOR(iv4);
                
                for (int c = 0; c < kw; c += 4) {
                    PRINT_LABEL("carregando"); 
                    register const __m128 kv = _mm_load_ps(&kernel[idxFtmp + c]);   PRINT_VECTOR(kv);
                    iv5 = _mm_load_ps(&input[idxIntmp + c + 20]);              PRINT_VECTOR(iv5);
                    PRINT_LABEL("sum0"); 
                    sum0 += _mm_dp_ps(kv, iv0, 241);    PRINT_VECTOR(sum0); 
                    sum1 += _mm_dp_ps(kv, iv1, 241);    PRINT_VECTOR(sum1);
                    sum2 += _mm_dp_ps(kv, iv2, 241);    PRINT_VECTOR(sum2);
                    sum3 += _mm_dp_ps(kv, iv3, 241);    PRINT_VECTOR(sum3);
                    sum4 += _mm_dp_ps(kv, iv4, 241);    PRINT_VECTOR(sum4);
                     
                    PRINT_LABEL("sum1"); 
                    BLEND_ROTATE5_LEFT(iv0, iv1, iv2, iv3, iv4, iv5);
                    sum0 += _mm_dp_ps(kv, iv0, 242);    PRINT_VECTOR(sum0);
                    sum1 += _mm_dp_ps(kv, iv1, 242);    PRINT_VECTOR(sum1);
                    sum2 += _mm_dp_ps(kv, iv2, 242);    PRINT_VECTOR(sum2);
                    sum3 += _mm_dp_ps(kv, iv3, 242);    PRINT_VECTOR(sum3);
                    sum4 += _mm_dp_ps(kv, iv4, 242);    PRINT_VECTOR(sum4);
                    
                    PRINT_LABEL("sum2"); 
                    BLEND_ROTATE5_LEFT(iv0, iv1, iv2, iv3, iv4, iv5);
                    sum0 += _mm_dp_ps(kv, iv0, 244);    PRINT_VECTOR(sum0);
                    sum1 += _mm_dp_ps(kv, iv1, 244);    PRINT_VECTOR(sum1);
                    sum2 += _mm_dp_ps(kv, iv2, 244);    PRINT_VECTOR(sum2);
                    sum3 += _mm_dp_ps(kv, iv3, 244);    PRINT_VECTOR(sum3);
                    sum4 += _mm_dp_ps(kv, iv4, 244);    PRINT_VECTOR(sum4);
                    
                    PRINT_LABEL("sum3"); 
                    BLEND_ROTATE5_LEFT(iv0, iv1, iv2, iv3, iv4, iv5);
                    sum0 += _mm_dp_ps(kv, iv0, 248);    PRINT_VECTOR(sum0);
                    sum1 += _mm_dp_ps(kv, iv1, 248);    PRINT_VECTOR(sum1);
                    sum2 += _mm_dp_ps(kv, iv2, 248);    PRINT_VECTOR(sum2);
                    sum3 += _mm_dp_ps(kv, iv3, 248);    PRINT_VECTOR(sum3);
                    sum4 += _mm_dp_ps(kv, iv4, 248);    PRINT_VECTOR(sum4);
                    
                    PRINT_LABEL("trocando"); 
                    BLEND_ROTATE5_LEFT(iv0, iv1, iv2, iv3, iv4, iv5);
                    
                }
            } //for (int r = 0...
            
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk)], sum0);         PRINT_VECTOR(sum0);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 4], sum1);     PRINT_VECTOR(sum1);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 8], sum2);     PRINT_VECTOR(sum2);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 12], sum3);    PRINT_VECTOR(sum3);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 16], sum4);    PRINT_VECTOR(sum4);
        } //for (int x = 0...
    } //for (int y = 0...
    processBoundaries2D (s, w, h, 
                       ks, kw, 
                       input, output, kernel);    
    
    //printImage(w, h, s, output);
}

//with image vector reuse
void sseReuse6Convolve (const int s, const int w, const int h, 
                           const int ks, int kw, 
                           const float* input, float* output, const float* kernel) {

    int hk = kw / 2;                       
    int startX  = 0;
    int stopX   = w - 2 * (kw / 2);
    int startY  = 0;
    int stopY   = h - 2 * (kw / 2);
                         
    #pragma omp parallel for shared (input, output) 
    for (int y = startY; y < stopY; ++y) {
        for (int x = startX; x < stopX; x += 24) {
            register __m128 sum0, sum1, sum2, sum3, sum4, sum5;
            sum0 = sum1 = sum2 = sum3 = sum4 = sum5  = _mm_setzero_ps();
            for (int r = 0; r < kw; ++r) {
                const int idxFtmp = r * ks;
                const int idxIntmp = (y + r) * s + x;
                __m128 iv0, iv1, iv2, iv3, iv4, iv5, iv6;
                iv0 = _mm_load_ps(&input[idxIntmp]);               PRINT_VECTOR(iv0);
                iv1 = _mm_load_ps(&input[idxIntmp + 4]);           PRINT_VECTOR(iv1);
                iv2 = _mm_load_ps(&input[idxIntmp + 8]);           PRINT_VECTOR(iv2);
                iv3 = _mm_load_ps(&input[idxIntmp + 12]);          PRINT_VECTOR(iv3);
                iv4 = _mm_load_ps(&input[idxIntmp + 16]);          PRINT_VECTOR(iv4);
                iv5 = _mm_load_ps(&input[idxIntmp + 20]);          PRINT_VECTOR(iv5);
                
                for (int c = 0; c < kw; c += 4) {
                    PRINT_LABEL("carregando"); 
                    register const __m128 kv = _mm_load_ps(&kernel[idxFtmp + c]);   PRINT_VECTOR(kv);
                    iv6 = _mm_load_ps(&input[idxIntmp + c + 24]);              PRINT_VECTOR(iv6);
                    PRINT_LABEL("sum0"); 
                    sum0 += _mm_dp_ps(kv, iv0, 241);    PRINT_VECTOR(sum0); 
                    sum1 += _mm_dp_ps(kv, iv1, 241);    PRINT_VECTOR(sum1);
                    sum2 += _mm_dp_ps(kv, iv2, 241);    PRINT_VECTOR(sum2);
                    sum3 += _mm_dp_ps(kv, iv3, 241);    PRINT_VECTOR(sum3);
                    sum4 += _mm_dp_ps(kv, iv4, 241);    PRINT_VECTOR(sum4);
                    sum5 += _mm_dp_ps(kv, iv5, 241);    PRINT_VECTOR(sum5);
                     
                    PRINT_LABEL("sum1"); 
                    BLEND_ROTATE6_LEFT(iv0, iv1, iv2, iv3, iv4, iv5, iv6);
                    sum0 += _mm_dp_ps(kv, iv0, 242);    PRINT_VECTOR(sum0);
                    sum1 += _mm_dp_ps(kv, iv1, 242);    PRINT_VECTOR(sum1);
                    sum2 += _mm_dp_ps(kv, iv2, 242);    PRINT_VECTOR(sum2);
                    sum3 += _mm_dp_ps(kv, iv3, 242);    PRINT_VECTOR(sum3);
                    sum4 += _mm_dp_ps(kv, iv4, 242);    PRINT_VECTOR(sum4);
                    sum5 += _mm_dp_ps(kv, iv5, 242);    PRINT_VECTOR(sum5);
                    
                    PRINT_LABEL("sum2"); 
                    BLEND_ROTATE6_LEFT(iv0, iv1, iv2, iv3, iv4, iv5, iv6);
                    sum0 += _mm_dp_ps(kv, iv0, 244);    PRINT_VECTOR(sum0);
                    sum1 += _mm_dp_ps(kv, iv1, 244);    PRINT_VECTOR(sum1);
                    sum2 += _mm_dp_ps(kv, iv2, 244);    PRINT_VECTOR(sum2);
                    sum3 += _mm_dp_ps(kv, iv3, 244);    PRINT_VECTOR(sum3);
                    sum4 += _mm_dp_ps(kv, iv4, 244);    PRINT_VECTOR(sum4);
                    sum5 += _mm_dp_ps(kv, iv5, 244);    PRINT_VECTOR(sum5);
                    
                    PRINT_LABEL("sum3"); 
                    BLEND_ROTATE6_LEFT(iv0, iv1, iv2, iv3, iv4, iv5, iv6);
                    sum0 += _mm_dp_ps(kv, iv0, 248);    PRINT_VECTOR(sum0);
                    sum1 += _mm_dp_ps(kv, iv1, 248);    PRINT_VECTOR(sum1);
                    sum2 += _mm_dp_ps(kv, iv2, 248);    PRINT_VECTOR(sum2);
                    sum3 += _mm_dp_ps(kv, iv3, 248);    PRINT_VECTOR(sum3);
                    sum4 += _mm_dp_ps(kv, iv4, 248);    PRINT_VECTOR(sum4);
                    sum5 += _mm_dp_ps(kv, iv5, 248);    PRINT_VECTOR(sum5);
                    
                    PRINT_LABEL("trocando"); 
                    BLEND_ROTATE6_LEFT(iv0, iv1, iv2, iv3, iv4, iv5, iv6);
                }
            } //for (int r = 0...
            
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk)], sum0);         PRINT_VECTOR(sum0);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 4], sum1);     PRINT_VECTOR(sum1);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 8], sum2);     PRINT_VECTOR(sum2);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 12], sum3);    PRINT_VECTOR(sum3);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 16], sum4);    PRINT_VECTOR(sum4);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 20], sum5);    PRINT_VECTOR(sum5);
        } //for (int x = 0...
    } //for (int y = 0...
    processBoundaries2D (s, w, h, 
                       ks, kw, 
                       input, output, kernel);    
    //printImage(w, h, s, output);
    
}

//with image vector reuse
void sseReuse7Convolve (const int s, const int w, const int h, 
                           const int ks, int kw, 
                           const float* input, float* output, const float* kernel) {

    int hk = kw / 2;                       
    int startX  = 0;
    int stopX   = w - 2 * (kw / 2);
    int startY  = 0;
    int stopY   = h - 2 * (kw / 2);
                         
    #pragma omp parallel for shared (input, output) 
    for (int y = startY; y < stopY; ++y) {
        for (int x = startX; x < stopX; x += 24) {
            register __m128 sum0, sum1, sum2, sum3, sum4, sum5, sum6;
            sum0 = sum1 = sum2 = sum3 = sum4 = sum5 = sum6 = _mm_setzero_ps();
            for (int r = 0; r < kw; ++r) {
                const int idxFtmp = r * ks;
                const int idxIntmp = (y + r) * s + x;
                __m128 iv0, iv1, iv2, iv3, iv4, iv5, iv6, iv7;
                iv0 = _mm_load_ps(&input[idxIntmp]);               PRINT_VECTOR(iv0);
                iv1 = _mm_load_ps(&input[idxIntmp + 4]);           PRINT_VECTOR(iv1);
                iv2 = _mm_load_ps(&input[idxIntmp + 8]);           PRINT_VECTOR(iv2);
                iv3 = _mm_load_ps(&input[idxIntmp + 12]);          PRINT_VECTOR(iv3);
                iv4 = _mm_load_ps(&input[idxIntmp + 16]);          PRINT_VECTOR(iv4);
                iv5 = _mm_load_ps(&input[idxIntmp + 20]);          PRINT_VECTOR(iv5);
                iv6 = _mm_load_ps(&input[idxIntmp + 24]);          PRINT_VECTOR(iv6);
                
                for (int c = 0; c < kw; c += 4) {
                    PRINT_LABEL("carregando"); 
                    register const __m128 kv = _mm_load_ps(&kernel[idxFtmp + c]);   PRINT_VECTOR(kv);
                    iv7 = _mm_load_ps(&input[idxIntmp + c + 28]);              PRINT_VECTOR(iv7);
                    PRINT_LABEL("sum0"); 
                    sum0 += _mm_dp_ps(kv, iv0, 241);    PRINT_VECTOR(sum0); 
                    sum1 += _mm_dp_ps(kv, iv1, 241);    PRINT_VECTOR(sum1);
                    sum2 += _mm_dp_ps(kv, iv2, 241);    PRINT_VECTOR(sum2);
                    sum3 += _mm_dp_ps(kv, iv3, 241);    PRINT_VECTOR(sum3);
                    sum4 += _mm_dp_ps(kv, iv4, 241);    PRINT_VECTOR(sum4);
                    sum5 += _mm_dp_ps(kv, iv5, 241);    PRINT_VECTOR(sum5);
                    sum6 += _mm_dp_ps(kv, iv6, 241);    PRINT_VECTOR(sum6);
                     
                    PRINT_LABEL("sum1"); 
                    BLEND_ROTATE7_LEFT(iv0, iv1, iv2, iv3, iv4, iv5, iv6, iv7);
                    sum0 += _mm_dp_ps(kv, iv0, 242);    PRINT_VECTOR(sum0);
                    sum1 += _mm_dp_ps(kv, iv1, 242);    PRINT_VECTOR(sum1);
                    sum2 += _mm_dp_ps(kv, iv2, 242);    PRINT_VECTOR(sum2);
                    sum3 += _mm_dp_ps(kv, iv3, 242);    PRINT_VECTOR(sum3);
                    sum4 += _mm_dp_ps(kv, iv4, 242);    PRINT_VECTOR(sum4);
                    sum5 += _mm_dp_ps(kv, iv5, 242);    PRINT_VECTOR(sum5);
                    sum6 += _mm_dp_ps(kv, iv6, 242);    PRINT_VECTOR(sum6);
                    
                    PRINT_LABEL("sum2"); 
                    BLEND_ROTATE7_LEFT(iv0, iv1, iv2, iv3, iv4, iv5, iv6, iv7);
                    sum0 += _mm_dp_ps(kv, iv0, 244);    PRINT_VECTOR(sum0);
                    sum1 += _mm_dp_ps(kv, iv1, 244);    PRINT_VECTOR(sum1);
                    sum2 += _mm_dp_ps(kv, iv2, 244);    PRINT_VECTOR(sum2);
                    sum3 += _mm_dp_ps(kv, iv3, 244);    PRINT_VECTOR(sum3);
                    sum4 += _mm_dp_ps(kv, iv4, 244);    PRINT_VECTOR(sum4);
                    sum5 += _mm_dp_ps(kv, iv5, 244);    PRINT_VECTOR(sum5);
                    sum6 += _mm_dp_ps(kv, iv6, 244);    PRINT_VECTOR(sum6);
                    
                    PRINT_LABEL("sum3"); 
                    BLEND_ROTATE7_LEFT(iv0, iv1, iv2, iv3, iv4, iv5, iv6, iv7);
                    sum0 += _mm_dp_ps(kv, iv0, 248);    PRINT_VECTOR(sum0);
                    sum1 += _mm_dp_ps(kv, iv1, 248);    PRINT_VECTOR(sum1);
                    sum2 += _mm_dp_ps(kv, iv2, 248);    PRINT_VECTOR(sum2);
                    sum3 += _mm_dp_ps(kv, iv3, 248);    PRINT_VECTOR(sum3);
                    sum4 += _mm_dp_ps(kv, iv4, 248);    PRINT_VECTOR(sum4);
                    sum5 += _mm_dp_ps(kv, iv5, 248);    PRINT_VECTOR(sum5);
                    sum6 += _mm_dp_ps(kv, iv6, 248);    PRINT_VECTOR(sum6);
                    
                    PRINT_LABEL("trocando"); 
                    BLEND_ROTATE7_LEFT(iv0, iv1, iv2, iv3, iv4, iv5, iv6, iv7);
                }
            } //for (int r = 0...
            
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk)], sum0);         PRINT_VECTOR(sum0);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 4], sum1);     PRINT_VECTOR(sum1);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 8], sum2);     PRINT_VECTOR(sum2);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 12], sum3);    PRINT_VECTOR(sum3);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 16], sum4);    PRINT_VECTOR(sum4);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 20], sum5);    PRINT_VECTOR(sum5);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 24], sum6);    PRINT_VECTOR(sum6);
        } //for (int x = 0...
    } //for (int y = 0...
    processBoundaries2D (s, w, h, 
                       ks, kw, 
                       input, output, kernel);    
    
}

void unalignedSSEConvolve (const int s, const int w, const int h, 
                           const int ks, int kw, 
                           const float* input, float* output,
                           const float* kernel) {
    
    int hk      = kw / 2;                       
    int startX  = 0;
    int stopX   = w - hk * 2;
    int startY  = 0;
    int stopY   = h - hk * 2;
    
    #pragma omp parallel for shared (input, output) 
    for (int y = startY; y < stopY; ++y) {
        register __m128 sum;
        register __m128 kv;
        for (int x = startX; x < stopX; x += 4) {
            sum = _mm_setzero_ps();
            for (int r = 0; r < kw; ++r) {
                int idxFtmp = r * ks;
                int idxIntmp = (y + r) * s + x;
                for (int c = 0; c < kw; c += 4) {
                    kv = _mm_load_ps(&kernel[idxFtmp + c]);
                    sum += _mm_dp_ps(kv, _mm_loadu_ps(&input[idxIntmp + c]), 241);
                    sum += _mm_dp_ps(kv, _mm_loadu_ps(&input[idxIntmp + c + 1]), 242);
                    sum += _mm_dp_ps(kv, _mm_loadu_ps(&input[idxIntmp + c + 2]), 244);
                    sum += _mm_dp_ps(kv, _mm_loadu_ps(&input[idxIntmp + c + 3]), 248);
                }
            } //for (int r = 0...
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk)], sum);
        } //for (int x = 0...
    } //for (int y = 0...
    processBoundaries2D (s, w, h, 
                         ks, kw, 
                         input, output, kernel);    
}
 
 

 
void loopBlockConvolve (const int s, const int w, const int h, 
                        const int ks, const int kw, 
                        const float* input, float* output, const float* kernel, 
                        const int xBlock, const int yBlock) {
    int hk = kw / 2;                       
    int startX  = 0;
    int stopX   = (w - 2 * (kw / 2));
    int startY  = 0;
    int stopY   = (h - 2 * (kw / 2));
    for (int y = startY; y < stopY; y += yBlock) {
        #pragma omp parallel for shared (input, output) 
        for (int x = startX; x < stopX; x += xBlock) {
            for (int yy = y; yy < min(y + yBlock, stopY); ++yy) {
                for (int xx = x; xx < min(x + xBlock, stopX); ++xx) {
                    float sum = 0;
                    for (int r = 0; r < kw; ++r) {
                        int idxFtmp = r * ks;
                        int idxIntmp = (yy + r) * s + xx;
                        for (int c = 0; c < kw; ++c) {
                            sum += kernel[idxFtmp + c] * input[idxIntmp + c];
                        }
                    } //for (int r = 0...
                    output[(yy + hk) * s + (xx + hk)] = sum;
                } //for (int x = 0...
            } //for (int y = 0...
        }
    }
    processBoundaries2D (s, w, h, 
                       ks, kw, 
                       input, output, kernel);    
    
}

void loopBlockLoopUnrollConvolve (const int s, const int w, const int h, 
                        const int ks, const int kw, 
                        const float* input, float* output, const float* kernel, 
                        const int xBlock, const int yBlock) {
    int hk = kw / 2;                       
    int startX  = 0;
    int stopX   = (w - 2 * (kw / 2));
    int startY  = 0;
    int stopY   = (h - 2 * (kw / 2));
    for (int y = startY; y < stopY; y += yBlock) {
        #pragma omp parallel for shared (input, output)  
        for (int x = startX; x < stopX; x += xBlock) {
            for (int yy = y; yy < min(y + yBlock, stopY); ++yy) {
                const int yXstride = (yy + hk) * s;
                for (int xx = x; xx < min(x + xBlock, stopX);) {
                    output[yXstride + xx + hk] = convolution(input, s, kernel, ks, kw, xx, yy); ++xx;
                    output[yXstride + xx + hk] = convolution(input, s, kernel, ks, kw, xx, yy); ++xx;
                    output[yXstride + xx + hk] = convolution(input, s, kernel, ks, kw, xx, yy); ++xx;
                    output[yXstride + xx + hk] = convolution(input, s, kernel, ks, kw, xx, yy); ++xx;
                    output[yXstride + xx + hk] = convolution(input, s, kernel, ks, kw, xx, yy); ++xx;
                    output[yXstride + xx + hk] = convolution(input, s, kernel, ks, kw, xx, yy); ++xx;
                    output[yXstride + xx + hk] = convolution(input, s, kernel, ks, kw, xx, yy); ++xx;
                    output[yXstride + xx + hk] = convolution(input, s, kernel, ks, kw, xx, yy); ++xx;
                    output[yXstride + xx + hk] = convolution(input, s, kernel, ks, kw, xx, yy); ++xx;
                    output[yXstride + xx + hk] = convolution(input, s, kernel, ks, kw, xx, yy); ++xx;
                    output[yXstride + xx + hk] = convolution(input, s, kernel, ks, kw, xx, yy); ++xx;
                    output[yXstride + xx + hk] = convolution(input, s, kernel, ks, kw, xx, yy); ++xx;
                    output[yXstride + xx + hk] = convolution(input, s, kernel, ks, kw, xx, yy); ++xx;
                    output[yXstride + xx + hk] = convolution(input, s, kernel, ks, kw, xx, yy); ++xx;
                    output[yXstride + xx + hk] = convolution(input, s, kernel, ks, kw, xx, yy); ++xx;
                    output[yXstride + xx + hk] = convolution(input, s, kernel, ks, kw, xx, yy); ++xx;
                } //for (int x = 0...
            } //for (int y = 0...
        }
    }
    processBoundaries2D (s, w, h, 
                       ks, kw, 
                       input, output, kernel);    
    
}

void loopBlockAlignedSSEConvolve (const int s, const int w, const int h, 
                        const int ks, const int kw, 
                        const float* input, float* output, const float* kernel, 
                        const int xBlock, const int yBlock) {
    int hk = kw / 2;                       
    int startX  = 0;
    int stopX   = (w - 2 * (kw / 2));
    int startY  = 0;
    int stopY   = (h - 2 * (kw / 2));
    for (int y = startY; y < stopY; y += yBlock) {
        //#pragma omp parallel for shared (input, output) 
        for (int x = startX; x < stopX; x += xBlock) {
            for (int yy = y; yy < min(y + yBlock, stopY); ++yy) {
                for (int xx = x; xx < min(x + xBlock, stopX); xx += 28) {
                    register __m128 sum0, sum1, sum2, sum3, sum4, sum5, sum6;
                    sum0 = sum1 = sum2 = sum3 = sum4 = sum5 = sum6 = _mm_setzero_ps();
                    for (int r = 0; r < kw; ++r) {
                        const int idxFtmp = r * ks;
                        const int idxIntmp = (yy + r) * s + xx;
                        for (int c = 0; c < kw; c += 4) {
                            __m128 iv0, iv1, iv2, iv3, iv4, iv5, iv6, iv7;
                            register const __m128 kv = _mm_load_ps(&kernel[idxFtmp + c]);   PRINT_VECTOR(kv);
                            iv0 = _mm_load_ps(&input[idxIntmp + c]);               PRINT_VECTOR(iv0);
                            iv1 = _mm_load_ps(&input[idxIntmp + c + 4]);           PRINT_VECTOR(iv1);
                            iv2 = _mm_load_ps(&input[idxIntmp + c + 8]);           PRINT_VECTOR(iv2);
                            iv3 = _mm_load_ps(&input[idxIntmp + c + 12]);          PRINT_VECTOR(iv3);
                            iv4 = _mm_load_ps(&input[idxIntmp + c + 16]);          PRINT_VECTOR(iv4);
                            iv5 = _mm_load_ps(&input[idxIntmp + c + 20]);          PRINT_VECTOR(iv5);
                            iv6 = _mm_load_ps(&input[idxIntmp + c + 24]);          PRINT_VECTOR(iv6);
                            iv7 = _mm_load_ps(&input[idxIntmp + c + 28]);          PRINT_VECTOR(iv7);
                            
                            PRINT_LABEL("sum0"); 
                            sum0 += _mm_dp_ps(kv, iv0, 241);    PRINT_VECTOR(sum0); 
                            sum1 += _mm_dp_ps(kv, iv1, 241);    PRINT_VECTOR(sum1);
                            sum2 += _mm_dp_ps(kv, iv2, 241);    PRINT_VECTOR(sum2);
                            sum3 += _mm_dp_ps(kv, iv3, 241);    PRINT_VECTOR(sum3);
                            sum4 += _mm_dp_ps(kv, iv4, 241);    PRINT_VECTOR(sum4);
                            sum5 += _mm_dp_ps(kv, iv5, 241);    PRINT_VECTOR(sum5);
                            sum6 += _mm_dp_ps(kv, iv6, 241);    PRINT_VECTOR(sum6);
                             
                            BLEND_ROTATE7_LEFT(iv0, iv1, iv2, iv3, iv4, iv5, iv6, iv7);
        
                            PRINT_LABEL("sum1"); 
                            sum0 += _mm_dp_ps(kv, iv0, 242);    PRINT_VECTOR(sum0);
                            sum1 += _mm_dp_ps(kv, iv1, 242);    PRINT_VECTOR(sum1);
                            sum2 += _mm_dp_ps(kv, iv2, 242);    PRINT_VECTOR(sum2);
                            sum3 += _mm_dp_ps(kv, iv3, 242);    PRINT_VECTOR(sum3);
                            sum4 += _mm_dp_ps(kv, iv4, 242);    PRINT_VECTOR(sum4);
                            sum5 += _mm_dp_ps(kv, iv5, 242);    PRINT_VECTOR(sum5);
                            sum6 += _mm_dp_ps(kv, iv6, 242);    PRINT_VECTOR(sum6);
                            
                            BLEND_ROTATE7_LEFT(iv0, iv1, iv2, iv3, iv4, iv5, iv6, iv7);
        
                            PRINT_LABEL("sum2"); 
                            sum0 += _mm_dp_ps(kv, iv0, 244);    PRINT_VECTOR(sum0);
                            sum1 += _mm_dp_ps(kv, iv1, 244);    PRINT_VECTOR(sum1);
                            sum2 += _mm_dp_ps(kv, iv2, 244);    PRINT_VECTOR(sum2);
                            sum3 += _mm_dp_ps(kv, iv3, 244);    PRINT_VECTOR(sum3);
                            sum4 += _mm_dp_ps(kv, iv4, 244);    PRINT_VECTOR(sum4);
                            sum5 += _mm_dp_ps(kv, iv5, 244);    PRINT_VECTOR(sum5);
                            sum6 += _mm_dp_ps(kv, iv6, 244);    PRINT_VECTOR(sum6);
                            
                            BLEND_ROTATE7_LEFT(iv0, iv1, iv2, iv3, iv4, iv5, iv6, iv7);
        
                            PRINT_LABEL("sum3"); 
                            sum0 += _mm_dp_ps(kv, iv0, 248);    PRINT_VECTOR(sum0);
                            sum1 += _mm_dp_ps(kv, iv1, 248);    PRINT_VECTOR(sum1);
                            sum2 += _mm_dp_ps(kv, iv2, 248);    PRINT_VECTOR(sum2);
                            sum3 += _mm_dp_ps(kv, iv3, 248);    PRINT_VECTOR(sum3);
                            sum4 += _mm_dp_ps(kv, iv4, 248);    PRINT_VECTOR(sum4);
                            sum5 += _mm_dp_ps(kv, iv5, 248);    PRINT_VECTOR(sum5);
                            sum6 += _mm_dp_ps(kv, iv6, 248);    PRINT_VECTOR(sum6);

                        }
                    } //for (int r = 0...
                    _mm_storeu_ps(&output[(yy + hk) * s + (xx + hk)], sum0);         PRINT_VECTOR(sum0);
                    _mm_storeu_ps(&output[(yy + hk) * s + (xx + hk) + 4], sum1);     PRINT_VECTOR(sum1);
                    _mm_storeu_ps(&output[(yy + hk) * s + (xx + hk) + 8], sum2);     PRINT_VECTOR(sum2);
                    _mm_storeu_ps(&output[(yy + hk) * s + (xx + hk) + 12], sum3);    PRINT_VECTOR(sum3);
                    _mm_storeu_ps(&output[(yy + hk) * s + (xx + hk) + 16], sum4);    PRINT_VECTOR(sum4);
                    _mm_storeu_ps(&output[(yy + hk) * s + (xx + hk) + 20], sum5);    PRINT_VECTOR(sum5);
                    _mm_storeu_ps(&output[(yy + hk) * s + (xx + hk) + 24], sum6);    PRINT_VECTOR(sum6);
                } //for (int x = 0...
            } //for (int y = 0...
        }
    }
    processBoundaries2D (s, w, h, 
                       ks, kw, 
                       input, output, kernel);    
    
}

void sse3LbConvolve (const int s, const int w, const int h, const int ks, 
                       const float* input, float* output, const float* kernel) {
    const int kw = 3;
    int hk = kw / 2;                       
    const int stopX = w - 2 * (kw / 2);
    const int stopY = h - 2 * (kw / 2);
    
    const int is1 = s;
    const int is2 = s * 2;
    const int is3 = s * 3;
    const int yJump = is3;
    
    const float* stopYY = input + (s * (stopY - 3));
    
    const int yJump0 = s * 3;
    const int yJump1 = s * 4;
    const int yJump2 = s * 5;
    
    #pragma omp parallel for shared (input, output) schedule(guided)
    for (const float* ip = input; ip < stopYY; ip += yJump) {
        const register __m128 kv0 = _mm_load_ps(&kernel[0]); PRINT_VECTOR(kv0); //{1,2,3,0}
        const register __m128 kv1 = _mm_load_ps(&kernel[ks]); PRINT_VECTOR(kv1); //{1,2,3,0}
        const register __m128 kv2 = _mm_load_ps(&kernel[ks + ks]); PRINT_VECTOR(kv2); //{1,2,3,0}
        
        register __m128 sum0, sum1, sum2;
        register __m128 inv0, inv1, inv2, inv3; 
        const float* ipStopX = ip + stopX; 
        
        float* op = output + (ip - input); 
        const float* ipX;
        
        _mm_prefetch ( (const char*)(ip + yJump0), _MM_HINT_T0 );
        _mm_prefetch ( (const char*)(ip + yJump1), _MM_HINT_T0 );
        _mm_prefetch ( (const char*)(ip + yJump2), _MM_HINT_T0 );
         
        for (ipX = ip; ipX < ipStopX - 12; ipX += 12) {
             
            // col 0, row 0
            PRINT_VECTOR(kv0); //{1,2,3,0}
            PRINT_VECTOR(kv1); //{4,5,6,0}
            PRINT_VECTOR(kv2); //{7,8,9,0}

            #define CONVOLVE3LB_LOAD(ipX) \
                 inv0 = _mm_load_ps(ipX); PRINT_VECTOR(inv0); \
                 inv1 = _mm_load_ps(ipX + 4); PRINT_VECTOR(inv1); \
                 inv2 = _mm_load_ps(ipX + 8); PRINT_VECTOR(inv2); \
                 inv3 = _mm_load_ps(ipX + 12); PRINT_VECTOR(inv3); \
            
            #define CONVOLVE3LB(kv,sum, inv0, inv1) \
                 PRINT_VECTOR(inv0); \
                 PRINT_VECTOR(inv1); \
                 /* 0 */ \
                 sum += _mm_dp_ps(kv, inv0, 113); PRINT_VECTOR(sum);     /*{68,0,0,0}*/ \
                 /* 1 */ \
                 ROTATE_LEFT(inv0);                                      /*{11,12,13,10}*/ \
                 sum += _mm_dp_ps(kv, inv0, 114); PRINT_VECTOR(sum);     /*{68,74,0,0}*/ \
                 /* 2 */ \
                 ROTATE_LEFT(inv0);                                      /*{12,13,10,11}*/ \
                 inv0 = _mm_movelh_ps(inv0, inv1); PRINT_VECTOR(inv0);   /*{12,13,14,15}*/ \
                 sum += _mm_dp_ps(kv, inv0, 116); PRINT_VECTOR(sum);     /*{68,74,80,0}*/ \
                 /* 3 */ \
                 ROTATE_LEFT(inv0);                                      /*{13,14,15,12}*/ \
                 sum += _mm_dp_ps(kv, inv0, 120); PRINT_VECTOR(sum);     /*{68,74,80,86}*/ 
            
            #define CONVOLVE3LB_LINE(ipX, kv) \
                 CONVOLVE3LB_LOAD(ipX); \
                 PRINT_LABEL("x 0"); \
                 CONVOLVE3LB(kv, sum0, inv0, inv1); \
                 PRINT_LABEL("x 1"); \
                 CONVOLVE3LB(kv, sum1, inv1, inv2); \
                 PRINT_LABEL("x 2"); \
                 CONVOLVE3LB(kv, sum2, inv2, inv3); 

            #define CONVOLVE3LB_LOOPUNROLL(ipX, op) \
                sum0 = sum1 = sum2 = _mm_setzero_ps(); \
                PRINT_LABEL("krow 0"); \
                CONVOLVE3LB_LINE(ipX, kv0); \
                PRINT_LABEL("krow 1"); \
                CONVOLVE3LB_LINE((ipX + is1), kv1); \
                PRINT_LABEL("krow 2"); \
                CONVOLVE3LB_LINE((ipX + is2), kv2); \
                _mm_storeu_ps((op + hk) + (hk * s), sum0); \
                _mm_storeu_ps((op + hk) + (hk * s) + 4, sum1); \
                _mm_storeu_ps((op + hk) + (hk * s) + 8, sum2); \
             
             
            CONVOLVE3LB_LOOPUNROLL(ipX, op);
            CONVOLVE3LB_LOOPUNROLL((ipX + is1), (op + is1));
            CONVOLVE3LB_LOOPUNROLL((ipX + is2), (op + is2));
            op += 12;  
        } //for (int x = 0...
        
        for (; ipX < ipStopX; ipX += 4) {
            CONVOLVE3LB_LOOPUNROLL(ipX, op);
            CONVOLVE3LB_LOOPUNROLL((ipX + is1), (op + is1));
            CONVOLVE3LB_LOOPUNROLL((ipX + is2), (op + is2));
            op += 4;  
        } //for (int x = 0...
        
    } //for (int y = 0...
    stopYY = input + (s * stopY);
    for (const float* ip = input + (s * (stopY - 3)); ip < stopYY; ip += s) {
        const register __m128 kv0 = _mm_load_ps(&kernel[0]); PRINT_VECTOR(kv0); //{1,2,3,0}
        const register __m128 kv1 = _mm_load_ps(&kernel[ks]); PRINT_VECTOR(kv1); //{1,2,3,0}
        const register __m128 kv2 = _mm_load_ps(&kernel[ks + ks]); PRINT_VECTOR(kv2); //{1,2,3,0}
        register __m128 sum0, sum1, sum2;
        register __m128 inv0, inv1, inv2, inv3; 
        const float* ipStopX = ip + stopX; 
        
        float* op = output + (ip - input); 
        const float* ipX;
        
        _mm_prefetch ( (const char*)(ip + yJump0), _MM_HINT_T0 );
        _mm_prefetch ( (const char*)(ip + yJump1), _MM_HINT_T0 );
        _mm_prefetch ( (const char*)(ip + yJump2), _MM_HINT_T0 );
         
        for (ipX = ip; ipX < ipStopX - 12; ipX += 12) {
            sum0 = sum1 = sum2 = _mm_setzero_ps(); 
            CONVOLVE3LB_LOOPUNROLL(ipX, op);
            op += 12;  
        }
        for (; ipX < ipStopX; ipX += 4) {
            CONVOLVE3LB_LOOPUNROLL(ipX, op);
            op += 4;  
        } //for (int x = 0...
    }
    processBoundaries2D (s, w, h, 
                       ks, kw, 
                       input, output, kernel);    
    
}


void loopUnrollConvolve (const int s, const int w, const int h, 
                         const int ks, const int kw, 
                         const float* __restrict input, 
                         float* __restrict output, 
                         const float* kernel) {
    int hk  = kw / 2;                       
    int startX      = 0;
    int stopX       = w - 2 * hk;
    int startY      = 0;
    int stopY       = h - 2 * hk;
    #pragma omp parallel for shared (input, output) 
    for (int y = startY; y < stopY; y++) {
        int x = 0;
        int yXstride = (y + hk) * s;
        for (x = startX; x < stopX - 31;) {
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); ++x;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); ++x;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); ++x;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); ++x;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); ++x;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); ++x;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); ++x;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); ++x;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); ++x;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); ++x;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); ++x;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); ++x;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); ++x;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); ++x;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); ++x;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); ++x; 
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); ++x;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); ++x;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); ++x;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); ++x;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); ++x;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); ++x;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); ++x;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); ++x;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); ++x;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); ++x;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); ++x;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); ++x;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); ++x;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); ++x;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); ++x;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); ++x; 
        } //for (int x = 0...
        for ( ; x < stopX; ++x) { 
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y);
        }                
    } //for (int y = 0...
    processBoundaries2D (s, w, h, 
                       ks, kw, 
                       input, output, kernel);     
}



void prefetchConvolve64 (const int s, const int w, const int h, 
                         const int ks, const int kw, 
                         const float* input, float* output, const float* kernel) {
    int hk = kw / 2;                       
    int startX  = 0;
    int stopX   = w - 2 * (kw / 2);
    int startY  = 0;
    int stopY   = h - 2 * (kw / 2);
    #pragma omp parallel for shared (input, output) 
    for (int y = startY; y < stopY; ++y) {
        int x = 0;
        int yXstride = (y + hk) * s;
        for (x = startX; x < stopX - 15; ) {
            
            for (int k = 0; k < kw; k++) {
                _mm_prefetch ( &input[(y + k) * s + x] + CACHE_LINE_SIZE, _MM_HINT_T0 );
            }    
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
        } //for (int x = 0...
        for ( ; x < stopX; ++x) { 
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y);
        }                
    } //for (int y = 0...
    processBoundaries2D (s, w, h, 
                       ks, kw, 
                       input, output, kernel);    
}

void prefetchConvolve128 (const int s, const int w, const int h, 
                          const int ks, const int kw, 
                          const float* __restrict input, float* __restrict output, const float* __restrict kernel)  {

    int hk = kw / 2;                       
    int startX  = 0;
    int stopX   = w - 2 * (kw / 2);
    int startY  = 0;
    int stopY   = h - 2 * (kw / 2);
    const int offset = CACHE_LINE_SIZE * 2;
    #pragma omp parallel for shared (input, output) 
    for (int y = startY; y < stopY; ++y) {
        int x = 0;
        int yXstride = (y + hk) * s;
        for (x = startX; x < stopX - 31; ) {
            
            for (int k = 0; k < kw; k++) {
                _mm_prefetch ( &input[(y + k) * s + x] + offset, _MM_HINT_T0 );
            }   
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y); x++;
        } //for (int x = 0...
        for ( ; x < stopX; ++x) { 
            output[yXstride + x + hk] = convolution(input, s, kernel, ks, kw, x, y);
        }                
    } //for (int y = 0...
    processBoundaries2D (s, w, h, 
                       ks, kw, 
                       input, output, kernel);    
    
}

//__attribute__ ((noinline))
inline float convolution(const float* __restrict input, const int s, 
                         const float* __restrict kernel, 
                         const int ks, const int kw, 
                         const int x, const int y)   {
    float sum = 0;
    for (int r = 0; r < kw; ++r) {
        const int idxFtmp = r * ks;
        const int idxIntmp = (y + r) * s + x;
        for (int c = 0; c < kw; ++c) { 
            sum += kernel[idxFtmp + c] * input[idxIntmp + c];
        }
    }
    return sum;
}


void sse3CmConvolve (const int s, const int w, const int h, const int ks, 
                     const float* input, float* output, const float* kernel) { 
     int kw = 3;
    int hk = kw / 2;                       
    const int stopX   = w - 2 * (kw / 2); 
    const int stopY   = h - 2 * (kw / 2);  
    const float* stopXX  = input + stopX; 
    
    const int is1 = s;
    const int is2 = s * 2;
    
    register __m128 kv0 = _mm_load_ps(&kernel[0]); PRINT_VECTOR(kv0); //{1,2,3,0}
    register __m128 kv1 = _mm_load_ps(&kernel[ks]); PRINT_VECTOR(kv1); //{1,2,3,0}
    register __m128 kv2 = _mm_load_ps(&kernel[ks + ks]); PRINT_VECTOR(kv2); //{1,2,3,0}

    #pragma omp parallel for shared (kv0, kv1, kv2, input, output) schedule(guided)
    for (const float* ipX = input; ipX < stopXX; ipX += 12) {
        register __m128 sum0, sum1, sum2, sum3;
        register __m128 inv0, inv1, inv2, inv3; 
        const float* ipStopYY = ipX + (s * stopY); 
        float* op = output + (ipX - input); 
        const float* ip = ipX;  
        
        for (int k = 40; k < 60; ++k) {
            _mm_prefetch ( (const char*)(ip + s * k), _MM_HINT_T0 );
        }
        
        for (ip = ipX; ip < ipStopYY; ip += s) {
            
            _mm_prefetch ( (const char*)(ip + s * 61), _MM_HINT_T0 );
            
            sum0 = sum1 = sum2 = sum3 = _mm_setzero_ps(); 
                     
            // col 0, row 0
            PRINT_VECTOR(kv0); //{1,2,3,0}
            PRINT_VECTOR(kv1); //{4,5,6,0}
            PRINT_VECTOR(kv2); //{7,8,9,0}
    
            #define CONVOLVE3CM2_LOAD(ip) \
                inv0 = _mm_load_ps(ip); PRINT_VECTOR(inv0); \
                inv1 = _mm_load_ps(ip + 4); PRINT_VECTOR(inv1); \
                inv2 = _mm_load_ps(ip + 8); PRINT_VECTOR(inv2); \
                inv3 = _mm_load_ps(ip + 12); PRINT_VECTOR(inv3); \
    
            #define CONVOLVE3CM2(kv,sum, inv0, inv1) \
                PRINT_VECTOR(kv); \
                PRINT_VECTOR(inv0); \
                PRINT_VECTOR(inv1); \
                /* 0 */ \
                sum += _mm_dp_ps(kv, inv0, 113); PRINT_VECTOR(sum);     /*{68,0,0,0}*/ \
                /* 1 */ \
                ROTATE_LEFT(inv0);                                      /*{11,12,13,10}*/ \
                sum += _mm_dp_ps(kv, inv0, 114); PRINT_VECTOR(sum);     /*{68,74,0,0}*/ \
                /* 2 */ \
                ROTATE_LEFT(inv0);                                      /*{12,13,10,11}*/ \
                inv0 = _mm_movelh_ps(inv0, inv1); PRINT_VECTOR(inv0);   /*{12,13,14,15}*/ \
                sum += _mm_dp_ps(kv, inv0, 116); PRINT_VECTOR(sum);     /*{68,74,80,0}*/ \
                /* 3 */ \
                ROTATE_LEFT(inv0);                                      /*{13,14,15,12}*/ \
                sum += _mm_dp_ps(kv, inv0, 120); PRINT_VECTOR(sum);     /*{68,74,80,86}*/ 
            
            #define CONVOLVE3CM2_LOOPUNROLL(ip, kv) \
                CONVOLVE3CM2_LOAD(ip); \
                PRINT_LABEL("x 0"); \
                CONVOLVE3CM2(kv, sum0, inv0, inv1); \
                PRINT_LABEL("x 1"); \
                CONVOLVE3CM2(kv, sum1, inv1, inv2); \
                PRINT_LABEL("x 2"); \
                CONVOLVE3CM2(kv, sum2, inv2, inv3); 
                         
            PRINT_LABEL("krow 0");
            CONVOLVE3CM2_LOOPUNROLL(ip, kv0);
            
            PRINT_LABEL("krow 1");
            CONVOLVE3CM2_LOOPUNROLL((ip + is1), kv1);
            
            PRINT_LABEL("krow 2");
            CONVOLVE3CM2_LOOPUNROLL((ip + is2), kv2);
            
            _mm_storeu_ps((op + hk) + (hk * s), sum0); 
            _mm_storeu_ps((op + hk) + (hk * s) + 4, sum1); 
            _mm_storeu_ps((op + hk) + (hk * s) + 8, sum2); 
            
            op += s;  
        } //for (int x = 0...
    } //for (int y = 0...
    processBoundaries2D (s, w, h, 
                       ks, kw, 
                       input, output, kernel);    
    
}

  
  

void sse3Convolve (const int s, const int w, const int h, const int ks, 
                   const float* input, float* output, const float* kernel) {
    int kw = 3;
    
    int hk = kw / 2;                       
    const int stopY   = h - 2 * (kw / 2);  
    const int stopXX  = (w - 2 * (kw / 2)); // - (w % 4); 
    const float* stopYY = &input[s * stopY];
    
    const int is1 = s;
    const int is2 = s * 2;

    PRINT_LINE(); 
 
    #pragma omp parallel for shared (input, output) 
    for (const float* ip = input; ip < stopYY; ip += s) {
        //FIXME Create constant
        const register __m128 kv0 = _mm_load_ps(&kernel[0]); PRINT_VECTOR(kv0); //{1,2,3,0}
        const register __m128 kv1 = _mm_load_ps(&kernel[ks]); PRINT_VECTOR(kv1); //{4,5,6,0}
        const register __m128 kv2 = _mm_load_ps(&kernel[ks + ks]); PRINT_VECTOR(kv2); //{7,8,9,0}
        
        register __m128 sum0, sum1, sum2;
        register __m128 inv0, inv1, inv2, inv3; 
        const float* ipStopX = ip + stopXX; 
        float* op = output + (ip - input); 
        const float* ipX;
      
        
        for (ipX = ip ;ipX < ipStopX; ipX += 12) {
            sum0 = sum1 = sum2 = _mm_setzero_ps(); 
            // col 0, row 0
            PRINT_VECTOR(kv0); //{1,2,3,0}
            PRINT_VECTOR(kv1); //{4,5,6,0}
            PRINT_VECTOR(kv2); //{7,8,9,0}
            
            #define CONVOLVE3_LOAD(ipX) \
            inv0 = _mm_load_ps(ipX); PRINT_VECTOR(inv0); \
            inv1 = _mm_load_ps(ipX + 4); PRINT_VECTOR(inv1); \
            inv2 = _mm_load_ps(ipX + 8); PRINT_VECTOR(inv2); \
            inv3 = _mm_load_ps(ipX + 12); PRINT_VECTOR(inv3); \
            
            #define CONVOLVE3(kv,sum, inv0, inv1) \
            PRINT_VECTOR(inv0); \
            PRINT_VECTOR(inv1); \
            /* 0 */ \
            sum += _mm_dp_ps(kv, inv0, 113); PRINT_VECTOR(sum);     /*{68,0,0,0}*/ \
            /* 1 */ \
            ROTATE_LEFT(inv0);                                      /*{11,12,13,10}*/ \
            sum += _mm_dp_ps(kv, inv0, 114); PRINT_VECTOR(sum);     /*{68,74,0,0}*/ \
            /* 2 */ \
            ROTATE_LEFT(inv0);                                      /*{12,13,10,11}*/ \
            inv0 = _mm_movelh_ps(inv0, inv1); PRINT_VECTOR(inv0);   /*{12,13,14,15}*/ \
            sum += _mm_dp_ps(kv, inv0, 116); PRINT_VECTOR(sum);     /*{68,74,80,0}*/ \
            /* 3 */ \
            ROTATE_LEFT(inv0);                                      /*{13,14,15,12}*/ \
            sum += _mm_dp_ps(kv, inv0, 120); PRINT_VECTOR(sum);     /*{68,74,80,86}*/ 
            
            PRINT_LABEL("krow 0");
            CONVOLVE3_LOAD(ipX)
            PRINT_LABEL("x 0");
            CONVOLVE3(kv0, sum0, inv0, inv1);
            PRINT_LABEL("x 1");
            CONVOLVE3(kv0, sum1, inv1, inv2); 
            PRINT_LABEL("x 2");
            CONVOLVE3(kv0, sum2, inv2, inv3); 
            
            PRINT_LABEL("krow 1");
            CONVOLVE3_LOAD((ipX + is1))
            PRINT_LABEL("x 0");
            CONVOLVE3(kv1, sum0, inv0, inv1);
            PRINT_LABEL("x 1");
            CONVOLVE3(kv1, sum1, inv1, inv2); 
            PRINT_LABEL("x 2");
            CONVOLVE3(kv1, sum2, inv2, inv3); 
            
            PRINT_LABEL("krow 2");
            CONVOLVE3_LOAD((ipX + is2))
            PRINT_LABEL("x 0");
            CONVOLVE3(kv2, sum0, inv0, inv1);
            PRINT_LABEL("x 1");
            CONVOLVE3(kv2, sum1, inv1, inv2); 
            PRINT_LABEL("x 2");
            CONVOLVE3(kv2, sum2, inv2, inv3); 
            
            _mm_storeu_ps((op + hk) + (hk * s), sum0); 
            _mm_storeu_ps((op + hk) + (hk * s) + 4, sum1); 
            _mm_storeu_ps((op + hk) + (hk * s) + 8, sum2); 
            op += 12;  
        } //for (int x = 0...
        PRINT_LINE(); 
    } //for (int y = 0...
    processBoundaries2D (s, w, h, 
                       ks, kw, 
                       input, output, kernel);    
    
}


void sse5Convolve (const int s, const int w, const int h, const int ks, 
                   const float* input, float* output, const float* kernel) {
    int kw = 5;
    int hk = kw / 2;                       
    const int stopY   = h - 2 * (kw / 2);  
    const int stopXX  = (w - 2 * (kw / 2)); // - (w % 4);

    const int ks1 = ks;
    const int ks2 = ks * 2;
    const int ks3 = ks * 3;
    const int ks4 = ks * 4;
    
    const int is1 = s;
    const int is2 = s * 2;
    const int is3 = s * 3;
    const int is4 = s * 4; 
    
    const float* totalBytes = &input[(s * (stopY - 1) + stopXX)];
    const float* kp = kernel;        
 
    #pragma omp parallel for shared (input, output) 
    for (const float* ip = input; ip < totalBytes; ip += s) {
        register __m128 sum0, sum1;
        register __m128 kv0, kv1, inv0, inv1, inv2; 
        const float* ipStopX = ip + stopXX; 
        PRINT("# 1 #"); 
        PRINT((long)ip);  
        PRINT((long)ipStopX); 
        PRINT(s); 
        
        float* op = output + (ip - input); 
        const float* ipX;
        for (ipX = ip; ipX < ipStopX; ipX += 8) {
        
            PRINT("# 2 #"); 
            PRINT((long)ip);  
            
            sum0 = sum1 = _mm_setzero_ps();
            
            #define CONVOLVE0(kp, ipX, dp1, dp2) \
            kv0 = _mm_load_ps(kp); PRINT_VECTOR(kv0); \
            kv1 = _mm_load_ps(kp + 4); PRINT_VECTOR(kv1); \
            kv1 = _mm_blend_ps(kv1, kv0, 14); PRINT_VECTOR(kv1); \
            inv0 = _mm_load_ps(ipX); PRINT_VECTOR(inv0); \
            inv1 = _mm_load_ps(ipX + 4); PRINT_VECTOR(inv1); \
            inv2 = _mm_load_ps(ipX + 8); PRINT_VECTOR(inv2); \
            sum0 += _mm_dp_ps(kv0, inv0, dp1) + _mm_dp_ps (kv1, inv1, dp2); PRINT_VECTOR(sum0); 
            
            #define CONVOLVE1(sum, inv0, inv1, dpm1, dpm2) \
            ROTATE_RIGHT(kv0); \
            ROTATE_RIGHT(kv1); \
            PRINT_VECTOR(kv0); PRINT_VECTOR(kv1); \
            PRINT_VECTOR(inv0); PRINT_VECTOR(inv1); \
            sum += _mm_dp_ps (kv0, inv0, dpm1) + _mm_dp_ps (kv1, inv1, dpm2); PRINT_VECTOR(sum);
            
            _mm_prefetch ( ipX + 128, _MM_HINT_NTA );
            _mm_prefetch ( ipX + is1 + 128, _MM_HINT_NTA );
            _mm_prefetch ( ipX + is2 + 128, _MM_HINT_NTA );
            _mm_prefetch ( ipX + is3 + 128, _MM_HINT_NTA );
            _mm_prefetch ( ipX + is4 + 128, _MM_HINT_NTA );
            
            #define CONVOLVE_CHUNK(kp, ipX) \
            PRINT_LABEL("0,0");     CONVOLVE0(kp, ipX, 241, 17); \
            PRINT_LABEL("1,0");     CONVOLVE1(sum0, inv0, inv1, 226, 50); \
            PRINT_LABEL("2,0");     CONVOLVE1(sum0, inv0, inv1, 196, 116); \
            PRINT_LABEL("3,0");     CONVOLVE1(sum0, inv0, inv1, 136, 248); \
            PRINT_LABEL("4,0");     CONVOLVE1(sum1, inv1, inv2, 241, 17); \
            PRINT_LABEL("5,0");     CONVOLVE1(sum1, inv1, inv2, 226, 50); \
            PRINT_LABEL("6,0");     CONVOLVE1(sum1, inv1, inv2, 196, 116); \
            PRINT_LABEL("7,0");     CONVOLVE1(sum1, inv1, inv2, 136, 248); \
            
            CONVOLVE_CHUNK(kp, ipX);
            CONVOLVE_CHUNK((kp + ks1), (ipX + is1));
            CONVOLVE_CHUNK((kp + ks2), (ipX + is2));
            CONVOLVE_CHUNK((kp + ks3), (ipX + is3));
            CONVOLVE_CHUNK((kp + ks4), (ipX + is4));
            
            _mm_storeu_ps((op + hk) + (hk * s), sum0); 
            _mm_storeu_ps((op + hk) + (hk * s) + 4, sum1); 
            PRINT("# 3 #"); 
            PRINT((long)ip);  
            op += 8;  
        } //for (int x = 0...
    }              
    processBoundaries2D (s, w, h, 
                       ks, kw, 
                       input, output, kernel);    
    
}




void sse7Convolve (const int s, const int w, const int h, const int ks, 
                   const float* input, float* output, const float* kernel) {

    int kw = 7;
    int hk = kw / 2;                       
    const int stopY   = h - 2 * (kw / 2);  
    const int stopXX  = (w - 2 * (kw / 2)); // - (w % 4);
    
    const int ks1 = ks;
    const int ks2 = ks * 2;
    const int ks3 = ks * 3;
    const int ks4 = ks * 4;
    const int ks5 = ks * 5;
    const int ks6 = ks * 6;
    
    const int is1 = s;
    const int is2 = s * 2;
    const int is3 = s * 3;
    const int is4 = s * 4; 
    const int is5 = s * 5; 
    const int is6 = s * 6; 
    
    const float* totalBytes = &input[(s * (stopY - 1) + stopXX)];
    const float* kp = kernel;        
 
     #pragma omp parallel for shared (input, output) 
     for (const float* ip = input; ip < totalBytes; ip += s) {
         register __m128 sum0, sum1;
         register __m128 kv0, kv1, kv2, inv0, inv1, inv2; 
         const float* ipStopX = ip + stopXX; 
         float* op = output + (ip - input); 
         const float* ipX;
         for (ipX = ip; ipX < ipStopX; ipX += 8) {
             sum0 = sum1 = _mm_setzero_ps();
                 
             _mm_prefetch ( ipX + 128, _MM_HINT_T0 );
             _mm_prefetch ( ipX + is1 + 128, _MM_HINT_T0 );
             _mm_prefetch ( ipX + is2 + 128, _MM_HINT_T0 );
             _mm_prefetch ( ipX + is3 + 128, _MM_HINT_T0 );
             _mm_prefetch ( ipX + is4 + 128, _MM_HINT_T0 );
             _mm_prefetch ( ipX + is5 + 128, _MM_HINT_T0 );
             _mm_prefetch ( ipX + is6 + 128, _MM_HINT_T0 );

             #define CONVOLVE70(sum, kp, ipX) \
                 inv0 = _mm_load_ps(ipX); PRINT_VECTOR(inv0); \
                 inv1 = _mm_load_ps(ipX + 4); PRINT_VECTOR(inv1); \
                 inv2 = _mm_load_ps(ipX + 8); PRINT_VECTOR(inv2); \
                 kv0 = _mm_load_ps(kp); PRINT_VECTOR(kv0); \
                 kv1 = _mm_load_ps(kp + 4); PRINT_VECTOR(kv1); \
                 kv2 = kv1; \
                 kv2 = _mm_shuffle_ps(kv2, kv2, _MM_SHUFFLE(1,0,3,2)); PRINT_VECTOR(kv2); \
                 sum += _mm_dp_ps(kv0, inv0, 241) + _mm_dp_ps (kv1, inv1, 113); PRINT_VECTOR(sum); \
                 /* 1 */ \
                 ROTATE_RIGHT(kv0); \
                 ROTATE_RIGHT(kv1); \
                 kv1 = _mm_blend_ps(kv1, kv0, 1); PRINT_VECTOR(kv1); \
                 sum += _mm_dp_ps(kv0, inv0, 226) + _mm_dp_ps (kv1, inv1, 242); PRINT_VECTOR(sum); \
                 /* 2 */ \
                 ROTATE_RIGHT(kv0); \
                 ROTATE_RIGHT(kv1); \
                 kv1 = _mm_blend_ps(kv1, kv0, 1); PRINT_VECTOR(kv1); \
                 sum += _mm_dp_ps(kv0, inv0, 196) + _mm_dp_ps (kv1, inv1, 244) + _mm_dp_ps (kv2, inv2, 20);  PRINT_VECTOR(sum); \
                 /* 3 */ \
                 ROTATE_RIGHT(kv0); \
                 ROTATE_RIGHT(kv1); \
                 kv1 = _mm_blend_ps(kv1, kv0, 1); PRINT_VECTOR(kv1); \
                 ROTATE_RIGHT(kv2); \
                 sum += _mm_dp_ps(kv0, inv0, 136) + _mm_dp_ps (kv1, inv1, 248) + _mm_dp_ps (kv2, inv2, 56); PRINT_VECTOR(sum);
             
             
             //------------------------------------
             #define CONVOLVE71(sum, ipX) \
                 /* 0 */ \
                 inv0 = inv1; PRINT_VECTOR(inv0); \
                 inv1 = inv2; PRINT_VECTOR(inv1); \
                 inv2 = _mm_load_ps(ipX + 12); PRINT_VECTOR(inv2); \
                 ROTATE_RIGHT(kv0); \
                 kv1 = _mm_shuffle_ps(kv2, kv2, _MM_SHUFFLE(2,1,0,3)); PRINT_VECTOR(kv1); \
                 ROTATE_LEFT(kv2); \
                 sum += _mm_dp_ps(kv0, inv0, 241) + _mm_dp_ps (kv1, inv1, 113); PRINT_VECTOR(sum); \
                 /* 1 */ \
                 ROTATE_RIGHT(kv0); \
                 ROTATE_RIGHT(kv1); \
                 kv1 = _mm_blend_ps(kv1, kv0, 1); PRINT_VECTOR(kv1); \
                 sum += _mm_dp_ps(kv0, inv0, 226) + _mm_dp_ps (kv1, inv1, 242); PRINT_VECTOR(sum); \
                 /* 2 */ \
                 ROTATE_RIGHT(kv0); \
                 ROTATE_RIGHT(kv1); \
                 kv1 = _mm_blend_ps(kv1, kv0, 1); PRINT_VECTOR(kv1); \
                 sum += _mm_dp_ps(kv0, inv0, 196) + _mm_dp_ps (kv1, inv1, 244) + _mm_dp_ps (kv2, inv2, 20);  PRINT_VECTOR(sum);  \
                 /* 3 */ \
                 ROTATE_RIGHT(kv0); \
                 ROTATE_RIGHT(kv1); \
                 kv1 = _mm_blend_ps(kv1, kv0, 1); PRINT_VECTOR(kv1); \
                 ROTATE_RIGHT(kv2); \
                 sum += _mm_dp_ps(kv0, inv0, 136) + _mm_dp_ps (kv1, inv1, 248) + _mm_dp_ps (kv2, inv2, 56); PRINT_VECTOR(sum); 
             

             #define CONVOLVE7_CHUNK(kp, ipX) \
                 PRINT_LABEL("0,0");     CONVOLVE70(sum0, kp, ipX); \
                 PRINT_LABEL("1,0");     CONVOLVE71(sum1, ipX); 


             CONVOLVE7_CHUNK(kp, ipX);
             CONVOLVE7_CHUNK((kp + ks1), (ipX + is1));
             CONVOLVE7_CHUNK((kp + ks2), (ipX + is2));
             CONVOLVE7_CHUNK((kp + ks3), (ipX + is3));
             CONVOLVE7_CHUNK((kp + ks4), (ipX + is4));
             CONVOLVE7_CHUNK((kp + ks5), (ipX + is5));
             CONVOLVE7_CHUNK((kp + ks6), (ipX + is6));
             
             PRINT_INLINE("########## "); PRINT_VECTOR(sum0);
             PRINT_INLINE("########## "); PRINT_VECTOR(sum1);

             _mm_storeu_ps((op + hk) + (hk * s), sum0); 
             _mm_storeu_ps((op + hk) + (hk * s) + 4, sum1); 
                op += 8;  
         } //for (int x = 0...
     } //for (int y = 0...
    processBoundaries2D (s, w, h, 
                       ks, kw, 
                       input, output, kernel);    
     
}

void sse9Convolve (const int s, const int w, const int h, const int ks, 
                   const float* input, float* output, const float* kernel) {
    int kw = 9;
    int hk = kw / 2;                       
    const int stopY   = h - 2 * (kw / 2);    
 
    const int stopXX  = (w - 2 * (kw / 2)); //- (w % 4);

    PRINT_LINE();
    PRINT(w);
    PRINT(s);
    
    const int ks1 = ks;
    const int ks2 = ks * 2;
    const int ks3 = ks * 3;
    const int ks4 = ks * 4;
    const int ks5 = ks * 5;
    const int ks6 = ks * 6;
    const int ks7 = ks * 7;
    const int ks8 = ks * 8;
    
    const int is1 = s;
    const int is2 = s * 2;
    const int is3 = s * 3;
    const int is4 = s * 4; 
    const int is5 = s * 5; 
    const int is6 = s * 6; 
    const int is7 = s * 7; 
    const int is8 = s * 8;  
    
    const float* totalBytes = &input[(s * (stopY - 1) + stopXX)];
    PRINT(stopXX); 
    PRINT((long)input); 
    const float* kp = kernel;         
    #pragma omp parallel for shared (input, output) 
    for (const float* ip = input; ip < totalBytes; ip += s) {
        register __m128 sum0, sum1;
        register __m128 kv0, kv1, kv2, kv3, inv0, inv1, inv2; 
        const float* ipStopX = ip + stopXX; 
        float* op = output + (ip - input);
        const float* ipX;
        for (ipX = ip; ipX < ipStopX; ipX += 8) {
            sum0 = sum1 = _mm_setzero_ps();

                
            _mm_prefetch ( ipX + 128, _MM_HINT_T0 );
            _mm_prefetch ( ipX + is1 + 128, _MM_HINT_T0 );
            _mm_prefetch ( ipX + is2 + 128, _MM_HINT_T0 );
            _mm_prefetch ( ipX + is3 + 128, _MM_HINT_T0 );
            _mm_prefetch ( ipX + is4 + 128, _MM_HINT_T0 );
            _mm_prefetch ( ipX + is5 + 128, _MM_HINT_T0 );
            _mm_prefetch ( ipX + is6 + 128, _MM_HINT_T0 );
            _mm_prefetch ( ipX + is7 + 128, _MM_HINT_T0 );
            _mm_prefetch ( ipX + is8 + 128, _MM_HINT_T0 );

            #define CONVOLVE90(sum, kp, ipX) \
                /* 0 */ \
                inv0 = _mm_load_ps(ipX); PRINT_VECTOR(inv0); \
                inv1 = _mm_load_ps(ipX + 4); PRINT_VECTOR(inv1); \
                inv2 = _mm_load_ps(ipX + 8); PRINT_VECTOR(inv2); \
                kv0 = _mm_load_ps(kp); PRINT_VECTOR(kv0); \
                kv1 = _mm_load_ps(kp + 4); PRINT_VECTOR(kv1); \
                kv2 = _mm_load_ps(kp + 8); PRINT_VECTOR(kv2); \
                kv2 = _mm_blend_ps(kv2, kv1, 14); PRINT_VECTOR(kv2); \
                kv3 = kv2; PRINT_VECTOR(kv3); \
                sum += _mm_dp_ps(kv0, inv0, 241) + _mm_dp_ps (kv1, inv1, 241) + _mm_dp_ps (kv2, inv2, 17); PRINT_VECTOR(sum); \
                /* 1 */ \
                ROTATE_RIGHT(kv0); \
                ROTATE_RIGHT(kv1); \
                kv1 = _mm_blend_ps(kv1, kv0, 1); PRINT_VECTOR(kv1); \
                ROTATE_RIGHT(kv2); \
                sum += _mm_dp_ps(kv0, inv0, 226) + _mm_dp_ps (kv1, inv1, 242) + _mm_dp_ps (kv2, inv2, 50); PRINT_VECTOR(sum); \
                /* 2 */ \
                ROTATE_RIGHT(kv0); \
                ROTATE_RIGHT(kv1); \
                kv1 = _mm_blend_ps(kv1, kv0, 1); PRINT_VECTOR(kv1); \
                ROTATE_RIGHT(kv2); \
                sum += _mm_dp_ps(kv0, inv0, 196) + _mm_dp_ps (kv1, inv1, 244) + _mm_dp_ps (kv2, inv2, 116);    PRINT_VECTOR(sum); \
                /* 3 */ \
                ROTATE_RIGHT(kv0); \
                ROTATE_RIGHT(kv1); \
                kv1 = _mm_blend_ps(kv1, kv0, 1); PRINT_VECTOR(kv1); \
                ROTATE_RIGHT(kv2); \
                sum += _mm_dp_ps(kv0, inv0, 136) + _mm_dp_ps (kv1, inv1, 248) + _mm_dp_ps (kv2, inv2, 248); PRINT_VECTOR(sum);
            
            
            //------------------------------------
            #define CONVOLVE91(sum, ipX) \
                /* 0 */ \
                inv0 = inv1; PRINT_VECTOR(inv0); \
                inv1 = inv2; PRINT_VECTOR(inv1); \
                inv2 = _mm_load_ps(ipX + 12); PRINT_VECTOR(inv2); \
                ROTATE_RIGHT(kv0); \
                kv1 = _mm_blend_ps(kv2, kv1, 8); PRINT_VECTOR(kv1); \
                ROTATE_RIGHT(kv1); \
                kv2 = kv3; PRINT_VECTOR(kv2); \
                sum += _mm_dp_ps(kv0, inv0, 241) + _mm_dp_ps (kv1, inv1, 241) + _mm_dp_ps (kv2, inv2, 17); PRINT_VECTOR(sum); \
                /* 1 */ \
                ROTATE_RIGHT(kv0); \
                ROTATE_RIGHT(kv1); \
                kv1 = _mm_blend_ps(kv1, kv0, 1); PRINT_VECTOR(kv1); \
                ROTATE_RIGHT(kv2); \
                sum += _mm_dp_ps(kv0, inv0, 226) + _mm_dp_ps (kv1, inv1, 242) + _mm_dp_ps (kv2, inv2, 50); PRINT_VECTOR(sum); \
                /* 2 */ \
                ROTATE_RIGHT(kv0); \
                ROTATE_RIGHT(kv1); \
                kv1 = _mm_blend_ps(kv1, kv0, 1); PRINT_VECTOR(kv1); \
                ROTATE_RIGHT(kv2); \
                sum += _mm_dp_ps(kv0, inv0, 196) + _mm_dp_ps (kv1, inv1, 244) + _mm_dp_ps (kv2, inv2, 116);    PRINT_VECTOR(sum); \
                /* 3 */ \
                ROTATE_RIGHT(kv0); \
                ROTATE_RIGHT(kv1); \
                kv1 = _mm_blend_ps(kv1, kv0, 1); PRINT_VECTOR(kv1); \
                ROTATE_RIGHT(kv2); \
                sum += _mm_dp_ps(kv0, inv0, 136) + _mm_dp_ps (kv1, inv1, 248) + _mm_dp_ps (kv2, inv2, 248); PRINT_VECTOR(sum);
            

            #define CONVOLVE9_CHUNK(kp, ipX) \
                PRINT_LABEL("0,0");     CONVOLVE90(sum0, kp, ipX); \
                PRINT_LABEL("1,0");     CONVOLVE91(sum1, ipX); 


            CONVOLVE9_CHUNK(kp, ipX);
            CONVOLVE9_CHUNK((kp + ks1), (ipX + is1));
            CONVOLVE9_CHUNK((kp + ks2), (ipX + is2));
            CONVOLVE9_CHUNK((kp + ks3), (ipX + is3));
            CONVOLVE9_CHUNK((kp + ks4), (ipX + is4));
            CONVOLVE9_CHUNK((kp + ks5), (ipX + is5));
            CONVOLVE9_CHUNK((kp + ks6), (ipX + is6));
            CONVOLVE9_CHUNK((kp + ks7), (ipX + is7));
            CONVOLVE9_CHUNK((kp + ks8), (ipX + is8));
            
            PRINT_INLINE("########## "); PRINT_VECTOR(sum0);
            PRINT_INLINE("########## "); PRINT_VECTOR(sum1);

            _mm_storeu_ps((op + hk) + (hk * s), sum0); 
            _mm_storeu_ps((op + hk) + (hk * s) + 4, sum1); 
            PRINT((long)ipX + 8);
            PRINT((long)ipStopX);
            PRINT(((long)ipX + 8) - (long)ipStopX);
            
            op += 8;  
        } //for (int x = 0...
    } //for (int y = 0...
    processBoundaries2D (s, w, h, 
                       ks, kw, 
                       input, output, kernel);    
    
}

  

void sse11Convolve (const int s, const int w, const int h, const int ks, 
                    const float* input, float* output, const float* kernel) {
    int kw = 11;
    int hk = kw / 2;                       
    const int stopY   = h - 2 * (kw / 2);  
    
    const int stopXX  = (w - 2 * (kw / 2)); //- (w % 4);
    
    const int ks1 = ks;
    const int ks2 = ks * 2;
    const int ks3 = ks * 3;
    const int ks4 = ks * 4;
    const int ks5 = ks * 5;
    const int ks6 = ks * 6;
    const int ks7 = ks * 7;
    const int ks8 = ks * 8;
    const int ks9 = ks * 9;
    const int ks10 = ks * 10;
    
    const int is1 = s;
    const int is2 = s * 2;
    const int is3 = s * 3;
    const int is4 = s * 4; 
    const int is5 = s * 5; 
    const int is6 = s * 6; 
    const int is7 = s * 7; 
    const int is8 = s * 8;  
    const int is9 = s * 9;  
    const int is10 = s * 10;   
 
    const float* totalBytes = &input[(s * (stopY - 1) + stopXX)];
    const float* kp = kernel;        
    #pragma omp parallel for shared (input, output) 
    for (const float* ip = input; ip < totalBytes; ip += s) {
        register __m128 sum0, sum1;
        register __m128 kv0, kv1, kv2, kv3, inv0, inv1, inv2, inv3; 
        const float* ipStopX = ip + stopXX; 
        float* op = output + (ip - input);
        const float* ipX;
        for (ipX = ip; ipX < ipStopX; ipX += 8) {
            sum0 = sum1 = _mm_setzero_ps();
            
            
            _mm_prefetch ( ipX + 128, _MM_HINT_T0 );
            _mm_prefetch ( ipX + is1 + 128, _MM_HINT_T0 );
            _mm_prefetch ( ipX + is2 + 128, _MM_HINT_T0 );
            _mm_prefetch ( ipX + is3 + 128, _MM_HINT_T0 );
            _mm_prefetch ( ipX + is4 + 128, _MM_HINT_T0 );
            _mm_prefetch ( ipX + is5 + 128, _MM_HINT_T0 );
            _mm_prefetch ( ipX + is6 + 128, _MM_HINT_T0 );
            _mm_prefetch ( ipX + is7 + 128, _MM_HINT_T0 );
            _mm_prefetch ( ipX + is8 + 128, _MM_HINT_T0 );
            _mm_prefetch ( ipX + is9 + 128, _MM_HINT_T0 );
            _mm_prefetch ( ipX + is10 + 128, _MM_HINT_T0 );
            
            #define CONVOLVE110(sum, kp, ipX) \
                /* 0 */ \
                inv0 = _mm_load_ps(ipX); PRINT_VECTOR(inv0); \
                inv1 = _mm_load_ps(ipX + 4); PRINT_VECTOR(inv1); \
                inv2 = _mm_load_ps(ipX + 8); PRINT_VECTOR(inv2); \
                inv3 = _mm_load_ps(ipX + 12); PRINT_VECTOR(inv3); \
                kv0 = _mm_load_ps(kp); PRINT_VECTOR(kv0); \
                kv1 = _mm_load_ps(kp + 4); PRINT_VECTOR(kv1); \
                kv2 = _mm_load_ps(kp + 8); PRINT_VECTOR(kv2); \
                kv2 = _mm_blend_ps(kv2, kv1, 8); PRINT_VECTOR(kv2); \
                kv3 = _mm_shuffle_ps(kv2, kv2, _MM_SHUFFLE(1,0,3,2)); PRINT_VECTOR(kv3); \
                sum += _mm_dp_ps(kv0, inv0, 241) + _mm_dp_ps (kv1, inv1, 241) + _mm_dp_ps (kv2, inv2, 113); PRINT_VECTOR(sum); \
                /* 1 */ \
                ROTATE_RIGHT(kv0); \
                ROTATE_RIGHT(kv1); \
                kv1 = _mm_blend_ps(kv1, kv0, 1); PRINT_VECTOR(kv1); \
                ROTATE_RIGHT(kv2); \
                sum += _mm_dp_ps(kv0, inv0, 226) + _mm_dp_ps (kv1, inv1, 242) + _mm_dp_ps (kv2, inv2, 242); PRINT_VECTOR(sum); \
                /* 2 */ \
                ROTATE_RIGHT(kv0); \
                ROTATE_RIGHT(kv1); \
                ROTATE_RIGHT(kv2); \
                kv2 = _mm_blend_ps(kv2, kv1, 1); PRINT_VECTOR(kv2); \
                kv1 = _mm_blend_ps(kv1, kv0, 1); PRINT_VECTOR(kv1); \
                PRINT_VECTOR(kv0); \
                PRINT_VECTOR(kv1); \
                PRINT_VECTOR(kv2); \
                PRINT_VECTOR(kv3); \
                sum += _mm_dp_ps(kv0, inv0, 196) + _mm_dp_ps (kv1, inv1, 244) + _mm_dp_ps (kv2, inv2, 244) + _mm_dp_ps (kv3, inv3, 20); PRINT_VECTOR(sum); \
                /* 3 */ \
                ROTATE_RIGHT(kv0); \
                ROTATE_RIGHT(kv1); \
                ROTATE_RIGHT(kv2); \
                ROTATE_RIGHT(kv3); \
                kv2 = _mm_blend_ps(kv2, kv1, 1); PRINT_VECTOR(kv2); \
                kv1 = _mm_blend_ps(kv1, kv0, 1); PRINT_VECTOR(kv1); \
                sum += _mm_dp_ps(kv0, inv0, 136) + _mm_dp_ps (kv1, inv1, 248) + _mm_dp_ps (kv2, inv2, 248) + _mm_dp_ps (kv3, inv3, 56); PRINT_VECTOR(sum); 
            
            
            //------------------------------------
            #define CONVOLVE111(sum, ipX) \
                /* 0 */ \
                inv0 = inv1; PRINT_VECTOR(inv0); \
                inv1 = inv2; PRINT_VECTOR(inv1); \
                inv2 = inv3; \
                inv3 = _mm_load_ps(ipX + 16); PRINT_VECTOR(inv3); \
                ROTATE_RIGHT(kv0); \
                kv1 = _mm_blend_ps(kv2, kv1, 8); PRINT_VECTOR(kv1); \
                ROTATE_RIGHT(kv1); \
                kv2 = kv3; PRINT_VECTOR(kv2); \
                ROTATE_RIGHT(kv2); \
                ROTATE_LEFT(kv3); \
                sum += _mm_dp_ps(kv0, inv0, 241) + _mm_dp_ps (kv1, inv1, 241) + _mm_dp_ps (kv2, inv2, 113); PRINT_VECTOR(sum); \
                /* 1 */ \
                ROTATE_RIGHT(kv0); \
                ROTATE_RIGHT(kv1); \
                kv1 = _mm_blend_ps(kv1, kv0, 1); PRINT_VECTOR(kv1); \
                ROTATE_RIGHT(kv2); \
                sum += _mm_dp_ps(kv0, inv0, 226) + _mm_dp_ps (kv1, inv1, 242) + _mm_dp_ps (kv2, inv2, 242); PRINT_VECTOR(sum); \
                /* 2 */ \
                ROTATE_RIGHT(kv0); \
                ROTATE_RIGHT(kv1); \
                ROTATE_RIGHT(kv2); \
                kv2 = _mm_blend_ps(kv2, kv1, 1); PRINT_VECTOR(kv2); \
                kv1 = _mm_blend_ps(kv1, kv0, 1); PRINT_VECTOR(kv1); \
                PRINT_VECTOR(kv0); \
                PRINT_VECTOR(kv1); \
                PRINT_VECTOR(kv2); \
                PRINT_VECTOR(kv3); \
                sum += _mm_dp_ps(kv0, inv0, 196) + _mm_dp_ps (kv1, inv1, 244) + _mm_dp_ps (kv2, inv2, 244) + _mm_dp_ps (kv3, inv3, 20); PRINT_VECTOR(sum); \
                /* 3 */ \
                ROTATE_RIGHT(kv0); \
                ROTATE_RIGHT(kv1); \
                ROTATE_RIGHT(kv2); \
                ROTATE_RIGHT(kv3); \
                kv2 = _mm_blend_ps(kv2, kv1, 1); PRINT_VECTOR(kv2); \
                kv1 = _mm_blend_ps(kv1, kv0, 1); PRINT_VECTOR(kv1); \
                sum += _mm_dp_ps(kv0, inv0, 136) + _mm_dp_ps (kv1, inv1, 248) + _mm_dp_ps (kv2, inv2, 248) + _mm_dp_ps (kv3, inv3, 56); PRINT_VECTOR(sum); 
            
            
            #define CONVOLVE11_CHUNK(kp, ipX) \
                PRINT_LABEL("0,0");     CONVOLVE110(sum0, kp, ipX); \
                PRINT_LABEL("1,0");     CONVOLVE111(sum1, ipX); 
            
            
            CONVOLVE11_CHUNK(kp, ipX);
            CONVOLVE11_CHUNK((kp + ks1), (ipX + is1));
            CONVOLVE11_CHUNK((kp + ks2), (ipX + is2));
            CONVOLVE11_CHUNK((kp + ks3), (ipX + is3));
            CONVOLVE11_CHUNK((kp + ks4), (ipX + is4));
            CONVOLVE11_CHUNK((kp + ks5), (ipX + is5));
            CONVOLVE11_CHUNK((kp + ks6), (ipX + is6));
            CONVOLVE11_CHUNK((kp + ks7), (ipX + is7));
            CONVOLVE11_CHUNK((kp + ks8), (ipX + is8));
            CONVOLVE11_CHUNK((kp + ks9), (ipX + is9));
            CONVOLVE11_CHUNK((kp + ks10), (ipX + is10));
            
            PRINT_INLINE("########## "); PRINT_VECTOR(sum0);
            PRINT_INLINE("########## "); PRINT_VECTOR(sum1);
            
            _mm_storeu_ps((op + hk) + (hk * s), sum0); 
            _mm_storeu_ps((op + hk) + (hk * s) + 4, sum1); 
            
            op += 8;  
        } //for (int x = 0...
    }   
    processBoundaries2D (s, w, h, 
                       ks, kw, 
                       input, output, kernel);    
    
}

void separableConvolve (const int s, const int w, const int h, const int kw, 
                        const float* __restrict input, float* __restrict output, 
                        const float* __restrict kernelX, 
                        const float* __restrict kernelY,
                        const bool processBoundaries) {    
                        
    const int hk = kw / 2;
    const int startX  = 0;
    const int stopX   = w - 2 * hk;
    const int startY  = 0;
    const int stopY   = h - 2 * hk;
     
    #pragma omp parallel for shared (input, output)
    for (int y = startY; y < stopY; ++y) {
     
        for (int x = 0; x < kw - 1; ++x) {
            float sum = 0;
            int idxIntmp = y * s + x;
            for (int r = 0; r < kw; ++r) {
                sum += kernelY[r] * input[idxIntmp + (r * s)];
            } 
            output[(y + hk) * s + x + hk] = sum; 
        }
        
        for (int x = startX; x < stopX; ++x) {  
            float sum = 0;
            int idxIntmp = y * s + x + (kw - 1);
            for (int r = 0; r < kw; ++r) {
                sum += kernelY[r] * input[idxIntmp + (r * s)];
            }
            output[(y + hk) * s + x + (kw - 1) + hk] = sum; 
            sum = 0;
            for (int c = 0; c < kw; ++c) {
                sum += kernelX[c] * output[(y + hk) * s + x + c + hk];
            }
            output[(y + hk) * s + x + hk] = sum; 
        }   
        
    }
    if(processBoundaries) {
        processBoundariesS2D (s, w, h, 
                              kw, 
                              input, output, 
                              kernelX, kernelY);        
    }
}
 
void separableLoopBlockConvolve (const int s, const int w, const int h, const int kw, 
                        const float* input, float* output, const float* kernelX, const float* kernelY, 
                        const int xBlock, const int yBlock) {
    
    const int hk = kw / 2;
    int startX  = 0;
    int stopX   = w - 2 * hk;
    int startY  = 0;
    int stopY   = h - 1; //(h - 2 * (kw / 2));
    const int kernelOffset = 2 * hk;
    
    #pragma omp parallel for shared (input, output) 
    for (int y = startY; y < kw - 1; ++y) {
        for (int x = startX; x < stopX; ++x) {
            float sum = 0;
            int idxIntmp = y * s + x;
            for (int c = 0; c < kw; ++c) {
                sum += kernelX[c] * input[idxIntmp + c];
            }
            output[(y + 1) * s + (x + hk)] = sum;
        }
    }
    
    for (int y = kw - 1; y < stopY; y += yBlock) {
        #pragma omp parallel for shared (input, output) 
        for (int x = startX; x < stopX; x += xBlock) {
            for (int yy = y; yy < min(y + yBlock, stopY); ++yy) {
                for (int xx = x; xx < min(x + xBlock, stopX); ++xx) {
                    float sum = 0;
                    int idxIntmp = yy * s + xx;
                    for (int c = 0; c < kw; ++c) {
                        sum += kernelX[c] * input[idxIntmp + c];
                    }
                    output[(idxIntmp + s) + hk] = sum;
                    
                    sum = 0;
                    idxIntmp = yy - kernelOffset + 1;
                    for (int r = 0; r < kw; ++r) {
                        sum += kernelY[r] * output[(idxIntmp + r) * s + xx + hk];
                    }
                    output[((idxIntmp + hk) - 1) * s + xx + hk] = sum;
                } //for (int x = 0...
            } //for (int y = 0...
        }
    }
    
    #pragma omp parallel for shared (input, output) 
    for (int x = startX; x < stopX; ++x) {
        float sumX = 0;
        int y = h - 1;
        int idxIntmp = y * s + x;
        for (int c = 0; c < kw; ++c) {
            sumX += kernelX[c] * input[idxIntmp + c];
        }
        float sum = 0;
        idxIntmp = y - kernelOffset + 1;
        int r = 0;
        for (; r < kw - 1; ++r) {
            sum += kernelY[r] * output[(idxIntmp + r) * s + x + hk];
        }
        sum += kernelY[r] * sumX;
        
        output[(y - 1) * s + x + hk] = sum;
    }
        
    processBoundariesS2D (s, w, h, 
                         kw, 
                         input, output, kernelX, kernelY);        
    //printImage(w, h, s, output);
}

void scSSE (const int s, const int w, const int h, int kw, 
            const float* input, float* output, 
            const float* kernelX, const float* kernelY) {
              
    int hk = kw / 2;
    int startX  = 0;
    int stopX   = w - hk * 2; 
    int startY  = 0;
    int stopY   = h - (kw + (4 - (kw % 4)));
                     
    #pragma omp parallel for shared (input, output)
    for (int y = startY; y < stopY; ++y) {
        for (int x = startX; x < stopX; x += 16) { 
            register __m128 sum0, sum1, sum2, sum3, 
                            sumy0, sumy1, sumy2, sumy3, sumy4,  
                            kvx, kvy; 
            sum0 = sum1 = sum2 = sum3 = 
            sumy0 = sumy1 = sumy2 = sumy3 = sumy4 = _mm_setzero_ps();
            __m128 iv0, iv1, iv2, iv3, iv4;
            for (int r = 0; r < kw; r += 4) {
                const int idxIntmp = (y + r) * s + x; 
                kvx = _mm_load_ps(kernelY + r);                                      PRINT_VECTOR(kvx); 
                
                kvy =  _mm_shuffle_ps(kvx, kvx, 0);                                  PRINT_VECTOR(kvy);
                iv0 = _mm_load_ps(&input[idxIntmp]);                           PRINT_VECTOR(iv0);
                sumy0 += kvy * iv0;                                                  PRINT_VECTOR(sumy0);
            
                iv0 = _mm_load_ps(&input[idxIntmp + 4]);                       PRINT_VECTOR(iv0);
                sumy1 += kvy * iv0;                                                  PRINT_VECTOR(sumy1);        
                
                iv0 = _mm_load_ps(&input[idxIntmp + 8]);                       PRINT_VECTOR(iv0);
                sumy2 += kvy * iv0;                                                  PRINT_VECTOR(sumy2);
                
                iv0 = _mm_load_ps(&input[idxIntmp + 12]);                      PRINT_VECTOR(iv0);
                sumy3 += kvy * iv0;                                                  PRINT_VECTOR(sumy3);
                
                
                kvy = _mm_shuffle_ps(kvx, kvx, 85);                                 PRINT_VECTOR(kvy);
                iv0 = _mm_load_ps(&input[idxIntmp + s]);             PRINT_VECTOR(iv0);
                sumy0 += kvy * iv0;                                                  PRINT_VECTOR(sumy0);
                
                iv0 = _mm_load_ps(&input[idxIntmp + 4 + s]);         PRINT_VECTOR(iv0);
                sumy1 += kvy * iv0;                                                  PRINT_VECTOR(sumy1);        
                
                iv0 = _mm_load_ps(&input[idxIntmp + 8 + s]);         PRINT_VECTOR(iv0);
                sumy2 += kvy * iv0;                                                  PRINT_VECTOR(sumy2);
                
                iv0 = _mm_load_ps(&input[idxIntmp + 12 + s]);        PRINT_VECTOR(iv0);
                sumy3 += kvy * iv0;                                                  PRINT_VECTOR(sumy3);
                
                
                kvy = _mm_shuffle_ps(kvx, kvx, 170);                                PRINT_VECTOR(kvy);
                iv0 = _mm_load_ps(&input[idxIntmp + (s * 2)]);       PRINT_VECTOR(iv0);
                sumy0 += kvy * iv0;                                                  PRINT_VECTOR(sumy0);
                
                iv0 = _mm_load_ps(&input[idxIntmp + 4 + (s * 2)]);   PRINT_VECTOR(iv0);
                sumy1 += kvy * iv0;                                                  PRINT_VECTOR(sumy1);        
                
                iv0 = _mm_load_ps(&input[idxIntmp + 8 + (s * 2)]);   PRINT_VECTOR(iv0);
                sumy2 += kvy * iv0;                                                  PRINT_VECTOR(sumy2);
                
                iv0 = _mm_load_ps(&input[idxIntmp + 12 + (s * 2)]);  PRINT_VECTOR(iv0);
                sumy3 += kvy * iv0;                                                  PRINT_VECTOR(sumy3);
                
                
                kvy = _mm_shuffle_ps(kvx, kvx, 255);                                PRINT_VECTOR(kvy);
                iv0 = _mm_load_ps(&input[idxIntmp + (s * 3)]);       PRINT_VECTOR(iv0);
                sumy0 += kvy * iv0;                                                  PRINT_VECTOR(sumy0);
                
                iv0 = _mm_load_ps(&input[idxIntmp + 4 + (s * 3)]);   PRINT_VECTOR(iv0);
                sumy1 += kvy * iv0;                                                  PRINT_VECTOR(sumy1);        
                
                iv0 = _mm_load_ps(&input[idxIntmp + 8 + (s * 3)]);   PRINT_VECTOR(iv0);
                sumy2 += kvy * iv0;                                                  PRINT_VECTOR(sumy2);
                
                iv0 = _mm_load_ps(&input[idxIntmp + 12 + (s * 3)]);  PRINT_VECTOR(iv0);
                sumy3 += kvy * iv0;                                                  PRINT_VECTOR(sumy3);
                
            } //for (int r = 0...
            
            PRINT_VECTOR_TRACE (sumy0);
            PRINT_VECTOR_TRACE (sumy1);
            PRINT_VECTOR_TRACE (sumy2);
            PRINT_VECTOR_TRACE (sumy3);
            
            iv0 = sumy0;                                                             PRINT_VECTOR(iv0);
            iv1 = sumy1;                                                             PRINT_VECTOR(iv1);
            iv2 = sumy2;                                                             PRINT_VECTOR(iv2);
            iv3 = sumy3;                                                             PRINT_VECTOR(iv3);
            
            PRINT_LABEL("kvx"); 
            for (int c = 0; c < kw; c += 4) {
                sumy4 = _mm_setzero_ps();
                PRINT_LABEL("kvy + 16"); 
                for (int r = 0; r < kw; r += 4) {
                    
                    const int idxIntmp = (y + r) * s + x + c; 
                    
                    kvx = _mm_load_ps(kernelY + r);                                      PRINT_VECTOR(kvx);
                    
                    kvy =  _mm_shuffle_ps(kvx, kvx, 0);                                 PRINT_VECTOR(kvy);
                    iv4 = _mm_load_ps(&input[idxIntmp + 16]);                      PRINT_VECTOR(iv4);
                    sumy4 += kvy * iv4;                                                 PRINT_VECTOR(sumy4);
                    
                    
                    kvy = _mm_shuffle_ps(kvx, kvx, 85);                                 PRINT_VECTOR(kvy);
                    iv4 = _mm_load_ps(&input[idxIntmp + 16 + s]);        PRINT_VECTOR(iv4);
                    sumy4 += kvy * iv4;    PRINT_VECTOR(sumy0);                           PRINT_VECTOR(sumy4);
                    
                                        
                    kvy = _mm_shuffle_ps(kvx, kvx, 170);                                PRINT_VECTOR(kvy);
                    iv4 = _mm_load_ps(&input[idxIntmp + 16 + (s * 2)]);  PRINT_VECTOR(iv4);
                    sumy4 += kvy * iv4;    PRINT_VECTOR(sumy0);                           PRINT_VECTOR(sumy4);
                    
                    kvy = _mm_shuffle_ps(kvx, kvx, 255);                                PRINT_VECTOR(kvy);
                    iv4 = _mm_load_ps(&input[idxIntmp + 16 + (s * 3)]);  PRINT_VECTOR(iv4);
                    sumy4 += kvy * iv4;                                                  PRINT_VECTOR(sumy4);
                    
                } //for (int r = 0...
                PRINT_VECTOR_TRACE (sumy4);
             
                PRINT_LABEL("end kvy + 16"); 
                
                kvx = _mm_load_ps(&kernelX[c]);                                          PRINT_VECTOR(kvx);
                PRINT_VECTOR_TRACE (kvx);
                iv4 = sumy4;                                                             PRINT_VECTOR(iv4);
                
                PRINT_LABEL("sum0"); 
                sum0 += _mm_dp_ps(kvx, iv0, 241);    PRINT_VECTOR(sum0); 
                sum1 += _mm_dp_ps(kvx, iv1, 241);    PRINT_VECTOR(sum1);
                sum2 += _mm_dp_ps(kvx, iv2, 241);    PRINT_VECTOR(sum2);
                sum3 += _mm_dp_ps(kvx, iv3, 241);    PRINT_VECTOR(sum3);
                 
                BLEND_ROTATE4_LEFT(iv0, iv1, iv2, iv3, iv4);

                PRINT_LABEL("sum1"); 
                sum0 += _mm_dp_ps(kvx, iv0, 242);    PRINT_VECTOR(sum0);
                sum1 += _mm_dp_ps(kvx, iv1, 242);    PRINT_VECTOR(sum1);
                sum2 += _mm_dp_ps(kvx, iv2, 242);    PRINT_VECTOR(sum2);
                sum3 += _mm_dp_ps(kvx, iv3, 242);    PRINT_VECTOR(sum3);
                
                BLEND_ROTATE4_LEFT(iv0, iv1, iv2, iv3, iv4);

                PRINT_LABEL("sum2"); 
                sum0 += _mm_dp_ps(kvx, iv0, 244);    PRINT_VECTOR(sum0);
                sum1 += _mm_dp_ps(kvx, iv1, 244);    PRINT_VECTOR(sum1);
                sum2 += _mm_dp_ps(kvx, iv2, 244);    PRINT_VECTOR(sum2);
                sum3 += _mm_dp_ps(kvx, iv3, 244);    PRINT_VECTOR(sum3);
                
                BLEND_ROTATE4_LEFT(iv0, iv1, iv2, iv3, iv4);

                PRINT_LABEL("sum3"); 
                sum0 += _mm_dp_ps(kvx, iv0, 248);    PRINT_VECTOR(sum0);
                sum1 += _mm_dp_ps(kvx, iv1, 248);    PRINT_VECTOR(sum1);
                sum2 += _mm_dp_ps(kvx, iv2, 248);    PRINT_VECTOR(sum2);
                sum3 += _mm_dp_ps(kvx, iv3, 248);    PRINT_VECTOR(sum3);
                
                BLEND_ROTATE4_LEFT(iv0, iv1, iv2, iv3, iv4);
                
            }
            
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk)], sum0);         PRINT_VECTOR(sum0);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 4], sum1);     PRINT_VECTOR(sum1);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 8], sum2);     PRINT_VECTOR(sum2);
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk) + 12], sum3);    PRINT_VECTOR(sum3);
        } //for (int x = 0...
    } //for (int y = 0...
    
    int hh = h - stopY;
    const float *inputT = &input[stopY * s];
    float *outputT = &output[stopY * s];
    
    separableConvolve (s, w, hh, kw, inputT, outputT, kernelX, kernelY, false);
    
    
    processBoundariesS2D (s, w, h, 
                         kw, 
                         input, output, kernelX, kernelY);        
}


void sc3SSE (const int s, const int w, const int h, 
             const float* input, float* output, const float* kernelX, const float* kernelY) {

    const int kw = 3;
    const int hk = kw / 2;
    
    int stopX   = w;
    int startY  = 0;
    int stopY   = h - 2 * (kw / 2);

      
    #pragma omp parallel for shared (input, output) 
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
        
        __m128 inv0 = _mm_load_ps(&input[y * s]);            PRINT_VECTOR(inv0);
        __m128 inv1 = _mm_load_ps(&input[(y + 1) * s]);      PRINT_VECTOR(inv1);
        __m128 inv2 = _mm_load_ps(&input[(y + 2) * s]);      PRINT_VECTOR(inv2);
        
        PRINT(y); 
                    
        sum0 += kvy0 * inv0;    PRINT_VECTOR(sum0);
        sum0 += kvy1 * inv1;    PRINT_VECTOR(sum0);
        sum0 += kvy2 * inv2;    PRINT_VECTOR(sum0);
        
        for (int x = 0; x < stopX; x += 4) {
            
            PRINT_LINE(); 
            PRINT(x); 
            PRINT_VECTOR(sum0)
            PRINT_VECTOR(sum1)
            
            inv0 = _mm_load_ps(&input[y * s + x + 4]);            PRINT_VECTOR(inv0);
            inv1 = _mm_load_ps(&input[(y + 1) * s + x + 4]);      PRINT_VECTOR(inv1);
            inv2 = _mm_load_ps(&input[(y + 2) * s + x + 4]);      PRINT_VECTOR(inv2);
            
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
            PRINT((y + hk) * s + (x + hk));
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk)], inv0); PRINT_VECTOR(inv0);
            PRINT(output[(y + hk) * s + (x + hk)]);
            PRINT(output[(y + hk) * s + (x + hk) + 1]);
            PRINT(output[(y + hk) * s + (x + hk) + 2]);
            PRINT(output[(y + hk) * s + (x + hk) + 3]);            
            
            sum0 = sum1;
            sum1 = _mm_setzero_ps();
        }
     
    }
    
    processBoundariesS2D (s, w, h, 
                         kw, 
                         input, output, kernelX, kernelY);        
}

void sc5SSE (const int s, const int w, const int h, 
             const float* input, float* output, const float* kernelX, const float* kernelY) {

    const int kw = 5;
    const int hk = kw / 2;
    
    int stopX   = w;
    int startY  = 0;
    int stopY   = h - 2 * (kw / 2);

    #pragma omp parallel for shared (input, output) 
    for (int y = startY; y < stopY; ++y) {
     
        //TODO For gaussian we only need kw / 2 + 1 kernel vectors since guassian function repeats 
        register __m128 inv0 = _mm_load_ps(kernelY);                          PRINT_VECTOR(inv0);
        const register __m128 kvy0 = _mm_shuffle_ps(inv0, inv0, 0);             PRINT_VECTOR(kvy0);
        const register __m128 kvy1 = _mm_shuffle_ps(inv0, inv0, 85);            PRINT_VECTOR(kvy1);
        const register __m128 kvy2 = _mm_shuffle_ps(inv0, inv0, 170);           PRINT_VECTOR(kvy2);
        const register __m128 kvy3 = _mm_shuffle_ps(inv0, inv0, 255);           PRINT_VECTOR(kvy3);

        inv0 = _mm_load_ps(kernelY + 4);                                      PRINT_VECTOR(inv0);
        const register __m128 kvy4 = _mm_shuffle_ps(inv0, inv0, 0);             PRINT_VECTOR(kvy4);

        register __m128 kvx0 = _mm_load_ps(kernelX);                        PRINT_VECTOR(kvx0);
        register __m128 kvx1 = _mm_load_ps(kernelX + 4);                    PRINT_VECTOR(kvx1);
         
        __m128 sum0, sum1;
        sum0 = sum1 = _mm_setzero_ps();
        
        PRINT_LABEL("inv"); 
        
        inv0 = _mm_load_ps(&input[y * s]);            PRINT_VECTOR(inv0);
        register __m128 inv1 = _mm_load_ps(&input[(y + 1) * s]);      PRINT_VECTOR(inv1);
        register __m128 inv2 = _mm_load_ps(&input[(y + 2) * s]);      PRINT_VECTOR(inv2);
        register __m128 inv3 = _mm_load_ps(&input[(y + 3) * s]);      PRINT_VECTOR(inv3);
        register __m128 inv4 = _mm_load_ps(&input[(y + 4) * s]);      PRINT_VECTOR(inv4);
        
        PRINT(y); 
                    
        sum0 += kvy0 * inv0;    PRINT_VECTOR(sum0);
        sum0 += kvy1 * inv1;    PRINT_VECTOR(sum0);
        sum0 += kvy2 * inv2;    PRINT_VECTOR(sum0);
        sum0 += kvy3 * inv3;    PRINT_VECTOR(sum0);
        sum0 += kvy4 * inv4;    PRINT_VECTOR(sum0);
        
        for (int x = 0; x < stopX; x += 4) {
            
            PRINT_LINE(); 
            PRINT(x); 
            PRINT_VECTOR(sum0)
            PRINT_VECTOR(sum1)
            
            inv0 = _mm_load_ps(&input[y * s + x + 4]);            PRINT_VECTOR(inv0);
            inv1 = _mm_load_ps(&input[(y + 1) * s + x + 4]);      PRINT_VECTOR(inv1);
            inv2 = _mm_load_ps(&input[(y + 2) * s + x + 4]);      PRINT_VECTOR(inv2);
            inv3 = _mm_load_ps(&input[(y + 3) * s + x + 4]);      PRINT_VECTOR(inv3);
            inv4 = _mm_load_ps(&input[(y + 4) * s + x + 4]);      PRINT_VECTOR(inv4);
            
            sum1 += kvy0 * inv0;    PRINT_VECTOR(sum1);
            sum1 += kvy1 * inv1;    PRINT_VECTOR(sum1);
            sum1 += kvy2 * inv2;    PRINT_VECTOR(sum1);
            sum1 += kvy3 * inv3;    PRINT_VECTOR(sum1);
            sum1 += kvy4 * inv4;    PRINT_VECTOR(sum1);
            
            //TODO parei aqui, fazer para no x agora.
            
            inv0 = _mm_dp_ps(sum0, kvx0, 241) + _mm_dp_ps(sum1, kvx1, 17);          PRINT_VECTOR(inv0);
            
            ROTATE_RIGHT(kvx1);                                                     
            ROTATE_RIGHT_BLEND(kvx0, kvx1);                                         
            inv0 += _mm_dp_ps(sum0, kvx0, 226) + _mm_dp_ps(sum1, kvx1, 50);         PRINT_VECTOR(inv0);
            
            ROTATE_RIGHT(kvx1);                                                     
            ROTATE_RIGHT_BLEND(kvx0, kvx1);                                         
            inv0 += _mm_dp_ps(sum0, kvx0, 196) + _mm_dp_ps(sum1, kvx1, 116);         PRINT_VECTOR(inv0);
            
            ROTATE_RIGHT(kvx1);                                                     
            ROTATE_RIGHT_BLEND(kvx0, kvx1);                                         
            inv0 += _mm_dp_ps(sum0, kvx0, 136) + _mm_dp_ps(sum1, kvx1, 248);         PRINT_VECTOR(inv0);
            
            ROTATE_RIGHT(kvx0);
            ROTATE_RIGHT(kvx1);
            
            PRINT_LABEL("sum"); 
            PRINT((y + hk) * s + (x + hk));
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk)], inv0);                 PRINT_VECTOR(inv0);
            PRINT(output[(y + hk) * s + (x + hk)]);
            PRINT(output[(y + hk) * s + (x + hk) + 1]);
            PRINT(output[(y + hk) * s + (x + hk) + 2]);
            PRINT(output[(y + hk) * s + (x + hk) + 3]);            
            
            sum0 = sum1;
            sum1 = _mm_setzero_ps();
        }
     
    }
    
    processBoundariesS2D (s, w, h, 
                         kw, 
                         input, output, kernelX, kernelY);        
    
}


void sc7SSE (const int s, const int w, const int h, 
             const float* input, float* output, 
             const float* kernelX, const float* kernelY) {

    const int kw = 7;
    const int hk = kw / 2;
    
    int stopX   = w;
    int startY  = 0;
    int stopY   = h - 2 * (kw / 2);

      
    #pragma omp parallel for shared (input, output) 
    for (int y = startY; y < stopY; ++y) {
     
        //TODO For gaussian we only need kw / 2 + 1 kernel vectors since guassian function repeats 
        register __m128 kvx0 = _mm_load_ps(kernelY);                        PRINT_VECTOR(kvx0);
        const register __m128 kvy0 = _mm_shuffle_ps(kvx0, kvx0, 0);         PRINT_VECTOR(kvy0);
        const register __m128 kvy1 = _mm_shuffle_ps(kvx0, kvx0, 85);        PRINT_VECTOR(kvy1);
        const register __m128 kvy2 = _mm_shuffle_ps(kvx0, kvx0, 170);       PRINT_VECTOR(kvy2);
        const register __m128 kvy3 = _mm_shuffle_ps(kvx0, kvx0, 255);       PRINT_VECTOR(kvy3);

        kvx0 = _mm_load_ps(kernelY + 4);                                    PRINT_VECTOR(kvx0);
        const register __m128 kvy4 = _mm_shuffle_ps(kvx0, kvx0, 0);         PRINT_VECTOR(kvy4);
        const register __m128 kvy5 = _mm_shuffle_ps(kvx0, kvx0, 85);        PRINT_VECTOR(kvy5);
        const register __m128 kvy6 = _mm_shuffle_ps(kvx0, kvx0, 170);       PRINT_VECTOR(kvy6);

        kvx0 = _mm_load_ps(kernelX);                                        PRINT_VECTOR(kvx0);
        register __m128 kvx1 = _mm_load_ps(kernelX + 4);                    PRINT_VECTOR(kvx1);
         
        __m128 sum0, sum1, sum2;
        sum0 = sum1 = sum2 = _mm_setzero_ps();
        
        PRINT_LABEL("inv"); 
        PRINT(y); 
        
        register __m128 inv0 = _mm_load_ps(&input[y * s]);            PRINT_VECTOR(inv0);
        register __m128 inv1 = _mm_load_ps(&input[(y + 1) * s]);      PRINT_VECTOR(inv1);
        register __m128 inv2 = _mm_load_ps(&input[(y + 2) * s]);      PRINT_VECTOR(inv2);
        sum0 += kvy0 * inv0;    PRINT_VECTOR(sum0);
        sum0 += kvy1 * inv1;    PRINT_VECTOR(sum0);
        sum0 += kvy2 * inv2;    PRINT_VECTOR(sum0);
        
        inv0 = _mm_load_ps(&input[(y + 3) * s]);      PRINT_VECTOR(inv0);
        inv1 = _mm_load_ps(&input[(y + 4) * s]);      PRINT_VECTOR(inv1);
        inv2 = _mm_load_ps(&input[(y + 5) * s]);      PRINT_VECTOR(inv2);
        sum0 += kvy3 * inv0;    PRINT_VECTOR(sum0);
        sum0 += kvy4 * inv1;    PRINT_VECTOR(sum0);
        sum0 += kvy5 * inv2;    PRINT_VECTOR(sum0);
        
        inv0 = _mm_load_ps(&input[(y + 6) * s]);      PRINT_VECTOR(inv0);
        sum0 += kvy6 * inv0;    PRINT_VECTOR(sum0);
        
        inv0 = _mm_load_ps(&input[y * s + 4]);            PRINT_VECTOR(inv0);
        inv1 = _mm_load_ps(&input[(y + 1) * s + 4]);      PRINT_VECTOR(inv1);
        inv2 = _mm_load_ps(&input[(y + 2) * s + 4]);      PRINT_VECTOR(inv2);
        sum1 += kvy0 * inv0;    PRINT_VECTOR(sum1);
        sum1 += kvy1 * inv1;    PRINT_VECTOR(sum1);
        sum1 += kvy2 * inv2;    PRINT_VECTOR(sum1);
        
        inv0 = _mm_load_ps(&input[(y + 3) * s + 4]);      PRINT_VECTOR(inv0);
        inv1 = _mm_load_ps(&input[(y + 4) * s + 4]);      PRINT_VECTOR(inv1);
        inv2 = _mm_load_ps(&input[(y + 5) * s + 4]);      PRINT_VECTOR(inv2);
        sum1 += kvy3 * inv0;    PRINT_VECTOR(sum1);
        sum1 += kvy4 * inv1;    PRINT_VECTOR(sum1);
        sum1 += kvy5 * inv2;    PRINT_VECTOR(sum1);
        
        inv0 = _mm_load_ps(&input[(y + 6) * s + 4]);      PRINT_VECTOR(inv0);
        sum1 += kvy6 * inv0;    PRINT_VECTOR(sum1);
        
        
        for (int x = 0; x < stopX; x += 4) {
            
            PRINT_LINE(); 
            PRINT(x); 
            PRINT_VECTOR(sum0)
            PRINT_VECTOR(sum1)
            
            inv0 = _mm_load_ps(&input[y * s + x + 8]);            PRINT_VECTOR(inv0);
            inv1 = _mm_load_ps(&input[(y + 1) * s + x + 8]);      PRINT_VECTOR(inv1);
            inv2 = _mm_load_ps(&input[(y + 2) * s + x + 8]);      PRINT_VECTOR(inv2);
            sum2 += kvy0 * inv0;                                                 PRINT_VECTOR(sum2);
            sum2 += kvy1 * inv1;                                                 PRINT_VECTOR(sum2);
            sum2 += kvy2 * inv2;                                                 PRINT_VECTOR(sum2);
            
            inv0 = _mm_load_ps(&input[(y + 3) * s + x + 8]);      PRINT_VECTOR(inv0);
            inv1 = _mm_load_ps(&input[(y + 4) * s + x + 8]);      PRINT_VECTOR(inv1);
            inv2 = _mm_load_ps(&input[(y + 5) * s + x + 8]);      PRINT_VECTOR(inv2);
            sum2 += kvy3 * inv0;    PRINT_VECTOR(sum2);
            sum2 += kvy4 * inv1;    PRINT_VECTOR(sum2);
            sum2 += kvy5 * inv2;    PRINT_VECTOR(sum2);
            
            inv0 = _mm_load_ps(&input[(y + 6) * s + x + 8]);      PRINT_VECTOR(inv0);
            sum2 += kvy6 * inv0;    PRINT_VECTOR(sum2);
            
            //TODO parei aqui, fazer para no x agora.
            
            inv0 = _mm_dp_ps(sum0, kvx0, 241) + _mm_dp_ps(sum1, kvx1, 113);          PRINT_VECTOR(inv0);
            inv1 = _mm_setzero_ps(); // kvx2
            
            
            ROTATE_RIGHT(kvx1);                                                     
            ROTATE_RIGHT_BLEND(kvx0, kvx1);                                         
            inv0 += _mm_dp_ps(sum0, kvx0, 226) + _mm_dp_ps(sum1, kvx1, 242);  PRINT_VECTOR(inv0);
            
            PRINT_LABEL("Terceiro"); 
            ROTATE_RIGHT_BLEND(kvx1, /*kvx2*/ inv1 ); 
            ROTATE_RIGHT_BLEND(kvx0, kvx1);                                         
            inv0 += _mm_dp_ps(sum0, kvx0, 196) + _mm_dp_ps(sum1, kvx1, 244) + _mm_dp_ps(sum2, /*kvx2*/ inv1, 20);    PRINT_VECTOR(inv0);
            
            PRINT_LABEL("Quarto"); 
            PRINT_VECTOR(sum0);
            PRINT_VECTOR(sum1);
            PRINT_VECTOR(sum2);
            
            ROTATE_RIGHT(/*kvx2*/ inv1);
            ROTATE_RIGHT_BLEND(kvx1, /*kvx2*/ inv1 );
            ROTATE_RIGHT_BLEND(kvx0, kvx1);                                         
            inv0 += _mm_dp_ps(sum0, kvx0, 136) + _mm_dp_ps(sum1, kvx1, 248) + _mm_dp_ps(sum2, /*kvx2*/ inv1, 56);    PRINT_VECTOR(inv0);
            
            PRINT_LABEL("Rotate"); 
            
            ROTATE_RIGHT(kvx0);
            
            
            //FIXME O erro está aqui. kvx1 não está carregado de forma correta. Dar um shuffle com inv1 para pegar de volta os elementos.
            kvx1 = _mm_blend_ps(kvx1, inv1, 7);
            ROTATE_RIGHT(kvx1);
            
            PRINT_LABEL("sum"); 
            PRINT((y + hk) * s + (x + hk));
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk)], inv0);                 PRINT_VECTOR(inv0);
            PRINT(output[(y + hk) * s + (x + hk)]);
            PRINT(output[(y + hk) * s + (x + hk) + 1]);
            PRINT(output[(y + hk) * s + (x + hk) + 2]);
            PRINT(output[(y + hk) * s + (x + hk) + 3]);            
            
            sum0 = sum1;
            sum1 = sum2;
            sum2 = _mm_setzero_ps();
        }
    }
    processBoundariesS2D (s, w, h, 
                         kw, 
                         input, output, kernelX, kernelY);        
    
    //printImage(w, h, s, output);
    
}

void sc9SSE (const int s, const int w, const int h, 
             const float* input, float* output, 
             const float* kernelX, const float* kernelY) {

    const int kw = 9;
    const int hk = kw / 2;
    
    int stopX   = w;
    int startY  = 0;
    int stopY   = h - 2 * (kw / 2);

      
    #pragma omp parallel for shared (input, output) 
    for (int y = startY; y < stopY; ++y) {
     
        //Load kernel lines. Make it const, so we don't have to load every time
        register __m128 inv0 = _mm_load_ps(kernelY);                        PRINT_VECTOR(inv0);
        const register __m128 kvy0 = _mm_shuffle_ps(inv0, inv0, 0);         PRINT_VECTOR(kvy0);
        const register __m128 kvy1 = _mm_shuffle_ps(inv0, inv0, 85);        PRINT_VECTOR(kvy1);
        const register __m128 kvy2 = _mm_shuffle_ps(inv0, inv0, 170);       PRINT_VECTOR(kvy2);
        const register __m128 kvy3 = _mm_shuffle_ps(inv0, inv0, 255);       PRINT_VECTOR(kvy3);

        inv0 = _mm_load_ps(kernelY + 4);                                    PRINT_VECTOR(inv0);
        const register __m128 kvy4 = _mm_shuffle_ps(inv0, inv0, 0);         PRINT_VECTOR(kvy4);
        const register __m128 kvy5 = _mm_shuffle_ps(inv0, inv0, 85);        PRINT_VECTOR(kvy5);
        const register __m128 kvy6 = _mm_shuffle_ps(inv0, inv0, 170);       PRINT_VECTOR(kvy6);
        const register __m128 kvy7 = _mm_shuffle_ps(inv0, inv0, 255);       PRINT_VECTOR(kvy7);

        inv0 = _mm_load_ps(kernelY + 8);                                    PRINT_VECTOR(inv0);
        const register __m128 kvy8 = _mm_shuffle_ps(inv0, inv0, 0);         PRINT_VECTOR(kvy8);

         //vectors that will hold y dot product results
        __m128 sum0, sum1, sum2;
        sum0 = sum1 = sum2 = _mm_setzero_ps();
        
        PRINT_LABEL("inv"); 
        PRINT(y); 
        
        //calculate y dot products
        
        //x
        inv0 = _mm_load_ps(&input[y * s]);                            PRINT_VECTOR(inv0);
        register __m128 inv1 = _mm_load_ps(&input[(y + 1) * s]);      PRINT_VECTOR(inv1);
        register __m128 inv2 = _mm_load_ps(&input[(y + 2) * s]);      PRINT_VECTOR(inv2);
        sum0 += kvy0 * inv0;    PRINT_VECTOR(sum0);
        sum0 += kvy1 * inv1;    PRINT_VECTOR(sum0);
        sum0 += kvy2 * inv2;    PRINT_VECTOR(sum0);
        
        inv0 = _mm_load_ps(&input[(y + 3) * s]);      PRINT_VECTOR(inv0);
        inv1 = _mm_load_ps(&input[(y + 4) * s]);      PRINT_VECTOR(inv1);
        inv2 = _mm_load_ps(&input[(y + 5) * s]);      PRINT_VECTOR(inv2);
        sum0 += kvy3 * inv0;    PRINT_VECTOR(sum0);
        sum0 += kvy4 * inv1;    PRINT_VECTOR(sum0);
        sum0 += kvy5 * inv2;    PRINT_VECTOR(sum0);
        
        inv0 = _mm_load_ps(&input[(y + 6) * s]);      PRINT_VECTOR(inv0);
        inv1 = _mm_load_ps(&input[(y + 7) * s]);      PRINT_VECTOR(inv1);
        inv2 = _mm_load_ps(&input[(y + 8) * s]);      PRINT_VECTOR(inv2);
        sum0 += kvy6 * inv0;    PRINT_VECTOR(sum0);
        sum0 += kvy7 * inv1;    PRINT_VECTOR(sum0);
        sum0 += kvy8 * inv2;    PRINT_VECTOR(sum0);
        
        //x + 4
        inv0 = _mm_load_ps(&input[y * s + 4]);            PRINT_VECTOR(inv0);
        inv1 = _mm_load_ps(&input[(y + 1) * s + 4]);      PRINT_VECTOR(inv1);
        inv2 = _mm_load_ps(&input[(y + 2) * s + 4]);      PRINT_VECTOR(inv2);
        sum1 += kvy0 * inv0;    PRINT_VECTOR(sum1);
        sum1 += kvy1 * inv1;    PRINT_VECTOR(sum1);
        sum1 += kvy2 * inv2;    PRINT_VECTOR(sum1);
        
        inv0 = _mm_load_ps(&input[(y + 3) * s + 4]);      PRINT_VECTOR(inv0);
        inv1 = _mm_load_ps(&input[(y + 4) * s + 4]);      PRINT_VECTOR(inv1);
        inv2 = _mm_load_ps(&input[(y + 5) * s + 4]);      PRINT_VECTOR(inv2);
        sum1 += kvy3 * inv0;    PRINT_VECTOR(sum1);
        sum1 += kvy4 * inv1;    PRINT_VECTOR(sum1);
        sum1 += kvy5 * inv2;    PRINT_VECTOR(sum1);
        
        inv0 = _mm_load_ps(&input[(y + 6) * s + 4]);      PRINT_VECTOR(inv0);
        inv1 = _mm_load_ps(&input[(y + 7) * s + 4]);      PRINT_VECTOR(inv1);
        inv2 = _mm_load_ps(&input[(y + 8) * s + 4]);      PRINT_VECTOR(inv2);
        sum1 += kvy6 * inv0;    PRINT_VECTOR(sum1);
        sum1 += kvy7 * inv1;    PRINT_VECTOR(sum1);
        sum1 += kvy8 * inv2;    PRINT_VECTOR(sum1);
        
        for (int x = 0; x < stopX; x += 4) {
            
            PRINT_LINE(); 
            PRINT(x); 
            PRINT_VECTOR(sum0)
            PRINT_VECTOR(sum1)
            
            inv0 = _mm_load_ps(&input[y * s + x + 8]);            PRINT_VECTOR(inv0);
            inv1 = _mm_load_ps(&input[(y + 1) * s + x + 8]);      PRINT_VECTOR(inv1);
            inv2 = _mm_load_ps(&input[(y + 2) * s + x + 8]);      PRINT_VECTOR(inv2);
            sum2 += kvy0 * inv0;                                                 PRINT_VECTOR(sum2);
            sum2 += kvy1 * inv1;                                                 PRINT_VECTOR(sum2);
            sum2 += kvy2 * inv2;                                                 PRINT_VECTOR(sum2);
            
            inv0 = _mm_load_ps(&input[(y + 3) * s + x + 8]);      PRINT_VECTOR(inv0);
            inv1 = _mm_load_ps(&input[(y + 4) * s + x + 8]);      PRINT_VECTOR(inv1);
            inv2 = _mm_load_ps(&input[(y + 5) * s + x + 8]);      PRINT_VECTOR(inv2);
            sum2 += kvy3 * inv0;                                                 PRINT_VECTOR(sum2);
            sum2 += kvy4 * inv1;                                                 PRINT_VECTOR(sum2);
            sum2 += kvy5 * inv2;                                                 PRINT_VECTOR(sum2);
            
            inv0 = _mm_load_ps(&input[(y + 6) * s + x + 8]);      PRINT_VECTOR(inv0);
            inv1 = _mm_load_ps(&input[(y + 7) * s + x + 8]);      PRINT_VECTOR(inv1);
            inv2 = _mm_load_ps(&input[(y + 8) * s + x + 8]);      PRINT_VECTOR(inv2);
            sum2 += kvy6 * inv0;    PRINT_VECTOR(sum2);
            sum2 += kvy7 * inv1;    PRINT_VECTOR(sum2);
            sum2 += kvy8 * inv2;    PRINT_VECTOR(sum2);
            
            register __m128 kvx0 = _mm_load_ps(kernelX);                        PRINT_VECTOR(kvx0);
            /*kvx1*/ inv1 = _mm_load_ps(kernelX + 4);                           PRINT_VECTOR(inv1);
            /*kvx2*/ inv2 = _mm_load_ps(kernelX + 8);                           PRINT_VECTOR(inv2);
              
            // ---------   ---------   ---------
            // |0|1|2|3|   |4|5|6|7|   |8|-|-|-|
            // ---------   ---------   ---------
            inv0 = _mm_dp_ps(sum0, kvx0, 241) + 
                   _mm_dp_ps(sum1,  /*kvx1*/ inv1 , 241) + 
                   _mm_dp_ps(sum2,  /*kvx2*/ inv2 , 17);          PRINT_VECTOR(inv0);
                   
            // ---------   ---------   ---------
            // |3|0|1|2|   |3|4|5|6|   |7|8|-|-|
            // ---------   ---------   ---------
            ROTATE_RIGHT( /*kvx2*/ inv2 );
            ROTATE_RIGHT_BLEND(/*kvx1*/ inv1, /*kvx2*/ inv2);
            ROTATE_RIGHT_BLEND(kvx0,  /*kvx1*/ inv1 );
            inv0 += _mm_dp_ps(sum0, kvx0, 226) + 
                   _mm_dp_ps(sum1,  /*kvx1*/ inv1 , 242) + 
                   _mm_dp_ps(sum2,  /*kvx2*/ inv2 , 50);          PRINT_VECTOR(inv0);
            
            PRINT_LABEL("Terceiro"); 
            // ---------   ---------   ---------
            // |2|3|0|1|   |2|3|4|5|   |6|7|8|-|
            // ---------   ---------   ---------
            ROTATE_RIGHT( /*kvx2*/ inv2 );
            ROTATE_RIGHT_BLEND(/*kvx1*/ inv1, /*kvx2*/ inv2);
            ROTATE_RIGHT_BLEND(kvx0,  /*kvx1*/ inv1 );
            inv0 += _mm_dp_ps(sum0, kvx0, 196) + 
                   _mm_dp_ps(sum1,  /*kvx1*/ inv1 , 244) + 
                   _mm_dp_ps(sum2,  /*kvx2*/ inv2 , 116);          PRINT_VECTOR(inv0);
            
            PRINT_LABEL("Quarto"); 
            PRINT_VECTOR(sum0);
            PRINT_VECTOR(sum1);
            PRINT_VECTOR(sum2);
            
            // ---------   ---------   ---------
            // |1|2|3|0|   |1|2|3|4|   |5|6|7|8|
            // ---------   ---------   ---------
            ROTATE_RIGHT( /*kvx2*/ inv2 );
            ROTATE_RIGHT_BLEND(/*kvx1*/ inv1, /*kvx2*/ inv2);
            ROTATE_RIGHT_BLEND(kvx0,  /*kvx1*/ inv1 );
            inv0 += _mm_dp_ps(sum0, kvx0, 136) + 
                   _mm_dp_ps(sum1,  /*kvx1*/ inv1 , 248) + 
                   _mm_dp_ps(sum2,  /*kvx2*/ inv2 , 120);          PRINT_VECTOR(inv0);
            
            PRINT_LABEL("Rotate"); 
            
            PRINT_LABEL("sum"); 
            PRINT((y + hk) * s + (x + hk));
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk)], inv0);                 PRINT_VECTOR(inv0);
            
            PRINT(output[(y + hk) * s + (x + hk)]);
            PRINT(output[(y + hk) * s + (x + hk) + 1]);
            PRINT(output[(y + hk) * s + (x + hk) + 2]);
            PRINT(output[(y + hk) * s + (x + hk) + 3]);            
            
            sum0 = sum1;
            sum1 = sum2;
            sum2 = _mm_setzero_ps();
        }
     
    }
    
    processBoundariesS2D (s, w, h, 
                         kw, 
                         input, output, kernelX, kernelY);        
    
}



void scGaussian5SSE (const int s, const int w, const int h, 
                     const float* input, float* output, 
                     const float* kernel) {

    const int kw = 5;
    const int hk = kw / 2;
    
    int stopX   = w;
    int startY  = 0;
    int stopY   = h - 2 * (kw / 2);

      
    #pragma omp parallel for shared (input, output) 
    for (int y = startY; y < stopY; ++y) {
     
        //TODO For gaussian we only need kw / 2 + 1 kernel vectors since guassian function repeats 
        register __m128 kv = _mm_load_ps(kernel);                          PRINT_VECTOR(kv);
        const register __m128 kvy0 = _mm_shuffle_ps(kv, kv, 0);             PRINT_VECTOR(kvy0);
        const register __m128 kvy1 = _mm_shuffle_ps(kv, kv, 85);            PRINT_VECTOR(kvy1);
        const register __m128 kvy2 = _mm_shuffle_ps(kv, kv, 170);           PRINT_VECTOR(kvy2);

        register __m128 kvx0 = _mm_load_ps(kernel);                        PRINT_VECTOR(kvx0);
        register __m128 kvx1 = _mm_load_ps(kernel + 4);                    PRINT_VECTOR(kvx1);
         
        __m128 sum0, sum1;
        sum0 = sum1 = _mm_setzero_ps();
        
        PRINT_LABEL("inv"); 
        
        register __m128 inv0 = _mm_load_ps(&input[y * s]);            PRINT_VECTOR(inv0);
        register __m128 inv1 = _mm_load_ps(&input[(y + 1) * s]);      PRINT_VECTOR(inv1);
        register __m128 inv2 = _mm_load_ps(&input[(y + 2) * s]);      PRINT_VECTOR(inv2);
        register __m128 inv3 = _mm_load_ps(&input[(y + 3) * s]);      PRINT_VECTOR(inv3);
        register __m128 inv4 = _mm_load_ps(&input[(y + 4) * s]);      PRINT_VECTOR(inv4);
        
        PRINT(y); 
                    
        sum0 += kvy0 * inv0;    PRINT_VECTOR(sum0);
        sum0 += kvy1 * inv1;    PRINT_VECTOR(sum0);
        sum0 += kvy2 * inv2;    PRINT_VECTOR(sum0);
        sum0 += kvy1 * inv3;    PRINT_VECTOR(sum0);
        sum0 += kvy0 * inv4;    PRINT_VECTOR(sum0);
        
        for (int x = 0; x < stopX; x += 4) {
            
            PRINT_LINE(); 
            PRINT(x); 
            PRINT_VECTOR(sum0)
            PRINT_VECTOR(sum1)
            
            inv0 = _mm_load_ps(&input[y * s + x + 4]);            PRINT_VECTOR(inv0);
            inv1 = _mm_load_ps(&input[(y + 1) * s + x + 4]);      PRINT_VECTOR(inv1);
            inv2 = _mm_load_ps(&input[(y + 2) * s + x + 4]);      PRINT_VECTOR(inv2);
            inv3 = _mm_load_ps(&input[(y + 3) * s + x + 4]);      PRINT_VECTOR(inv3);
            inv4 = _mm_load_ps(&input[(y + 4) * s + x + 4]);      PRINT_VECTOR(inv4);
            
            sum1 += kvy0 * inv0;    PRINT_VECTOR(sum1);
            sum1 += kvy1 * inv1;    PRINT_VECTOR(sum1);
            sum1 += kvy2 * inv2;    PRINT_VECTOR(sum1);
            sum1 += kvy1 * inv3;    PRINT_VECTOR(sum1);
            sum1 += kvy0 * inv4;    PRINT_VECTOR(sum1);
            
            //TODO parei aqui, fazer para no x agora.
            
            inv0 = _mm_dp_ps(sum0, kvx0, 241) + _mm_dp_ps(sum1, kvx1, 17);          PRINT_VECTOR(inv0);
            
            ROTATE_RIGHT(kvx1);                                                     
            ROTATE_RIGHT_BLEND(kvx0, kvx1);                                         
            inv0 += _mm_dp_ps(sum0, kvx0, 226) + _mm_dp_ps(sum1, kvx1, 50);         PRINT_VECTOR(inv0);
            
            ROTATE_RIGHT(kvx1);                                                     
            ROTATE_RIGHT_BLEND(kvx0, kvx1);                                         
            inv0 += _mm_dp_ps(sum0, kvx0, 196) + _mm_dp_ps(sum1, kvx1, 116);         PRINT_VECTOR(inv0);
            
            ROTATE_RIGHT(kvx1);                                                     
            ROTATE_RIGHT_BLEND(kvx0, kvx1);                                         
            inv0 += _mm_dp_ps(sum0, kvx0, 136) + _mm_dp_ps(sum1, kvx1, 248);         PRINT_VECTOR(inv0);
            
            ROTATE_RIGHT(kvx0);
            ROTATE_RIGHT(kvx1);
            
            PRINT_LABEL("sum"); 
            PRINT((y + hk) * s + (x + hk));
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk)], inv0);                 PRINT_VECTOR(inv0);
            PRINT(output[(y + hk) * s + (x + hk)]);
            PRINT(output[(y + hk) * s + (x + hk) + 1]);
            PRINT(output[(y + hk) * s + (x + hk) + 2]);
            PRINT(output[(y + hk) * s + (x + hk) + 3]);            
            
            sum0 = sum1;
            sum1 = _mm_setzero_ps();
        }
     
    }
    processBoundariesS2D (s, w, h, 
                         kw, 
                         input, output, kernel, kernel);        
    
}



void scGaussian7SSE (const int s, const int w, const int h, 
                     const float* input, float* output, const float* kernel) {

    const int kw = 7;
    const int hk = kw / 2;
    
    int stopX   = w;
    int startY  = 0;
    int stopY   = h - 2 * (kw / 2);

      
    #pragma omp parallel for shared (input, output) 
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
        inv0 = _mm_load_ps(&input[y * s]);                            PRINT_VECTOR(inv0);
        register __m128 inv1 = _mm_load_ps(&input[(y + 1) * s]);      PRINT_VECTOR(inv1);
        register __m128 inv2 = _mm_load_ps(&input[(y + 2) * s]);      PRINT_VECTOR(inv2);
        sum0 += kvy0 * inv0;    PRINT_VECTOR(sum0);
        sum0 += kvy1 * inv1;    PRINT_VECTOR(sum0);
        sum0 += kvy2 * inv2;    PRINT_VECTOR(sum0);
        
        inv0 = _mm_load_ps(&input[(y + 3) * s]);      PRINT_VECTOR(inv0);
        inv1 = _mm_load_ps(&input[(y + 4) * s]);      PRINT_VECTOR(inv1);
        inv2 = _mm_load_ps(&input[(y + 5) * s]);      PRINT_VECTOR(inv2);
        sum0 += kvy3 * inv0;    PRINT_VECTOR(sum0);
        sum0 += kvy2 * inv1;    PRINT_VECTOR(sum0);
        sum0 += kvy1 * inv2;    PRINT_VECTOR(sum0);
        
        inv0 = _mm_load_ps(&input[(y + 6) * s]);      PRINT_VECTOR(inv0);
        sum0 += kvy0 * inv0;    PRINT_VECTOR(sum0);
        
        //x + 4
        inv0 = _mm_load_ps(&input[y * s + 4]);            PRINT_VECTOR(inv0);
        inv1 = _mm_load_ps(&input[(y + 1) * s + 4]);      PRINT_VECTOR(inv1);
        inv2 = _mm_load_ps(&input[(y + 2) * s + 4]);      PRINT_VECTOR(inv2);
        sum1 += kvy0 * inv0;    PRINT_VECTOR(sum1);
        sum1 += kvy1 * inv1;    PRINT_VECTOR(sum1);
        sum1 += kvy2 * inv2;    PRINT_VECTOR(sum1);
        
        inv0 = _mm_load_ps(&input[(y + 3) * s + 4]);      PRINT_VECTOR(inv0);
        inv1 = _mm_load_ps(&input[(y + 4) * s + 4]);      PRINT_VECTOR(inv1);
        inv2 = _mm_load_ps(&input[(y + 5) * s + 4]);      PRINT_VECTOR(inv2);
        sum1 += kvy3 * inv0;    PRINT_VECTOR(sum1);
        sum1 += kvy2 * inv1;    PRINT_VECTOR(sum1);
        sum1 += kvy1 * inv2;    PRINT_VECTOR(sum1);
        
        inv0 = _mm_load_ps(&input[(y + 6) * s + 4]);      PRINT_VECTOR(inv0);
        sum1 += kvy0 * inv0;    PRINT_VECTOR(sum1);
        
        for (int x = 0; x < stopX; x += 4) {
            
            PRINT_LINE(); 
            PRINT(x); 
            PRINT_VECTOR(sum0)
            PRINT_VECTOR(sum1)
            
            inv0 = _mm_load_ps(&input[y * s + x + 8]);            PRINT_VECTOR(inv0);
            inv1 = _mm_load_ps(&input[(y + 1) * s + x + 8]);      PRINT_VECTOR(inv1);
            inv2 = _mm_load_ps(&input[(y + 2) * s + x + 8]);      PRINT_VECTOR(inv2);
            sum2 += kvy0 * inv0;                                                 PRINT_VECTOR(sum2);
            sum2 += kvy1 * inv1;                                                 PRINT_VECTOR(sum2);
            sum2 += kvy2 * inv2;                                                 PRINT_VECTOR(sum2);
            
            inv0 = _mm_load_ps(&input[(y + 3) * s + x + 8]);      PRINT_VECTOR(inv0);
            inv1 = _mm_load_ps(&input[(y + 4) * s + x + 8]);      PRINT_VECTOR(inv1);
            inv2 = _mm_load_ps(&input[(y + 5) * s + x + 8]);      PRINT_VECTOR(inv2);
            sum2 += kvy3 * inv0;                                                 PRINT_VECTOR(sum2);
            sum2 += kvy2 * inv1;                                                 PRINT_VECTOR(sum2);
            sum2 += kvy1 * inv2;                                                 PRINT_VECTOR(sum2);
            
            inv0 = _mm_load_ps(&input[(y + 6) * s + x + 8]);      PRINT_VECTOR(inv0);
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
            PRINT((y + hk) * s + (x + hk));
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk)], inv0);                 PRINT_VECTOR(inv0);
            PRINT(output[(y + hk) * s + (x + hk)]);
            PRINT(output[(y + hk) * s + (x + hk) + 1]);
            PRINT(output[(y + hk) * s + (x + hk) + 2]);
            PRINT(output[(y + hk) * s + (x + hk) + 3]);            
            
            sum0 = sum1;
            sum1 = sum2;
            sum2 = _mm_setzero_ps();
        }
    }
    
    processBoundariesS2D (s, w, h, 
                         kw, 
                         input, output, kernel, kernel);        
    
    //printImage(w, h, s, output);
    
}



void scGaussian9SSE (const int s, const int w, const int h, 
                     const float* input, float* output, 
                     const float* kernel) {

    const int kw = 9;
    const int hk = kw / 2;
    
    int stopX   = w;
    int startY  = 0;
    int stopY   = h - 2 * (kw / 2);

      
    #pragma omp parallel for shared (input, output) 
    for (int y = startY; y < stopY; ++y) {
     
        //Load kernel lines. Make it const, so we don't have to load every time
        register __m128 inv0 = _mm_load_ps(kernel);                        PRINT_VECTOR(inv0);
        const register __m128 kvy0 = _mm_shuffle_ps(inv0, inv0, 0);         PRINT_VECTOR(kvy0);
        const register __m128 kvy1 = _mm_shuffle_ps(inv0, inv0, 85);        PRINT_VECTOR(kvy1);
        const register __m128 kvy2 = _mm_shuffle_ps(inv0, inv0, 170);       PRINT_VECTOR(kvy2);
        const register __m128 kvy3 = _mm_shuffle_ps(inv0, inv0, 255);       PRINT_VECTOR(kvy3);

        inv0 = _mm_load_ps(kernel + 4);                                    PRINT_VECTOR(inv0);
        const register __m128 kvy4 = _mm_shuffle_ps(inv0, inv0, 0);         PRINT_VECTOR(kvy4); 


        register __m128 kvx0 = _mm_load_ps(kernel);                        PRINT_VECTOR(kvx0);
        register __m128 kvx1 = _mm_load_ps(kernel + 4);                    PRINT_VECTOR(kvx1);
        register __m128 kvx2 = _mm_load_ps(kernel + 8);                    PRINT_VECTOR(kvx2);


         //vectors that will hold y dot product results
        __m128 sum0, sum1, sum2;
        sum0 = sum1 = sum2 = _mm_setzero_ps();
        
        PRINT_LABEL("inv"); 
        PRINT(y); 
        
        //calculate y dot products
        
        //x
        inv0 = _mm_load_ps(&input[y * s]);                            PRINT_VECTOR(inv0);
        register __m128 inv1 = _mm_load_ps(&input[(y + 1) * s]);      PRINT_VECTOR(inv1);
        register __m128 inv2 = _mm_load_ps(&input[(y + 2) * s]);      PRINT_VECTOR(inv2);
        sum0 += kvy0 * inv0;    PRINT_VECTOR(sum0);
        sum0 += kvy1 * inv1;    PRINT_VECTOR(sum0);
        sum0 += kvy2 * inv2;    PRINT_VECTOR(sum0);
        
        inv0 = _mm_load_ps(&input[(y + 3) * s]);      PRINT_VECTOR(inv0);
        inv1 = _mm_load_ps(&input[(y + 4) * s]);      PRINT_VECTOR(inv1);
        inv2 = _mm_load_ps(&input[(y + 5) * s]);      PRINT_VECTOR(inv2);
        sum0 += kvy3 * inv0;    PRINT_VECTOR(sum0);
        sum0 += kvy4 * inv1;    PRINT_VECTOR(sum0);
        sum0 += kvy3 * inv2;    PRINT_VECTOR(sum0);
        
        inv0 = _mm_load_ps(&input[(y + 6) * s]);      PRINT_VECTOR(inv0);
        inv1 = _mm_load_ps(&input[(y + 7) * s]);      PRINT_VECTOR(inv1);
        inv2 = _mm_load_ps(&input[(y + 8) * s]);      PRINT_VECTOR(inv2);
        sum0 += kvy2 * inv0;    PRINT_VECTOR(sum0);
        sum0 += kvy1 * inv1;    PRINT_VECTOR(sum0);
        sum0 += kvy0 * inv2;    PRINT_VECTOR(sum0);
        
        //x + 4
        inv0 = _mm_load_ps(&input[y * s + 4]);            PRINT_VECTOR(inv0);
        inv1 = _mm_load_ps(&input[(y + 1) * s + 4]);      PRINT_VECTOR(inv1);
        inv2 = _mm_load_ps(&input[(y + 2) * s + 4]);      PRINT_VECTOR(inv2);
        sum1 += kvy0 * inv0;    PRINT_VECTOR(sum1);
        sum1 += kvy1 * inv1;    PRINT_VECTOR(sum1);
        sum1 += kvy2 * inv2;    PRINT_VECTOR(sum1);
        
        inv0 = _mm_load_ps(&input[(y + 3) * s + 4]);      PRINT_VECTOR(inv0);
        inv1 = _mm_load_ps(&input[(y + 4) * s + 4]);      PRINT_VECTOR(inv1);
        inv2 = _mm_load_ps(&input[(y + 5) * s + 4]);      PRINT_VECTOR(inv2);
        sum1 += kvy3 * inv0;    PRINT_VECTOR(sum1);
        sum1 += kvy4 * inv1;    PRINT_VECTOR(sum1);
        sum1 += kvy3 * inv2;    PRINT_VECTOR(sum1);
        
        inv0 = _mm_load_ps(&input[(y + 6) * s + 4]);      PRINT_VECTOR(inv0);
        inv1 = _mm_load_ps(&input[(y + 7) * s + 4]);      PRINT_VECTOR(inv1);
        inv2 = _mm_load_ps(&input[(y + 8) * s + 4]);      PRINT_VECTOR(inv2);
        sum1 += kvy2 * inv0;    PRINT_VECTOR(sum1);
        sum1 += kvy1 * inv1;    PRINT_VECTOR(sum1);
        sum1 += kvy0 * inv2;    PRINT_VECTOR(sum1);
        
        for (int x = 0; x < stopX; x += 4) {
            
            PRINT_LINE(); 
            PRINT(x); 
            PRINT_VECTOR(sum0)
            PRINT_VECTOR(sum1)
            
            inv0 = _mm_load_ps(&input[y * s + x + 8]);            PRINT_VECTOR(inv0);
            inv1 = _mm_load_ps(&input[(y + 1) * s + x + 8]);      PRINT_VECTOR(inv1);
            inv2 = _mm_load_ps(&input[(y + 2) * s + x + 8]);      PRINT_VECTOR(inv2);
            sum2 += kvy0 * inv0;                                                 PRINT_VECTOR(sum2);
            sum2 += kvy1 * inv1;                                                 PRINT_VECTOR(sum2);
            sum2 += kvy2 * inv2;                                                 PRINT_VECTOR(sum2);
            
            inv0 = _mm_load_ps(&input[(y + 3) * s + x + 8]);      PRINT_VECTOR(inv0);
            inv1 = _mm_load_ps(&input[(y + 4) * s + x + 8]);      PRINT_VECTOR(inv1);
            inv2 = _mm_load_ps(&input[(y + 5) * s + x + 8]);      PRINT_VECTOR(inv2);
            sum2 += kvy3 * inv0;                                                 PRINT_VECTOR(sum2);
            sum2 += kvy4 * inv1;                                                 PRINT_VECTOR(sum2);
            sum2 += kvy3 * inv2;                                                 PRINT_VECTOR(sum2);
            
            inv0 = _mm_load_ps(&input[(y + 6) * s + x + 8]);      PRINT_VECTOR(inv0);
            inv1 = _mm_load_ps(&input[(y + 7) * s + x + 8]);      PRINT_VECTOR(inv1);
            inv2 = _mm_load_ps(&input[(y + 8) * s + x + 8]);      PRINT_VECTOR(inv2);
            sum2 += kvy2 * inv0;    PRINT_VECTOR(sum2);
            sum2 += kvy1 * inv1;    PRINT_VECTOR(sum2);
            sum2 += kvy0 * inv2;    PRINT_VECTOR(sum2);
            
            // ---------   ---------   ---------
            // |0|1|2|3|   |4|5|6|7|   |8|-|-|-|
            // ---------   ---------   ---------
            inv0 = _mm_dp_ps(sum0, kvx0, 241) + 
                   _mm_dp_ps(sum1, kvx1 , 241) + 
                   _mm_dp_ps(sum2, kvx2 , 17);          PRINT_VECTOR(inv0);
                   
            // ---------   ---------   ---------
            // |3|0|1|2|   |3|4|5|6|   |7|8|-|-|
            // ---------   ---------   ---------
            ROTATE_RIGHT( kvx2 );
            ROTATE_RIGHT_BLEND(kvx1, kvx2);
            ROTATE_RIGHT_BLEND(kvx0,  kvx1);
            inv0 += _mm_dp_ps(sum0, kvx0, 226) + 
                   _mm_dp_ps(sum1, kvx1, 242) + 
                   _mm_dp_ps(sum2, kvx2 , 50);          PRINT_VECTOR(inv0);
            
            PRINT_LABEL("Terceiro"); 
            // ---------   ---------   ---------
            // |2|3|0|1|   |2|3|4|5|   |6|7|8|-|
            // ---------   ---------   ---------
            ROTATE_RIGHT( kvx2 );
            ROTATE_RIGHT_BLEND(kvx1, kvx2);
            ROTATE_RIGHT_BLEND(kvx0,  kvx1);
            inv0 += _mm_dp_ps(sum0, kvx0, 196) + 
                   _mm_dp_ps(sum1, kvx1, 244) + 
                   _mm_dp_ps(sum2, kvx2 , 116);          PRINT_VECTOR(inv0);
            
            PRINT_LABEL("Quarto"); 
            PRINT_VECTOR(sum0);
            PRINT_VECTOR(sum1);
            PRINT_VECTOR(sum2);
            
            // ---------   ---------   ---------
            // |1|2|3|0|   |1|2|3|4|   |5|6|7|8|
            // ---------   ---------   ---------
            ROTATE_RIGHT( kvx2 );
            ROTATE_RIGHT_BLEND( kvx1, kvx2);
            ROTATE_RIGHT_BLEND(kvx0, kvx1);
            inv0 += _mm_dp_ps(sum0, kvx0, 136) + 
                   _mm_dp_ps(sum1, kvx1, 248) + 
                   _mm_dp_ps(sum2, kvx2, 120);          PRINT_VECTOR(inv0);
            
            PRINT_LABEL("Rotate"); 
            PRINT_VECTOR(kvx0);
            PRINT_VECTOR(kvx1);
            PRINT_VECTOR(kvx2);

            ROTATE_RIGHT( kvx0 );
            kvx1 = _mm_blend_ps(kvx1, kvx2, 7);             PRINT_VECTOR(kvx1);
            ROTATE_RIGHT( kvx1 );
            kvx2 = _mm_blend_ps(kvx2, _mm_setzero_ps(), 7); PRINT_VECTOR(kvx2);
            ROTATE_RIGHT( kvx2 );
            
            
            PRINT_LABEL("sum"); 
            PRINT((y + hk) * s + (x + hk));
            _mm_storeu_ps(&output[(y + hk) * s + (x + hk)], inv0);                 PRINT_VECTOR(inv0);
            PRINT(output[(y + hk) * s + (x + hk)]);
            PRINT(output[(y + hk) * s + (x + hk) + 1]);
            PRINT(output[(y + hk) * s + (x + hk) + 2]);
            PRINT(output[(y + hk) * s + (x + hk) + 3]);            
            
            sum0 = sum1;
            sum1 = sum2;
            sum2 = _mm_setzero_ps();
        }
     
    }
    processBoundariesS2D (s, w, h, 
                         kw, 
                         input, output, kernel, kernel);        
    
}


//process boundaries 2d
void processTopLeftCorner2D(const int s, const int w, const int h, 
                          const int ks, const int kw, const float* kernel, const float* input, float* output) {
    int hk = kw / 2;

    for(int y = 0; y < hk; ++y) {
        for(int x = 0; x < hk; ++x) {

            float value = 0;

            
            int left = hk - x;
            int top = hk - y;
          
            //traverse corner ok
            float v = input[0];
            for (int i = 0; i < top + 1; ++i) {
                for (int j = 0; j < left + 1; ++j) {
                    value += kernel[i * ks + j] * v;
                }
            }
            
            //traverse top ok
            for (int i = 0; i < top + 1; ++i) {
                for (int j = left + 1, jj = 1; j < kw; ++j, ++jj) {
                    value += kernel[i * ks + j] * input[jj];
                }
            }
            
            //traverse left ok
            for (int i = top + 1, ii = s; i < kw; ++i, ii += s) {
                for (int j = 0; j < left + 1; ++j) {
                    value += kernel[i * ks + j] * input[ii];
                }
            }
            
            //traverse middle ok
            for (int i = top + 1, ii = 1; i < kw; ++i, ++ii) {
                for (int j = left + 1, jj = 1; j < kw; ++j, ++jj) {
                    value += kernel[i * ks + j] * input[ii * s + jj];
                }
            }            
            
            output[y * s + x] = value;
                
        }                 
    }                 
}

void processTopMiddle2D(const int s, const int w, const int h, 
                          const int ks, const int kw, const float* kernel, const float* input, float* output) {
    int hk = kw / 2;

    for(int y = 0; y < hk; ++y) {
        for(int x = hk; x < w - hk; ++x) {

            float value = 0;

            
            int top = hk - y;
          
            //traverse top ok
            for (int i = 0; i < top + 1; ++i) {
                for (int j = 0, jj = 0; j < kw; ++j, ++jj) {
                    value += kernel[i * ks + j] * input[(x - hk) + jj];
                }
            }
            
            //traverse middle
            for (int i = top + 1, ii = 1; i < kw; ++i, ++ii) {
                for (int j = 0, jj = 0; j < kw; ++j, ++jj) {
                    value += kernel[i * ks + j] * input[ii * s + (x - hk) + jj];
                }
            }            
            
            output[y * s + x] = value;
                                    
            
        }                 
    }                 
}

void processTopRightCorner2D(const int s, const int w, const int h, 
                          const int ks, const int kw, const float* kernel, const float* input, float* output) {
                           
    int hk = kw / 2;

    for(int y = 0; y < hk; ++y) {
        for(int x = w - 1; x > w - hk - 1; --x) {

            float value = 0;

            
            int top = hk - y;
            int right = hk - (w - x - 1);
          
            //traverse corner ok 
            float v = input[w - 1];
            for (int i = 0; i < top + 1; ++i) {
                for (int j = kw - 1; j >= kw - right - 1; --j) {
                    value += kernel[i * ks + j] * v;
                }
            }

            //traverse top ok 
            for (int i = 0; i < top + 1; ++i) {
                for (int j = kw - right - 2, jj = w - 2; j >= 0; --j, --jj) {
                    value += kernel[i * ks + j] * input[jj];
                }
            }
            
            //traverse right ok 
            for (int i = top + 1, ii = s + w - 1; i < kw; ++i, ii += s) {
                for (int j = kw - 1; j >=  kw - right - 1; --j) {
                    value += kernel[i * ks + j] * input[ii];
                }
            }
            
            //traverse middle ok
            for (int i = top + 1, ii = 1; i < kw; ++i, ++ii) {
                for (int j = kw - right - 2, jj = w - 2; j >= 0; --j, --jj) {
                    value += kernel[i * ks + j] * input[ii * s + jj];
                }
            }           
            
            output[y * s + x] = value;

            
        }                 
    }                 
}                           
                           

void processRightMiddle2D(const int s, const int w, const int h, 
                          const int ks, const int kw, const float* kernel, const float* input, float* output) {
    
    int hk = kw / 2;
    
    for(int x = w - 1; x > w - hk - 1; --x) {
        for(int y = hk; y < h - hk; ++y) {
    
            float value = 0;
    
    
            int right = hk - (w - x - 1);
          
            // traverse right ok
            for (int j = kw - 1; j >= kw - right - 1; --j) {
                for (int i = 0, ii = 0; i < kw; ++i, ++ii) {
                    value += kernel[i * ks + j] * input[((y - hk) + ii) * s + w - 1];
                }
            }
            
            
            // traverse middle ok
            for (int j = kw - right - 2, jj = 0; j >= 0; --j, ++jj) {
                for (int i = 0, ii = 0; i < kw; ++i, ++ii) {
                    value += kernel[i * ks + j] * input[(((y - hk) + ii) * s + w - 2) - jj];
                }
            }
            
            output[y * s + x] = value;
            
        }                 
    }                 
}

void processBottomRightCorner2D(const int s, const int w, const int h, 
                          const int ks, const int kw, const float* kernel, const float* input, float* output) {
                           
    int hk = kw / 2;
    
    
    for(int y = h - 1; y > h - hk - 1; --y) {
        for(int x = w - 1; x > w - hk - 1; --x) {
           
            float value = 0;
           
            int right = hk - (w - x - 1);
            int bottom = hk - (h - y - 1);
           
            //traverse corner ok
            float v = input[(s * (h - 1)) + w - 1];
            for (int i = kw - 1; i >= kw - bottom - 1; --i) {
                for (int j = kw - 1; j >= kw - right - 1; --j) {
                    value += kernel[i * ks + j] * v;
                }
            }          
            
            //traverse bottom ok
            for (int j = kw - right - 2, jj = 2; j >= 0; --j, ++jj) {
                int v = input[(s * (h - 1)) + w - jj];
                for (int i = kw - 1; i >= kw - bottom - 1; --i) {
                    value += kernel[i * ks + j] * v;
                }
            }
            
            //traverse right ok
            for (int i = kw - bottom - 2, ii = h - 2; i >= 0; --i, --ii) {
                int v = input[(ii * s + w) - 1];
                for (int j = kw - 1; j >= kw - right - 1; --j) {
                    value += kernel[i * ks + j] * v;
                }
            }
            
            //traverse middle ok
            for (int i = kw - bottom - 2, ii = h - 2; i >= 0; --i, --ii) {
                for (int j = kw - right - 2, jj = 2; j >= 0; --j, ++jj) {
                    value += kernel[i * ks + j] * input[ii * s + w - jj];
                }
            }                        
            output[y * s + x] = value;
            
        }                 
    }                 
}

void processBottomMiddle2D(const int s, const int w, const int h, 
                          const int ks, const int kw, const float* kernel, const float* input, float* output) {
                           
    int hk = kw / 2;
    
    for(int y = h - 1; y > h - hk - 1; --y) {
        for(int x = hk; x < w - hk; ++x) {
    
            float value = 0;
           
            int bottom = hk - (h - y - 1);
       
            //traverse bottom
            for (int i = kw - 1; i > kw - bottom - 1; --i) {
                int jj = ((h - 1) * s) + (x - hk);
                for (int j = 0; j < kw; ++j) {
                    value += kernel[i * ks + j] * input[j + jj];
                }
            }
            
            //traverse middle
            for (int i = kw - bottom - 1, ii = h - 1; i >= 0; --i, --ii) {
                int jj = (ii * s) + (x - hk);
                for (int j = 0; j < kw; ++j) {
                    value += kernel[i * ks + j] * input[j + jj];
                }
            }
            output[y * s + x] = value;
                                    
        }                 
    }                 
}

void processBottomLeftCorner2D(const int s, const int w, const int h, 
                          const int ks, const int kw, const float* kernel, const float* input, float* output) {

                           
    int hk = kw / 2;
    
    
    for(int y = h - 1; y > h - hk - 1; --y) {
        for(int x = 0; x < hk; ++x) {
    
            float value = 0;
    
            int left = hk - x;
            int bottom = hk - (h - y - 1);
           
            //traverse corner
            float v = input[s * (h - 1)];
            for (int i = kw - 1; i >= kw - bottom - 1; --i) {
                for (int j = 0; j < left + 1; ++j) {
                    value += kernel[i * ks + j] * v;
                }
            }          
            
            //traverse bottom
            for (int j = left + 1, jj = 1; j < kw; ++j, ++jj) {
                int v = input[(h - 1) * s + jj];
                for (int i = kw - 1; i >= kw - bottom - 1; --i) {
                    value += kernel[i * ks + j] * v;
                }
            }
            
            //traverse left
            for (int i = kw - bottom - 2, ii = h - 2; i >= 0; --i, --ii) {
                int v = input[ii * s];
                for (int j = 0; j < left + 1; ++j) {
                    value += kernel[i * ks + j] * v;
                }
            }
            
            for (int i = kw - bottom - 2, ii = h - 2; i >= 0; --i, --ii) {
                for (int j = left + 1, jj = 1; j < kw; ++j, ++jj) {
                    value += kernel[i * ks + j] * input[ii * s + jj];
                }
            }                        
            
            output[y * s + x] = value;
            
        }                 
    }                 
}

void processLeftMiddle2D(const int s, const int w, const int h, 
                          const int ks, const int kw, const float* kernel, const float* input, float* output) {
                           
    int hk = kw / 2;
    
    for(int x = 0; x < hk; ++x) {
        for(int y = hk; y < h - hk; ++y) {
    
            float value = 0;
           
            int left = hk - x;
          
       
            //traverse right
            for (int j = 0; j < left; ++j) {
                for (int i = 0; i < kw; ++i) {
                    value += kernel[i * ks + j] * input[((y - hk) + i) * s];
                }
            }
            
            //traverse middle
            for (int j = left, jj = 0; j < kw; ++j, ++jj) {
                for (int i = 0; i < kw; ++i) {
                    value += kernel[i * ks + j] * input[((y - hk) + i) * s + jj];
                }
            }
                     
            output[y * s + x] = value;
            
        }                 
    }                 
}


void processBoundaries2D(const int s, const int w, const int h, 
                        const int ks, int kw, 
                        const float* input, float* output, const float* kernel) {
    
        
    processTopLeftCorner2D (s, w, h, ks, kw, kernel, input, output);
    processTopMiddle2D (s, w, h, ks, kw, kernel, input, output);
    processTopRightCorner2D (s, w, h, ks, kw, kernel, input, output);
    processRightMiddle2D (s, w, h, ks, kw, kernel, input, output);
    processBottomRightCorner2D (s, w, h, ks, kw, kernel, input, output);
    processBottomMiddle2D (s, w, h, ks, kw, kernel, input, output);
    processBottomLeftCorner2D (s, w, h, ks, kw, kernel, input, output);
    processLeftMiddle2D (s, w, h, ks, kw, kernel, input, output);

     
}                         


inline void processBoundariesS2DTop(const int s, const int w, const int h, 
                                 const int kw, 
                                 const float* input, float* output, 
                                 const float* kernelX, const float* kernelY) {
    int hk = kw / 2;   
    int stopX = w - hk * 2;                      
    float* values = new float[w * hk];
    //top
    for (int y = 0; y < hk; ++y) {
     
        for (int c = 0; c < kw - 1; ++c) {
            const int idxIntmp = y * s + c;
            float sum = 0;
            for (int r = 0; r < kw; ++r) {
                float value = r < hk - y ? input[c] : input[idxIntmp + (r - hk) * s];
                sum += kernelY[r] * value;
            }
            values[y * w + c] = sum;
        }
     
        for (int x = 0; x < stopX; ++x) {
            float sum = 0;
            int idxIntmp = y * s + x + (kw -1);
            for (int r = 0; r < kw; ++r) {
                float value = r < hk - y ? input[x + (kw - 1)] : 
                                                   input[idxIntmp +  (r - hk) * s];
                sum += kernelY[r] * value;
            }
            values[y * w + x + (kw -1)] = sum;
        
            idxIntmp = y * w + x;
            sum = 0;
            for (int c = 0; c < kw; ++c) {
                sum += kernelX[c] * values[idxIntmp + c];
            }
            output[y * s + x + hk] = sum;         
        }
    }           
    
    delete [] values;
}



inline void processBoundariesS2DBottom(const int s, const int w, const int h, 
                                 const int kw, 
                                 const float* input, float* output, 
                                 const float* kernelX, const float* kernelY) {
    int hk = kw / 2;   
    int stopX = w - hk * 2;                      
    float* values = new float[w * hk];
    //top
    for (int y = h - 1; y >= h - hk; --y) {
     
        for (int c = 0; c < kw - 1; ++c) {
            const int idxIntmp = y * s + c;
            float sum = 0;
            for (int r = kw - 1; r >= 0; --r) {
                float value = 0;
                if (r >= hk + (h - y)) {
                 value = input[(h - 1) * s + c];
                }
                else {
                 value = input[idxIntmp - (hk - r) * s];
                }
                 
                sum += kernelY[r] * value;
            }
            values[(hk - (h - y)) * w + c] = sum;
        }
     
        for (int x = 0; x < stopX; ++x) {
            float sum = 0;
            int idxIntmp = y * s + x + (kw -1);
            for (int r = kw - 1; r >= 0; --r) {
                float value = r >= hk + (h - y) ? input[(h - 1) * s + x + (kw - 1)] : input[idxIntmp - (hk - r) * s];
                sum += kernelY[r] * value;
            }
            idxIntmp = (hk - (h - y)) * w + x;
            values[idxIntmp + (kw - 1)] = sum;
            sum = 0;
            for (int c = 0; c < kw; ++c) {
                sum += kernelX[c] * values[idxIntmp + c];
            }
            output[y * s + x + hk] = sum;         
        }
    }           
    
    delete [] values;
}

inline void processBoundariesS2DLeft(const int s, const int w, const int h, 
                                     const int kw, 
                                     const float* input, float* output, 
                                     const float* kernelX, const float* kernelY) {
    int hk = kw / 2;   
    int stopY = h;                      
    int vw = (kw - 1 + hk);
    float* values = new float[vw];
    //top
    for (int y = 0; y < stopY; ++y ) {
        float sum = 0; 
        for (int r = 0; r < kw; ++r) { 
            int idx = y + r < hk ? 0 : y + r - hk >= h ? h - 1 : (y + r) - hk;
            sum += kernelY[r] * input[idx * s];
        }
        
        for (int x = 0; x < hk; ++x) {
            values[x] = sum;  
        }
          
        for (int x = hk; x < vw; ++x) {
            sum = 0;
            for (int r = 0; r < kw; ++r) { 
                int idx = y + r < hk ? 0 : y + r - hk >= h ? h - 1 : (y + r) - hk;
                sum += kernelY[r] * input[idx * s + (x - hk)];
            }
            values[x] = sum;  
        }
        
        for (int x = 0; x < vw - 2 * hk; ++x) {
            sum = 0;
            for (int c = 0; c < kw; ++c) {
                sum += kernelX[c] * values[x + c];
            }
            output[y * s + x] = sum;
        }
    }   
    
    delete [] values;
}


inline void processBoundariesS2DRight(const int s, const int w, const int h, 
                                     const int kw, 
                                     const float* input, float* output, 
                                     const float* kernelX, const float* kernelY) {
    int hk = kw / 2;   
    int stopY = h;                      
    int vw = (kw - 1 + hk);
    float* values = new float[vw];
    //top
    for (int y = 0; y < stopY; ++y ) {
        float sum = 0; 
        for (int r = 0; r < kw; ++r) { 
            int idx = y + r < hk ? 0 : y + r - hk >= h ? h - 1 : (y + r) - hk;
            sum += kernelY[r] * input[idx * s + (w - 1)];
        }
        
        for (int x = vw - 1; x >= vw - hk; --x) {
            values[x] = sum;  
        }
          
        int i = 0;  
        for (int x = vw - hk - 1; x >= 0; --x) {
            sum = 0;
            for (int r = 0; r < kw; ++r) { 
                int idx = y + r < hk ? 0 : y + r - hk >= h ? h - 1 : (y + r) - hk;
                sum += kernelY[r] * input[idx * s + (w - 1) - i]; 
            }
            values[x] = sum;  
            ++i;
        }
        
        i = 0;  
        for (int x = 0; x < vw - 2 * hk; ++x) {
            sum = 0;
            for (int c = 0; c < kw; ++c) {
                sum += kernelX[c] * values[x + c];
            }
            output[y * s + (w - hk) + i] = sum;
            ++i;
        }
    }   
    
    delete [] values;
}
void processBoundariesS2D(const int s, const int w, const int h, 
                        const int kw, 
                        const float* input, float* output, const float* kernelX, const float* kernelY) {
                         
    processBoundariesS2DTop(s, w, h, 
                            kw, input, output, 
                            kernelX, kernelY);          
                            
    processBoundariesS2DBottom(s, w, h, 
                               kw, input, output, 
                               kernelX, kernelY);          
                                       
    processBoundariesS2DLeft(s, w, h, 
                             kw, input, output, 
                             kernelX, kernelY);          
                                       
    processBoundariesS2DRight(s, w, h, 
                             kw, input, output, 
                             kernelX, kernelY);          
                                       
}                         

//helper functions
void printImage(int w, int h, int stride, const float* out) {
    cout << endl;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            cout << left << setw(9) << out[y * stride + x]; 
        }
        cout << endl;
    }   
}

void printImageToFile(const string& file, int width, int height, int stride, const float* out) {
    ofstream myfile;
    myfile.open (file.c_str());
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            myfile << out[y * stride + x] << " "; 
        }
        myfile << "\n";
    }   
    myfile.close();
}


inline void printKernel2D(const int ks, const int kw, const float* kernel) {
    cout.precision(5); 
    
    cout << endl;
    for (int i = 0; i < kw; ++i) {
        for (int j = 0; j < kw; ++j) {
            cout << kernel[i * ks + j] << " ";             
        }
        cout << endl;
    }
        
    cout << endl;            
    cout.precision(3); 
}

//method for fast buffer clearing
void clear2DBuffer( float* buffer, 
                    int stride, 
                    int height )
{
    float* start  = buffer;
    float* end = &buffer[stride * height];
    float *p;
    
    //openmp is slower than sse only version 
    //#pragma omp parallel for num_threads(omp_get_num_procs()) shared (start, end) private(p)
    //for (p = start; p < end; ++p) {
    //     *p = 0;
    //}
    
    const __m128 value = _mm_setzero_ps();
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


float* allocateFloatAlignedBuffer (int width, int height)
{
    int stride = calculateAlignedStride (width, sizeof(float), ALIGMENT_BYTES);
    //aligned in 64 bytes for cache performance in Intel Core architecture
    //since 64 bytes is a multiple of 16 bytes, SSE alignment will be preserved  
#if defined(__GNUC__)  && !defined(__INTEL_COMPILER)  
    float *buffer __attribute__ ((aligned(ALIGMENT_BYTES))) = new float[stride * height];
#elif defined __INTEL_COMPILER  
    __declspec(align(ALIGMENT_BYTES)) float *buffer = new float[stride * height];
#endif
    
    return buffer;
}

int calculateAlignedStride (int width, int pixelSizeInBytes, int alignInBytes)
{
    if(width < alignInBytes) return alignInBytes;
    int widthInBytes = width * pixelSizeInBytes;
    return widthInBytes % alignInBytes == 0 ? width : 
        (widthInBytes + alignInBytes - (widthInBytes % alignInBytes)) / pixelSizeInBytes;
}

void copy2DBufferChunk(const float* inBuffer, float* outBuffer,
                       const int inX, const int inY, 
                       const int inStride, 
                       const int inWidth, const int inHeight,
                       const int outX, const int outY, 
                       const int outStride ){
   
    for (int y = inY, oY = outY; y < inHeight; ++y, ++oY) {
        for (int x = inX, oX = outX; x < inWidth; ++x, ++oX) {
            _mm_storeu_ps(&outBuffer[oY * outStride + oX], _mm_loadu_ps(&inBuffer[y * inStride + x]));
        }
    }
}
                       
void copy2DBoundaryChunk(const float* inBuffer, float* outBuffer,
                           const int outStride, const int outWidth, const int outHeight, 
                           const int replicateLeft, const int replicateTop,
                           const int replicateRight, const int replicateBottom,
                           const int inStride,  const int inWidth, const int inHeight) {
   
    int startIy = 0;
    int startIx = 0;
    
    if (!(replicateLeft && replicateRight)) {
        startIx = replicateLeft ? 0 : inWidth - outWidth + replicateRight;
    }
    
    if (!(replicateTop && replicateBottom)) {
        startIy = replicateTop ? 0 : inHeight - outHeight + replicateBottom;
    }
    
    int stopY = outHeight;
    int stopX = outWidth;
    for (int y = 0, iy = startIy; y < stopY; ++y, ++iy) {
        for (int x = 0, ix = startIx; x < stopX; ++x, ++ix) {
            if(y < replicateTop) {
                if (x < replicateLeft) {
                    outBuffer[y * outStride + x] = inBuffer[0];
                }
                else if (x >= outWidth - replicateRight) {
                    outBuffer[y * outStride + x] = inBuffer[inWidth - 1];
                }
                else {
                    outBuffer[y * outStride + x] = inBuffer[ix - replicateLeft];
                }
            }
            else if(y >= outHeight - replicateBottom) {
                if (x < replicateLeft) {
                    outBuffer[y * outStride + x] = inBuffer[(inHeight - 1) * inStride];
                }
                else if (x >= outWidth - replicateRight) {
                    outBuffer[y * outStride + x] = inBuffer[(inHeight - 1) * inStride + (inWidth - 1)];
                }
                else {
                    outBuffer[y * outStride + x] = inBuffer[(inHeight - 1) * inStride + (ix - replicateLeft)];
                }
            }
            else {
                if (x < replicateLeft) {
                    outBuffer[y * outStride + x] = inBuffer[(iy - replicateTop) * inStride];
                }
                else if (x >= outWidth - replicateRight) {
                    outBuffer[y * outStride + x] = inBuffer[(iy - replicateTop) * inStride + (inWidth - 1)];
                }
                else {
                    outBuffer[y * outStride + x] = inBuffer[(iy - replicateTop) * inStride + (ix - replicateLeft)];
                }
            }
        }
    }
}
                       
