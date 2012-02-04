// 
// opConvutionFilter.h
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

#ifndef _OPCONVUTIONFILTER_H
#define _OPCONVUTIONFILTER_H
 
#include <xmmintrin.h>  // SSE  (Required to use the __m128, and __m128d type)
#include <emmintrin.h>  // SSE2 (Required to use the __m128i type)
#include <pmmintrin.h>  // SSE3
 
#ifdef __SSE4_1__
#include <smmintrin.h>  
#endif 

#include <iostream>

//#define _USE_MATH_DEFINES
#define ALIGMENT_BYTES 64 
#define CACHE_LINE_SIZE 64


    #define DELETAR(x) \
        std::cout << #x << " { " << __builtin_ia32_vec_ext_v4sf(x, 0) << " " \
                     << __builtin_ia32_vec_ext_v4sf(x, 1) << " " \
                     << __builtin_ia32_vec_ext_v4sf(x, 2) << " " \
                     << __builtin_ia32_vec_ext_v4sf(x, 3) << " } " << std::endl;        

#define TRACE
#ifdef TRACE
    #define PRINT_TRACE(x) \
        cout << #x << ":\t" << x << endl;
    #define PRINT_VECTOR_TRACE(x) \
        cout << #x << " { " << __builtin_ia32_vec_ext_v4sf(x, 0) << " " \
                     << __builtin_ia32_vec_ext_v4sf(x, 1) << " " \
                     << __builtin_ia32_vec_ext_v4sf(x, 2) << " " \
                     << __builtin_ia32_vec_ext_v4sf(x, 3) << " } " << endl;        
#else
    #define PRINT_TRACE(x)
    #define PRINT_VECTOR_TRACE(x) 
#endif                      

//#define DEBUG
#ifndef DEBUG
 #define PRINT_IMAGE_TO_FILE(file, image, width, height, s)
    #define PRINT_IMAGE(width, height, s, kw, out)
    #define PRINT_LABEL(x)
    #define PRINT(x)
    #define PRINT_INLINE(x)
    #define PRINT_LINE()
    #define PRINT_VECTOR(x)
    #define PRINT_POSITION(x, y) 
#else 

 #define PRINT_IMAGE_TO_FILE(file, width, height, s, out) \
     printImageToFile(file, width, height, s, out);
    #define PRINT_IMAGE(width, height, s, kw, out) \
     printImage(width, height, s, kw, out)
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

 
  
   
//SSE instruction helpers    
                          
#define ROTATE_LEFT(vector) \
    vector = _mm_shuffle_ps(vector, vector, _MM_SHUFFLE(0,3,2,1)); PRINT_VECTOR(vector); 

#define ROTATE_RIGHT(vector) \
    vector = _mm_shuffle_ps(vector, vector, _MM_SHUFFLE(2, 1, 0, 3)); PRINT_VECTOR(vector); 
                     
#define ROTATE_RIGHT_BLEND(vector1, vector2) \
    vector1 = _mm_shuffle_ps(vector1, vector1, _MM_SHUFFLE(2, 1, 0, 3)); PRINT_VECTOR(vector1); \
    vector2 = _mm_move_ss(vector2, vector1); PRINT_VECTOR(vector2); 
    
    
//    vector1 = _mm_shuffle_ps(vector1, vector1, _MM_SHUFFLE(2, 1, 0, 3)); PRINT_VECTOR(vector1);
//    vector2 = _mm_blend_ps(vector2, vector1, 1); PRINT_VECTOR(vector2); 
    
    
#define BLEND_ROTATE_LEFT(vector0, vector1) \
    vector0 = _mm_move_ss(vector0, vector1); PRINT_VECTOR(vector0); \
    ROTATE_LEFT(vector0);
    
    
    //vector0 = _mm_blend_ps(vector0, vector1, 1); PRINT_VECTOR(vector0);
    //ROTATE_LEFT(vector0);


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


#if defined(__GNUC__)  && !defined(__INTEL_COMPILER)  

 
#elif defined __INTEL_COMPILER  

    inline void operator += (__m128& rhs, const __m128 B)
    {
     rhs = _mm_add_ps(rhs, B);
    }

    inline __m128 operator +(__m128& rhs, const __m128 B)
    {
     return _mm_add_ps(rhs, B);
    }    
    
    
#elif defined _MSC_VER  

#else  
    #error "nope"  
#endif  

#ifndef __SSE4_1__ 
  
    typedef union 
    {
        __m128  f;
        __m128i i;
    } ssp_m128;
    
    extern __inline __m128 __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_dp_ps (__m128 a, __m128 b, const int mask) {
        //http://sseplus.sourceforge.net/group__emulated___s_s_e3.html
        // Shift mask multiply moves 0,1,2,3 bits to left, becomes MSB
        const static __m128i mulShiftImm_0123 = _mm_set_epi32( 0x010000, 0x020000, 
                                                               0x040000, 0x080000 );  
        // Shift mask multiply moves 4,5,6,7 bits to left, becomes MSB 
        const static __m128i mulShiftImm_4567 = _mm_set_epi32( 0x100000, 0x200000, 
                                                               0x400000, 0x800000 );  
        
        // Begin mask preparation
        ssp_m128 mHi, mLo;
        mLo.i = _mm_set1_epi32( mask ); // Load the mask into register
        mLo.i = _mm_slli_si128( mLo.i, 3 );// Shift into reach of the 16 bit multiply
        
        mHi.i = _mm_mullo_epi16( mLo.i, mulShiftImm_0123 );  // Shift the bits
        mLo.i = _mm_mullo_epi16( mLo.i, mulShiftImm_4567 );  // Shift the bits
        
        // FFFFFFFF if bit set, 00000000 if not set
        mHi.i = _mm_cmplt_epi32( mHi.i, _mm_setzero_si128() );    
        // FFFFFFFF if bit set, 00000000 if not set
        mLo.i = _mm_cmplt_epi32( mLo.i, _mm_setzero_si128() );    
        // End mask preparation - Mask bits 0-3 in mLo, 4-7 in mHi
        a = _mm_and_ps( a, mHi.f ); // Clear input using the high bits of the mask
        a = _mm_mul_ps( a, b );
        a = _mm_hadd_ps( a, a ); // Horizontally add the 4 values
        a = _mm_hadd_ps( a, a ); // Horizontally add the 4 values
        a = _mm_and_ps( a, mLo.f );// Clear output using low bits of the mask
        return a;   
    }  

    extern __inline __m128i 
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    ssp_movmask_imm8_to_epi32_SSE2( int mask ) {
        __m128i screen;
        // Shift mask multiply moves all bits to left, becomes MSB
        const static __m128i mulShiftImm = 
          _mm_set_epi16( 0x1000, 0x0000, 0x2000, 0x0000, 
                         0x4000, 0x0000, 0x8000, 0x0000 ); 
        // Load the mask into register
        screen = _mm_set1_epi16 ( mask );   
        // Shift bits to MSB
        screen = _mm_mullo_epi16( screen, mulShiftImm );   
        // Shift bits to obtain all F's or all 0's
        screen = _mm_srai_epi32 ( screen, 31  );   
        return screen;  
    } 

    extern __inline __m128 
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_blend_ps( __m128  a, __m128  b, const int mask )// _mm_blend_ps [SSE4.1]
    {

        ssp_m128 screen, A, B;
        A.f = a;
        B.f = b;
        screen.i = ssp_movmask_imm8_to_epi32_SSE2 ( mask ); 
        B.i = _mm_and_si128   ( B.i, screen.i);  // clear a where mask = 0
        A.i = _mm_andnot_si128( screen.i, A.i ); // clear b where mask = 1
        screen.i = _mm_or_si128  ( A.i, B.i );   // a = a OR b        
        return screen.f;
    }  
    
#endif  

#ifdef __SSE4_1__ 
    extern __inline __m128 __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_dp241_ps (__m128 a, __m128 b) {
        return _mm_dp_ps(a, b, 241);      
    }
    extern __inline __m128 __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_dp242_ps (__m128 a, __m128 b) {
        return _mm_dp_ps(a, b, 242);      
    }
    extern __inline __m128 __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_dp244_ps (__m128 a, __m128 b) {
        return _mm_dp_ps(a, b, 244);      
    }
    extern __inline __m128 __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_dp248_ps (__m128 a, __m128 b) {
        return _mm_dp_ps(a, b, 248);      
    }
    extern __inline __m128 __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_dp17_ps (__m128 a, __m128 b) {
        return _mm_dp_ps(a, b, 17);      
    }
    extern __inline __m128 __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_dp226_ps (__m128 a, __m128 b) {
        return _mm_dp_ps(a, b, 226);      
    }
    extern __inline __m128 __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_dp50_ps (__m128 a, __m128 b) {
        return _mm_dp_ps(a, b, 50);      
    }
    extern __inline __m128 __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_dp196_ps (__m128 a, __m128 b) {
        return _mm_dp_ps(a, b, 196);      
    }
    extern __inline __m128 __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_dp116_ps (__m128 a, __m128 b) {
        return _mm_dp_ps(a, b, 116);      
    }
    extern __inline __m128 __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_dp136_ps (__m128 a, __m128 b) {
        return _mm_dp_ps(a, b, 136);      
    }
    extern __inline __m128 __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_dp120_ps (__m128 a, __m128 b) {
        return _mm_dp_ps(a, b, 120);      
    }
    extern __inline __m128 __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_dp114_ps (__m128 a, __m128 b) {
        return _mm_dp_ps(a, b, 114);      
    }
    extern __inline __m128 __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_dp113_ps (__m128 a, __m128 b) {
        return _mm_dp_ps(a, b, 113);      
    }
    extern __inline __m128 __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_dp18_ps (__m128 a, __m128 b) {
        return _mm_dp_ps(a, b, 18);      
    }
    extern __inline __m128 __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_dp20_ps (__m128 a, __m128 b) {
        return _mm_dp_ps(a, b, 20);      
    }
    extern __inline __m128 __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_dp24_ps (__m128 a, __m128 b) {
        return _mm_dp_ps(a, b, 24);      
    }
    extern __inline __m128 __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_dp56_ps (__m128 a, __m128 b) {
        return _mm_dp_ps(a, b, 56);      
    }
#else 
    extern __inline __m128 __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_dp241_ps (__m128 a, __m128 b) {
        __m128 omask = _mm_castsi128_ps(_mm_set_epi32(0x00000000, 0x00000000, 0x00000000, 0xFFFFFFFF));
        a = _mm_mul_ps( a, b );
        a = _mm_hadd_ps( a, a ); // Horizontally add the 4 values
        a = _mm_hadd_ps( a, a ); // Horizontally add the 4 values
        return _mm_and_ps( a, omask );
    }
    
    extern __inline __m128 __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_dp242_ps (__m128 a, __m128 b) {
        __m128 omask = _mm_castsi128_ps(_mm_set_epi32(0x0, 0x0, 0xFFFFFFFF, 0x0));
        a = _mm_mul_ps( a, b );
        a = _mm_hadd_ps( a, a ); // Horizontally add the 4 values
        a = _mm_hadd_ps( a, a ); // Horizontally add the 4 values
        return _mm_and_ps( a, omask );// Clear output using low bits of the mask
    }
    extern __inline __m128 __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_dp244_ps (__m128 a, __m128 b) {
        __m128 omask = _mm_castsi128_ps(_mm_set_epi32(0x0, 0xFFFFFFFF, 0x0, 0x0));
        a = _mm_mul_ps( a, b );
        a = _mm_hadd_ps( a, a ); // Horizontally add the 4 values
        a = _mm_hadd_ps( a, a ); // Horizontally add the 4 values
        return _mm_and_ps( a, omask );// Clear output using low bits of the mask
    }
    extern __inline __m128 __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_dp248_ps (__m128 a, __m128 b) {
        __m128 omask = _mm_castsi128_ps(_mm_set_epi32(0xFFFFFFFF, 0x0, 0x0, 0x0));
        a = _mm_mul_ps( a, b );
        a = _mm_hadd_ps( a, a ); // Horizontally add the 4 values
        a = _mm_hadd_ps( a, a ); // Horizontally add the 4 values
        return _mm_and_ps( a, omask );// Clear output using low bits of the mask
    }
    extern __inline __m128 __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_dp17_ps (__m128 a, __m128 b) {
        __m128 omask = _mm_castsi128_ps(_mm_set_epi32(0x0, 0x0, 0x0, 0xFFFFFFFF));
        a = _mm_mul_ps( a, b );
        return _mm_and_ps( a, omask );// Clear output using low bits of the mask
    }
    extern __inline __m128 __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_dp226_ps (__m128 a, __m128 b) {
        __m128 imask = _mm_castsi128_ps(_mm_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x0));
        __m128 omask = _mm_castsi128_ps(_mm_set_epi32(0x0, 0x0, 0xFFFFFFFF, 0x0));
        a = _mm_and_ps( a, imask );
        a = _mm_mul_ps( a, b );
        a = _mm_hadd_ps( a, a ); // Horizontally add the 4 values
        a = _mm_hadd_ps( a, a ); // Horizontally add the 4 values
        return _mm_and_ps( a, omask );// Clear output using low bits of the mask
    }
    extern __inline __m128 __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_dp50_ps (__m128 a, __m128 b) {
        __m128 imask = _mm_castsi128_ps(_mm_set_epi32(0x0, 0x0, 0xFFFFFFFF, 0xFFFFFFFF));
        __m128 omask = _mm_castsi128_ps(_mm_set_epi32(0x0, 0x0, 0xFFFFFFFF, 0x0));
        a = _mm_and_ps( a, imask );
        a = _mm_mul_ps( a, b );
        a = _mm_hadd_ps( a, a ); // Horizontally add the 4 values
        ROTATE_LEFT(a);
        return _mm_and_ps( a, omask );// Clear output using low bits of the mask
    }
    extern __inline __m128 __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_dp196_ps (__m128 a, __m128 b) {
        __m128 imask = _mm_castsi128_ps(_mm_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0x0, 0x0));
        __m128 omask = _mm_castsi128_ps(_mm_set_epi32(0x0, 0xFFFFFFFF, 0x0, 0x0));
        a = _mm_and_ps( a, imask );
        a = _mm_mul_ps( a, b );
        a = _mm_hadd_ps( a, a ); // Horizontally add the 4 values
        ROTATE_RIGHT(a); 
        return _mm_and_ps( a, omask );// Clear output using low bits of the mask
    }
    extern __inline __m128 __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_dp116_ps (__m128 a, __m128 b) {
        __m128 imask = _mm_castsi128_ps(_mm_set_epi32(0x0, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF));
        __m128 omask = _mm_castsi128_ps(_mm_set_epi32(0x0, 0xFFFFFFFF, 0x0, 0x0));
        a = _mm_and_ps( a, imask );
        a = _mm_mul_ps( a, b );
        a = _mm_hadd_ps( a, a ); // Horizontally add the 4 values
        a = _mm_hadd_ps( a, a ); // Horizontally add the 4 values
        return _mm_and_ps( a, omask );// Clear output using low bits of the mask
    }
    extern __inline __m128 __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_dp136_ps (__m128 a, __m128 b) {
        __m128 omask = _mm_castsi128_ps(_mm_set_epi32(0xFFFFFFFF, 0x0, 0x0, 0x0));
        a = _mm_mul_ps( a, b );
        return _mm_and_ps( a, omask );// Clear output using low bits of the mask
    }
    extern __inline __m128 __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_dp120_ps (__m128 a, __m128 b) {
        __m128 imask = _mm_castsi128_ps(_mm_set_epi32(0x0, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF));
        __m128 omask = _mm_castsi128_ps(_mm_set_epi32(0xFFFFFFFF, 0x0, 0x0, 0x0));
        a = _mm_and_ps( a, imask );
        a = _mm_mul_ps( a, b );
        a = _mm_hadd_ps( a, a ); // Horizontally add the 4 values
        a = _mm_hadd_ps( a, a ); // Horizontally add the 4 values
        return _mm_and_ps( a, omask );// Clear output using low bits of the mask
    }
    extern __inline __m128 __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_dp114_ps (__m128 a, __m128 b) {
        __m128 imask = _mm_castsi128_ps(_mm_set_epi32(0x0, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF));
        __m128 omask = _mm_castsi128_ps(_mm_set_epi32(0x0, 0x0, 0xFFFFFFFF, 0x0));
        a = _mm_and_ps( a, imask );
        a = _mm_mul_ps( a, b );
        a = _mm_hadd_ps( a, a ); // Horizontally add the 4 values
        a = _mm_hadd_ps( a, a ); // Horizontally add the 4 values
        return _mm_and_ps( a, omask );// Clear output using low bits of the mask
    }
    extern __inline __m128 __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_dp113_ps (__m128 a, __m128 b) {
        __m128 imask = _mm_castsi128_ps(_mm_set_epi32(0x0, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF));
        __m128 omask = _mm_castsi128_ps(_mm_set_epi32(0x0, 0x0, 0x0, 0xFFFFFFFF));
        a = _mm_and_ps( a, imask );
        a = _mm_mul_ps( a, b );
        a = _mm_hadd_ps( a, a ); // Horizontally add the 4 values
        a = _mm_hadd_ps( a, a ); // Horizontally add the 4 values
        return _mm_and_ps( a, omask );// Clear output using low bits of the mask
    }
    extern __inline __m128 __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_dp18_ps (__m128 a, __m128 b) {
        __m128 imask = _mm_castsi128_ps(_mm_set_epi32(0x0, 0x0, 0x0, 0xFFFFFFFF));
        __m128 omask = _mm_castsi128_ps(_mm_set_epi32(0x0, 0x0, 0xFFFFFFFF, 0x0));
        a = _mm_and_ps( a, imask );
        a = _mm_mul_ps( a, b );
        ROTATE_LEFT(a);
        return _mm_and_ps( a, omask );// Clear output using low bits of the mask
    }
    extern __inline __m128 __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_dp20_ps (__m128 a, __m128 b) {
        __m128 imask = _mm_castsi128_ps(_mm_set_epi32(0x0, 0x0, 0x0, 0xFFFFFFFF));
        __m128 omask = _mm_castsi128_ps(_mm_set_epi32(0x0, 0xFFFFFFFF, 0x0, 0x0));
        a = _mm_and_ps( a, imask );
        a = _mm_mul_ps( a, b );
        a = _mm_shuffle_ps(a, a, _MM_SHUFFLE(0,0,0,0));
        return _mm_and_ps( a, omask );// Clear output using low bits of the mask
    }
    extern __inline __m128 __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_dp24_ps (__m128 a, __m128 b) {
        __m128 imask = _mm_castsi128_ps(_mm_set_epi32(0x0, 0x0, 0x0, 0xFFFFFFFF));
        __m128 omask = _mm_castsi128_ps(_mm_set_epi32(0xFFFFFFFF, 0x0, 0x0, 0x0));
        a = _mm_and_ps( a, imask );
        a = _mm_mul_ps( a, b );
        ROTATE_RIGHT(a);
        return _mm_and_ps( a, omask );// Clear output using low bits of the mask
    }
    extern __inline __m128 __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_dp56_ps (__m128 a, __m128 b) {
        __m128 imask = _mm_castsi128_ps(_mm_set_epi32(0x0, 0x0, 0xFFFFFFFF, 0xFFFFFFFF));
        __m128 omask = _mm_castsi128_ps(_mm_set_epi32(0xFFFFFFFF, 0x0, 0x0, 0x0));
        a = _mm_and_ps( a, imask );
        a = _mm_mul_ps( a, b );
        a = _mm_hadd_ps( a, a ); // Horizontally add the 4 values
        ROTATE_RIGHT(a);
        return _mm_and_ps( a, omask );// Clear output using low bits of the mask
    }
#endif
    


void opConvolve (const int s, const int w, const int h,
                 const int ks, const int kw, 
                 const float* __restrict input, float* __restrict output, 
                 const float* __restrict kernel); 
                 
void opSeparableConvolve (const int s, const int w, const int h, const int kw, 
                          const float* __restrict input, float* __restrict output, 
                          const float* __restrict kernelX, 
                          const float* __restrict kernelY); 
                 
void naiveConvolve (const int s, const int w, const int h, 
                    const int ks, const int kw, 
                    const float* input, float* output, const float* kernel); 

void alignedConvolve (const int s, const int w, const int h,
                      const int ks, const int kw, 
                      const float* input, float* output, const float* kernel); 
                      
void sseNoReuse1Convolve (const int s, const int w, const int h,
                      const int ks, const int kw, 
                      const float* input, float* output, const float* kernel); 

void sseNoReuse2Convolve (const int s, const int w, const int h,
                      const int ks, const int kw, 
                      const float* input, float* output, const float* kernel); 

void sseNoReuse3Convolve (const int s, const int w, const int h,
                      const int ks, const int kw, 
                      const float* input, float* output, const float* kernel); 

void sseNoReuse4Convolve (const int s, const int w, const int h, 
                          const int ks, int kw, 
                          const float* __restrict input, float* __restrict output, 
                          const float* __restrict kernel); 

void sseNoReuse5Convolve (const int s, const int w, const int h,
                      const int ks, const int kw, 
                      const float* input, float* output, const float* kernel); 

void sseNoReuse6Convolve (const int s, const int w, const int h,
                      const int ks, const int kw, 
                      const float* input, float* output, const float* kernel); 

void sseNoReuse7Convolve (const int s, const int w, const int h,
                      const int ks, const int kw, 
                      const float* input, float* output, const float* kernel); 

void sseReuse1Convolve (const int s, const int w, const int h,
                      const int ks, const int kw, 
                      const float* input, float* output, const float* kernel); 

void sseReuse2Convolve (const int s, const int w, const int h,
                      const int ks, const int kw, 
                      const float* input, float* output, const float* kernel); 

void sseReuse3Convolve (const int s, const int w, const int h,
                      const int ks, const int kw, 
                      const float* input, float* output, const float* kernel); 

void sseReuse4Convolve (const int s, const int w, const int h,
                      const int ks, const int kw, 
                      const float* input, float* output, const float* kernel); 

void sseReuse5Convolve (const int s, const int w, const int h,
                      const int ks, const int kw, 
                      const float* input, float* output, const float* kernel); 

void sseReuse6Convolve (const int s, const int w, const int h,
                      const int ks, const int kw, 
                      const float* input, float* output, const float* kernel); 

void sseReuse7Convolve (const int s, const int w, const int h,
                      const int ks, const int kw, 
                      const float* input, float* output, const float* kernel);                       

void unalignedSSEConvolve (const int s, const int w, const int h,
                      const int ks, const int kw, 
                      const float* input, float* output, const float* kernel); 

void unalignedSSE4Convolve (const int s, const int w, const int h,
                      const int ks, const int kw, 
                      const float* input, float* output, const float* kernel); 

void loopUnrollConvolve (const int s, const int w, const int h, 
                         const int ks, const int kw, 
                         const float* __restrict input, float* __restrict output, const float* kernel);

void prefetchConvolve64 (const int s, const int w, const int h, 
                         const int ks, const int kw, 
                         const float* input, float* output, const float* kernel); 
                         
void prefetchConvolve128 (const int s, const int w, const int h, 
                          const int ks, const int kw, 
                          const float* input, float* output, const float* kernel);                                                    
                          
float convolution(const float* __restrict input, const int s, 
                         const float* __restrict kernel, const int ks, const int kw, 
                         const int x, const int y);

void loopBlockConvolve (const int s, const int w, const int h, 
                        const int ks, const int kw, 
                        const float* input, float* output, const float* kernel, 
                        const int xBlock, const int yBlock);
                        
void loopBlockLoopUnrollConvolve (const int s, const int w, const int h, 
                        const int ks, const int kw, 
                        const float* input, float* output, const float* kernel, 
                        const int xBlock, const int yBlock);                        
                                                               
void loopBlockAlignedSSEConvolve (const int s, const int w, const int h, 
                        const int ks, const int kw, 
                        const float* input, float* output, const float* kernel, 
                        const int xBlock, const int yBlock);
                                                               
void sse3Convolve (const int s, const int w, const int h, const int ks, 
                   const float* input, float* output, const float* kernel);
                   
void sse3ConvolveTest (const int s, const int w, const int h, 
                   const int kernelStride, 
                   const float* __restrict input, float* __restrict output, 
                   const float* __restrict k);                   
                   
void sse3Convolve1 (const int s, const int w, const int h, const int ks, 
                   const float* input, float* output, const float* kernel);
                   
void sse5Convolve (const int s, const int w, const int h, const int ks, 
                   const float* input, float* output, const float* kernel);
                   
void sse5Convolve1 (const int s, const int w, const int h, const int ks, 
                   const float* input, float* output, const float* kernel);
                   
void sse7Convolve (const int s, const int w, const int h, const int ks, 
                   const float* input, float* output, const float* kernel);
                   
void sse7Convolve1 (const int s, const int w, const int h, const int ks, 
                   const float* input, float* output, const float* kernel);
                   
void sse9Convolve (const int s, const int w, const int h, const int ks, 
                   const float* input, float* output, const float* kernel);
                   
void sse9Convolve1 (const int s, const int w, const int h, const int ks, 
                   const float* input, float* output, const float* kernel);
                   
void sse11Convolve (const int s, const int w, const int h, const int ks, 
                   const float* input, float* output, const float* kernel);
                   
void sse3CmConvolve (const int s, const int w, const int h, const int ks, 
                   const float* input, float* output, const float* kernel);
                   
void sse3LbConvolve (const int s, const int w, const int h, const int ks, 
                   const float* input, float* output, const float* kernel);
                   
void separableConvolve (const int s, const int w, const int h, const int kw, 
                        const float* __restrict input, float* __restrict output, 
                        const float* __restrict kernelX, 
                        const float* __restrict kernelY,
                        const bool processBoundaries); 
                        
void separableLoopBlockConvolve (const int s, const int w, const int h, const int kw, 
                        const float* input, float* output, const float* kernelX, const float* kernelY, 
                        const int xBlock, const int yBlock);

void scSSE (const int s, const int w, const int h, int kw, 
            const float* input, float* output, 
            const float* kernelX, const float* kernelY);

                                                                                       
void sc3SSE (const int s, const int w, const int h, 
             const float* input, float* output, 
             const float* kernelX, const float* kernelY);
                                       
void sc5SSE (const int s, const int w, const int h, 
             const float* input, float* output, const float* kernelX, const float* kernelY);
                                       
void sc7SSE (const int s, const int w, const int h, 
             const float* input, float* output, const float* kernelX, const float* kernelY);
                                       
void sc9SSE (const int s, const int w, const int h, 
             const float* input, float* output, const float* kernelX, const float* kernelY);

void scGaussian5SSE (const int s, const int w, const int h, 
                     const float* input, float* output, 
                     const float* kernel);
                     
void scGaussian7SSE (const int s, const int w, const int h, 
                     const float* input, float* output, 
                     const float* kernel);
                     
void scGaussian9SSE (const int s, const int w, const int h, 
                     const float* input, float* output, 
                     const float* kernel);       
                     
void processBoundaries2D(const int s, const int w, const int h, 
                       const int ks, int kw, 
                       const float* input, float* output, const float* kernel);

void processBoundariesS2D(const int s, const int w, const int h, 
                        const int kw, 
                        const float* input, float* output, const float* kernelX, const float* kernelY);

//helper functions
void printImage(int w, int h, int s, const float* out);
void printImageToFile(const std::string& file, int width, int height, int s, const float* out);
void printKernel2D(const int ks, const int kw, const float* kernel);
void clear2DBuffer( float* buffer, int stride, int height );

void copy2DBufferChunk(const float* inBuffer, float* outBuffer,
                       const int inX, const int inY, 
                       const int inStride, 
                       const int inWidth, const int inHeight,
                       const int outX, const int outY, 
                       const int outStride);
                       
void copy2DBoundaryChunk(const float* inBuffer, float* outBuffer,
                           const int outStride, const int outWidth, const int outHeight, 
                           const int replicateLeft, const int replicateTop,
                           const int replicateRight, const int replicateBottom,
                           const int inStride,  const int inWidth, const int inHeight);
                       
                       
float* allocateFloatAlignedBuffer (int width, int height);
int calculateAlignedStride (int width, int pixelSizeInBytes, int alignInBytes);

#endif // _OPCONVUTIONFILTER_H
