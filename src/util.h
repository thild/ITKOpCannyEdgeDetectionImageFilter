// 
// util.h
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

#ifndef _UTIL_H
#define _UTIL_H

namespace std {

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while(std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}


std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    return split(s, delim, elems);
}

template <typename T>
T& insert_at(T& pContainer, size_t pIndex, const typename T::value_type& pValue)
{
    pContainer.insert(pContainer.begin() + pIndex, pValue);

    return pContainer;
}


void populateBuffer(const int imageStride, const int imageWidth, const int imageHeight,
                    float* inputImage) {
    for (int i = 0; i < imageStride * imageHeight; i++) {
         inputImage[i] = 0;
    }       
     
    for (int i = 0; i < imageHeight; i++) {
        for (int j = 0; j < imageWidth; j++) {
            inputImage[i * imageStride + j] = rand() % 5;
        }
    }
}

}
//#define cout sout
//#include <fstream>
//#include <iostream>
//#include <ostream>
//#include <cstdio>
//
//using namespace std;
//
//class mstream
//{
//  public:
//  ofstream coss;
//  mstream(void);
//  ~mstream(void);
//};
//
//template <class T>
//mstream& operator<<(std::ostream& (*pfun)(std::ostream& a))
//{
//    pfun(coss);
//    pfun(cout);
//    return *this;
//}

#endif //_UTIL_H