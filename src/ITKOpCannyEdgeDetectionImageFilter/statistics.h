// 
// statistics.h
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

#ifndef _STATISTICS_H
#define _STATISTICS_H

#include <cmath>
#include <cstdlib>
#include <vector>

using namespace std;


template <typename T>
double Mean(const vector<T>& arr)
{
    T sum = 0;
    for (size_t idx = 0; idx < arr.size(); idx++)
    {
        sum += arr[idx];
    }
    return sum/static_cast<double>(arr.size());
}

template <typename T>
double StDev(const vector<T>& arr)
{
    double mean = Mean(arr);
    double variance = 0.0;
    for (size_t idx = 0; idx < arr.size(); idx++)
    {
        double temp = arr[idx] - mean;
        variance += temp*temp;
    }
    
    // Compute sample variance using Bessel's correction (see http://en.wikipedia.org/wiki/Bessel%27s_correction)
    variance /= static_cast<double>(arr.size()) - (arr.size() == 1 ? 0.0 : 1.0);  
    
    // Standard deviation is square root of variance
    return std::sqrt(variance);
}

#endif //_STATISTICS_H