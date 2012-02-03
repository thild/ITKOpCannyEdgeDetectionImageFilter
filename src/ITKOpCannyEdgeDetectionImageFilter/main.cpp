// 
// main.cpp
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

//#define CALLGRIND

#if defined(_MSC_VER)
#pragma warning ( disable : 4786 ) 
#endif  

#include "itkImage.h"
#include "itkImportImageFilter.h"

#include "itkImageFileWriter.h"                 
#include "itkImageFileReader.h"
#include "itkObjectFactoryBase.h"
#include "itkDynamicLoader.h"
#include "itkOpCannyEdgeDetectionImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkCannyEdgeDetectionImageFilter.h"
#include "itkStopWatch.h"

#include <iostream> 
#include <iomanip>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <list>
#include <map>
#include <math.h>
#include <omp.h>
#include "util.h"
#include "statistics.h"
#include <sys/time.h>
#include <sys/resource.h>
#include <utility>
#define BOOST_FILESYSTEM_VERSION 3
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>

#include <valgrind/callgrind.h>

#define foreach BOOST_FOREACH

namespace fs = boost::filesystem;
using std::cout;
using std::cerr;   
using std::endl;
using std::setw;
using std::string;
using std::ifstream; 

using namespace std;

typedef unsigned char     CharPixelType;  //  IO
typedef float             RealPixelType;  //  Operations
const   unsigned int      Dimension = 2;
 
typedef itk::Image< CharPixelType, Dimension >    CharImageType;
typedef itk::Image< RealPixelType, Dimension >    RealImageType;

typedef itk::ImageFileReader< CharImageType >  ReaderType;
typedef itk::ImageFileWriter< CharImageType >  WriterType;

typedef itk::ImageRegionIterator<RealImageType> ImageIterator;

typedef itk::CastImageFilter< CharImageType, RealImageType> CastToRealFilterType;
typedef itk::CastImageFilter< RealImageType, CharImageType> CastToCharFilterType;
typedef itk::RescaleIntensityImageFilter<RealImageType, CharImageType > RescaleFilter;

typedef itk::OpCannyEdgeDetectionImageFilter<RealImageType, RealImageType> OpCannyFilter;
//typedef typename OpCannyFilter::Pointer OpCannyFilterPointer;

typedef itk::CannyEdgeDetectionImageFilter<RealImageType, RealImageType> CannyFilter;
typedef itk::ImageToImageFilter<RealImageType, RealImageType> ImageToImageFilterTest;


struct StatisticInfo {
  string Tag;
  double Mean;
  double StDev; 
};

//struct ConformanceInfo {
//  public:
//  ConformanceInfo() : Width(0), Height(0), CorrectlyDetect(0) {}
//  int Width;
//  int Height; 
//  string Image;
//  long NoDetected;
//  long ErroneouslyDetected;
//  long CorrectlyDetect;
//  double GetNotMatchPercentage() {
//     return (double)NotMatchPixels / (double)GetNumberOfPixels();
//  }
//  long GetNumberOfPixels() {
//     return Width * Height;
//  }
//};

struct CheckpointStatistics {
  public:
  
  static map<string, StatisticInfo> GetStatistics(vector<Checkpoint>& checkpoints) {
    map<string, vector<double> > checkpointTimes = ProcessCheckpoints(checkpoints);
    map<string, StatisticInfo> stat;
    for ( map<string, vector<double> >::iterator it = checkpointTimes.begin(); 
                                                it != checkpointTimes.end(); 
                                                it++ ) { 
      stat[it->first].Mean = Mean(it->second);
      stat[it->first].StDev = StDev(it->second);
    }
    return stat;
  }
  
  private:
  
  static map<string, vector<double> > ProcessCheckpoints(vector<Checkpoint>& checkpoints) {
    map<string, vector<double> > checkPointTimes;
    for (vector<Checkpoint>::iterator it = checkpoints.begin(); 
         it != checkpoints.end(); it++) {
       checkPointTimes[it->Tag].push_back(it->Elapsed);
    } 
    return checkPointTimes;
  }
  
  CheckpointStatistics()  {  }
};

void TestDataset(string alg, string datasetsPath, string dataSet, int iterations, bool testIO, double variance, double upperThreshold, double lowerThreshold) {
    fs::path current_dir(datasetsPath + dataSet); 
    vector<Checkpoint> checkpoints;
    int nfiles = 0;
    StopWatch swTotal; 
    vector <double> readerStat;
    vector <double> writerStat;
    vector <double> filterStat;
    vector <double> ioStat;
    swTotal.Start();
    for (int i = 0; i < iterations; ++i) { 
      try { 
        fs::directory_iterator it(current_dir), eod;
        foreach (fs::path const & p, std::make_pair(it, eod)) { //for all files 
//          sleep(0.0016);
          if (is_regular_file(p)) { 
            ++nfiles;
            ReaderType::Pointer reader = ReaderType::New();
            reader->SetFileName(p.string());
            reader->Update();
            CastToRealFilterType::Pointer toReal = CastToRealFilterType::New();  
            toReal->SetInput( reader->GetOutput() );
            toReal->Update();
            ImageToImageFilterTest::Pointer cannyFilter;
            if(alg == "OpCannyEdgeDetectionImageFilter") {
              cannyFilter = OpCannyFilter::New();
              OpCannyFilter::Pointer filter = static_cast<OpCannyFilter*>(cannyFilter.GetPointer());
              filter->SetVariance( variance );
              filter->SetUpperThreshold( upperThreshold );
              filter->SetLowerThreshold( lowerThreshold );
            }
            else {
              cannyFilter = CannyFilter::New();
              CannyFilter::Pointer filter = static_cast<CannyFilter*>(cannyFilter.GetPointer());
              filter->SetVariance( variance );
              filter->SetUpperThreshold( upperThreshold );
              filter->SetLowerThreshold( lowerThreshold );
            }
            cannyFilter->SetInput( toReal->GetOutput() );
            StopWatch swFilter; 
            StopWatchPool::GetStopWatch(alg)->Reset();
            StopWatchPool::GetStopWatch(alg)->StartNew();
            swFilter.Start();
            cannyFilter->Update();
            swFilter.Stop();
            filterStat.push_back(swFilter.GetElapsedTime());
            vector<Checkpoint> result = 
              StopWatchPool::GetStopWatch(alg)->GetNotIgnoredCheckpoints();
            checkpoints.insert (checkpoints.end(), result.begin(), result.end());
            cannyFilter->Modified();   
          }
        }
      }    
      catch( itk::ExceptionObject & err ) 
      { 
        std::cout << "ExceptionObject caught !" << std::endl; 
        std::cout << err << std::endl; 
        throw err;
      } 
    }
    swTotal.Stop(); 
    
    //cout << string(125, '-') << endl;
    cout << left << alg << endl << "Dataset " << dataSet << " - " << nfiles / iterations << " files - " << 
        iterations << " iterations - ";
    if (alg == "OpCannyEdgeDetectionImageFilter" ) {
#ifdef _OPENMP     
      cout << itk::MultiThreader::GetGlobalMaximumNumberOfThreads() << 
       " threads ITK - " << omp_get_max_threads() << " threads OMP" << endl; 
#else
      cout << itk::MultiThreader::GetGlobalMaximumNumberOfThreads() << 
        " threads ITK - " << "OMP is disabled" << endl; 
#endif
    }
    else {
      cout << itk::MultiThreader::GetGlobalMaximumNumberOfThreads() << 
        " threads ITK" << endl; 
    }
    //cout << string(125, '-') << endl;

    map<string, StatisticInfo> info = CheckpointStatistics::GetStatistics(checkpoints); 
    
    bool first = true;
    double mean = 0;
    double stdev = 0;
    
    StatisticInfo ord[6];
    
    for (map<string, StatisticInfo>::iterator it = info.begin(); it != info.end(); it++ ) { 
      if (it->first == "GaussianBlur") {
        it->second.Tag = "GaussianBlur";
        ord[0] = it->second; 
      }
      else if (it->first == "Compute2ndDerivative") {
        it->second.Tag = "Compute2ndDerivative";
        ord[1] = it->second; 
      }
      else if (it->first == "Compute2ndDerivativePos") {
        it->second.Tag = "Compute2ndDerivativePos";
        ord[2] = it->second; 
      }
      else if (it->first == "ZeroCrossing") {
        it->second.Tag = "ZeroCrossing";
        ord[3] = it->second; 
      }
      else if (it->first == "Multiply") {
        it->second.Tag = "Multiply";
        ord[4] = it->second; 
      }
      else if (it->first == "HysteresisThresholding") {
        it->second.Tag = "HysteresisThresholding";
        ord[5] = it->second; 
      }
    }
    
    
    for ( int i = 0; i < 6; ++i ) {
      StatisticInfo si = ord[i];
      mean = si.Mean;
      stdev = si.StDev;
      if(first) {
        cout << left << setw(85) << "Checkpoint" << setw(13) << 
          "Mean" << setw(13) << "StDev" << "%" << endl;
        cout << left << setw(85) << si.Tag << setw(13) << setprecision(6) << mean << setw(13) << 
          stdev << setprecision(1) <<  stdev / mean * 100 << setprecision(6) << endl;
        first = false;  
      }
      else {
        cout << left << setw(85) << si.Tag << setw(13) << mean << setw(13) << 
          stdev << setprecision(1) <<  stdev / mean * 100 << setprecision(6) << endl;
      }
    }
    
    //cout << string(125, '-') << endl;
    mean = Mean(filterStat);
    stdev = StDev(filterStat);
    cout << left << setw(85) << "Total filter time" << setw(13) << 
      mean << setw(13) << stdev << setprecision(1) <<  
      stdev / mean * 100 << setprecision(6) << endl;
    //cout << string(125, '-') << endl;
    cout << left << setw(85) << "Total dataset test time";
    cout << left << mean * nfiles << endl;
    //cout <<  string(125, '-')  <<  endl;
}


int main (int argc, char *argv[])
{
// itk::MultiThreader::SetGlobalMaximumNumberOfThreads( 1 );
 cout.setf(ios::fixed);  
 cout.precision(4);
 
 bool conformanceTest = false;
 bool performanceTest = false;
 
  int optind=1;
  
  string configFile = "test.cfg";
  
  // decode arguments
  while ((optind < argc) && (argv[optind][0]=='-')) {
      string sw = argv[optind];
      if (sw=="-c") {
          configFile = argv[optind + 1];
      }
      else if (sw=="-ct") {
          conformanceTest = true; 
      }
      else if (sw=="-pt") {
          performanceTest = true; 
      }
      optind++;
  }    

  string line;
  ifstream confFile (configFile.c_str());
  map<string,string> config;
  
  if (confFile.is_open()) {
      while ( confFile.good() )
      {
          getline (confFile, line);
          uint eqPos = line.find("=");
          if (eqPos != string::npos) {
            string key = line.substr (0, eqPos);
            string value = line.substr (eqPos + 1, line.length() - eqPos + 1);
            config[key] = value;
          }
      }
      confFile.close();
  }
  else {
      cout << "Unable to open test.cfg file" << flush << endl; 
      return 1;
  }

   
  using boost::lexical_cast;
  using boost::bad_lexical_cast;  

  double variance = lexical_cast<float>(config["variance"]);
  double upperThreshold = lexical_cast<float>(config["upper_threshold"]);
  double lowerThreshold = lexical_cast<float>(config["lower_threshold"]); 
  
  if (config.find("max_threads_omp") != config.end()) {
    int maxThreads = lexical_cast<int>(config["max_threads_omp"]);
#ifdef _OPENMP     
     omp_set_num_threads(maxThreads);
#endif
  }
  
  if (config.find("max_threads_itk") != config.end()) {
    int maxThreads = lexical_cast<int>(config["max_threads_itk"]);
    itk::MultiThreader::SetGlobalMaximumNumberOfThreads( maxThreads );  
  }
  
  if(performanceTest) {
    fs::path rpath("results");
//    fs::remove_all(rpath);
    fs::create_directory(rpath);
   
    int iterations = lexical_cast<int>(config["iterations"]);
    
    vector<string> dataSets = split(config["ds"], ',');  
    bool testIO = config.find("test_io") == config.end() ? true : lexical_cast<int>(config["test_io"]);
    string algorithm = config["algorithm"];
//    cout << "testIO" << testIO << endl;
    //for all datasets  
    for ( vector<string>::iterator ds = dataSets.begin(); ds != dataSets.end(); ds++ ) {
      
      fs::create_directory(fs::path("results/" + *ds));
      fs::create_directory(fs::path("results/" + *ds + "/op"));
      fs::create_directory(fs::path("results/" + *ds));
      fs::create_directory(fs::path("results/" + *ds + "/native"));
  
      string datasetsPath = config["dsfolder"] +  string("/");
       
      if (algorithm == "both") {
        TestDataset("CannyEdgeDetectionImageFilter", datasetsPath, *ds, iterations, testIO, variance, upperThreshold, lowerThreshold);
        cout << endl;
        TestDataset("OpCannyEdgeDetectionImageFilter", datasetsPath, *ds, iterations, testIO, variance, upperThreshold, lowerThreshold);
      }
      else if (algorithm == "native") {
        TestDataset("CannyEdgeDetectionImageFilter", datasetsPath, *ds, iterations, testIO, variance, upperThreshold, lowerThreshold);
      }
      else if (algorithm == "op") {
        TestDataset("OpCannyEdgeDetectionImageFilter", datasetsPath, *ds, iterations, testIO, variance, upperThreshold, lowerThreshold);
      }
      cout << endl;
      cout << endl;
    }
  }
  
  if(conformanceTest) {
   
    vector<string> dataSets = split(config["ds"], ',');  
    string datasetsPath = config["dsfolder"] +  string("/");
    
    
    //for all datasets  
    for ( vector<string>::iterator ds = dataSets.begin(); ds != dataSets.end(); ds++ ) {
      long tf = 0;
     
      double pco = 0;
      double pnd = 0;
      double pfa = 0;
     
      cout << "Running " <<  *ds << " dataset conformance test..." << endl;
      
//      map<string, ConformanceInfo> conformanceInfo;
      
      string createPath = "results";
      fs::create_directory(createPath);
      
      createPath += "/conformance";
      fs::create_directory(createPath);
      createPath += *ds;
      fs::create_directory(createPath);
      
      fs::create_directory(createPath + "/op");
      fs::create_directory(createPath + "/native");
      
      fs::path current_dir(datasetsPath + *ds ); 
      
      fs::directory_iterator it(current_dir), eod;
      
      foreach (fs::path const & p, std::make_pair(it, eod)) { //for all files 
        if (is_regular_file(p)) { 
          ++tf;
          ReaderType::Pointer opReader = ReaderType::New();
          ReaderType::Pointer nativeReader = ReaderType::New();
          
          //cout << "Reading " + p.string() << endl;
          
          opReader->SetFileName(p.string());
          nativeReader->SetFileName(p.string());
          
          opReader->Update();
          nativeReader->Update();

          CastToRealFilterType::Pointer opToReal = CastToRealFilterType::New();  
          opToReal->SetInput( opReader->GetOutput() );
          opToReal->Update();
              
          CastToRealFilterType::Pointer nativeToReal = CastToRealFilterType::New();  
          nativeToReal->SetInput( nativeReader->GetOutput() );
          nativeToReal->Update();
              
          OpCannyFilter::Pointer opCannyFilter = OpCannyFilter::New();
          opCannyFilter->SetVariance( variance );
          opCannyFilter->SetUpperThreshold( upperThreshold );
          opCannyFilter->SetLowerThreshold( lowerThreshold );
          
          CannyFilter::Pointer nativeCannyFilter = CannyFilter::New();
          nativeCannyFilter->SetVariance( variance );
          nativeCannyFilter->SetUpperThreshold( upperThreshold );
          nativeCannyFilter->SetLowerThreshold( lowerThreshold );
              
          opCannyFilter->SetInput( opToReal->GetOutput() );
          nativeCannyFilter->SetInput( nativeToReal->GetOutput() );
          
          opCannyFilter->Update(); 
          nativeCannyFilter->Update();
              
//          RescaleFilter::Pointer opRescale = RescaleFilter::New();
//          opRescale->SetOutputMinimum(   0 );
//          opRescale->SetOutputMaximum( 255 );
//          opRescale->SetInput( opCannyFilter->GetOutput() );
//          opRescale->Update();
//          
//          RescaleFilter::Pointer nativeRescale = RescaleFilter::New();
//          nativeRescale->SetOutputMinimum(   0 );
//          nativeRescale->SetOutputMaximum( 255 );
//          nativeRescale->SetInput( nativeCannyFilter->GetOutput() );
//          nativeRescale->Update();
//              
//              
//          WriterType::Pointer writer = WriterType::New();
//          string f = "results/conformance/" + *ds + "/op/" + p.filename().string();
//          writer->SetFileName( f );
//          writer->SetInput( opRescale->GetOutput() );
//          writer->Update(); 
//          
////          cout << "Writing " << f << endl;
//          
//     
//          f = "results/conformance/" + *ds + "/native/" + p.filename().string();
//          writer->SetFileName( f );
//          writer->SetInput( nativeRescale->GetOutput() );
//          writer->Update(); 
//     
//          //cout << "Writing " << f << endl;
//     
//          CharImageType::Pointer opIm = opRescale->GetOutput();
//          CharImageType::Pointer nativeIm = nativeRescale->GetOutput();
//          
//          ImageIterator  opIt( opIm, opIm->GetLargestPossibleRegion() );
//          ImageIterator  nativeIt( nativeIm, nativeIm->GetLargestPossibleRegion());
//          
//            typename RealImageType::SizeType regionSize = opCannyFilter->GetOutput()->GetLargestPossibleRegion().GetSize(); 
//
//            float* op =  opCannyFilter->GetOutput()->GetBufferPointer() ;
//            float* native =  nativeCannyFilter->GetOutput()->GetBufferPointer() ; 
//            int imageStride = opCannyFilter->GetOutput()->GetOffsetTable()[1];
// 
//            for (uint y = 0; y < regionSize[1]; ++y) { 
//              uint x = 0;
//              for (; x < regionSize[1] - 4; x += 4) {
//              
//                __m128 opv = _mm_load_ps(&op[y * imageStride + x]); 
//                __m128 nativev = _mm_load_ps(&native[y * imageStride + x]); 
//                
//                unsigned int mask = _mm_movemask_ps(_mm_cmpeq_ps(opv, nativev)); 
//                if (mask == 0xFFF) continue;
//                cout << "Not equal" << endl;
//                 
//                if(mask & 1) //pixel 0  
//                {
//                } 
//                     
//                if((mask & 2) >> 1) //pixel 1
//                {
//                } 
//                     
//                if((mask & 4) >> 2) //pixel 2
//                {
//                } 
//                     
//                if((mask & 8) >> 3) //pixel 3
//                {
//                } 
//              }       
//            }
//      
//     
      
                  
          RealImageType::Pointer opIm = opCannyFilter->GetOutput();
          RealImageType::Pointer nativeIm = nativeCannyFilter->GetOutput();

          ImageIterator  opIt( opIm, opIm->GetLargestPossibleRegion() );
          ImageIterator  nativeIt( nativeIm, nativeIm->GetLargestPossibleRegion());
          
          opIt.GoToBegin();
          nativeIt.GoToBegin();
          long tp = 0;
          long fn = 0;
          long fp = 0;
          long ni = 0;
          long nb = 0;
          cout.precision(6);
          while( !opIt.IsAtEnd() && !nativeIt.IsAtEnd() )
          { 
            float  op = opIt.Get();
            float  n =  nativeIt.Get();
            if (n == 1) ++ni;
            if (op == 1) ++nb;
            if (op == 1 && n == 1) {
              ++tp;
            }
            else if (op == 0 && n == 1) {
              ++fn;  
            } 
            else if (op == 1 && n == 0) {
              ++fp;  
            }  
            ++opIt;
            ++nativeIt;
          }
          //pcd += cd / tp; 
          pco += (double)tp / (double)max(ni, nb); 
//          cout << tp << endl;
//          cout << ni << endl;
//          cout << nb << endl;
//          cout << max(ni, nb) << endl;
//          cout << pco << endl;
          pnd += (double)fn / (double)max(ni, nb);
          pfa += (double)fp / (double)max(ni, nb); 
        }   
      } //conformance test iterations
      cout << left << setw(20) <<  
      "Correclty detected" << setw(20) << "Not detected" << setw(20) << "Erroneously detected" << setw(20) << endl;
      cout << left << setw(20) << pco / tf * 100 << setw(20)  << pnd / tf * 100  << setw(20) << pfa / tf * 100 << endl;
      cout << endl << endl;
    }  
  } //dataset iterations
  
       
  return EXIT_SUCCESS;	
}