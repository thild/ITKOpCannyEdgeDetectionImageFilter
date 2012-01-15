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
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>

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

typedef itk::ImageRegionIterator<CharImageType> ImageIterator;

typedef itk::CastImageFilter< CharImageType, RealImageType> CastToRealFilterType;
typedef itk::CastImageFilter< RealImageType, CharImageType> CastToCharFilterType;
typedef itk::RescaleIntensityImageFilter<RealImageType, CharImageType > RescaleFilter;

typedef itk::OpCannyEdgeDetectionImageFilter<RealImageType, RealImageType> OpCannyFilter;
//typedef typename OpCannyFilter::Pointer OpCannyFilterPointer;

typedef itk::CannyEdgeDetectionImageFilter<RealImageType, RealImageType> CannyFilter;
typedef itk::ImageToImageFilter<RealImageType, RealImageType> ImageToImageFilterTest;


struct StatisticInfo {
  double Mean;
  double StDev; 
};

struct ConformanceInfo {
  public:
  ConformanceInfo() : Width(0), Height(0), NotMatchPixels(0) {}
  int Width;
  int Height; 
  string Image;
  long NotMatchPixels;
  double GetNotMatchPercentage() {
     return (double)NotMatchPixels / (double)GetNumberOfPixels();
  }
  long GetNumberOfPixels() {
     return Width * Height;
  }
};

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

void TestDataset(string alg, string ds, string spath, int iterations, double variance, double upperThreshold, double lowerThreshold) {
    std::cout << "Testing " << ds << " dataset..." << std::endl; 
    fs::path current_dir(spath); 
    vector<Checkpoint> checkpoints;
    int nfiles = 0;
    fs::directory_iterator it(current_dir), eod;
    StopWatch swTotal; 
    vector <double> readerStat;
    vector <double> writerStat;
    vector <double> filterStat;
    vector <double> processingStat;
    vector <double> ioStat;
    swTotal.Start();
    foreach (fs::path const & p, std::make_pair(it, eod)) { //for all files 
      if (is_regular_file(p)) { 
        ++nfiles;
        for (int i = 0; i < iterations; ++i) { 
          StopWatch swIO; 
          swIO.Start();
          string f;
          try { 
            ReaderType::Pointer reader = ReaderType::New();
            reader->SetFileName(p.string());
            
            StopWatch* swTemp = StopWatchPool::GetStopWatch("ImageFileReader");
            swTemp->StartNew();
            reader->Update();
            readerStat.push_back(swTemp->GetElapsedTime());
          
            CastToRealFilterType::Pointer toReal = CastToRealFilterType::New();  
            toReal->SetInput( reader->GetOutput() );
            toReal->Update();
            
            ImageToImageFilterTest::Pointer cannyFilter;
            
            if(alg == "OpCannyEdgeDetectionImageFilter") {
              f = p.parent_path().string() + "/result/op/" + p.filename();
              cannyFilter = OpCannyFilter::New();
              OpCannyFilter::Pointer filter = static_cast<OpCannyFilter*>(cannyFilter.GetPointer());
              filter->SetVariance( variance );
              filter->SetUpperThreshold( upperThreshold );
              filter->SetLowerThreshold( lowerThreshold );
            }
            else {
              f = p.parent_path().string() + "/result/native/" + p.filename();
              cannyFilter = CannyFilter::New();
              CannyFilter::Pointer filter = static_cast<CannyFilter*>(cannyFilter.GetPointer());
              filter->SetVariance( variance );
              filter->SetUpperThreshold( upperThreshold );
              filter->SetLowerThreshold( lowerThreshold );
            }
            
            cannyFilter->SetInput( toReal->GetOutput() );
//            cannyFilter->SetNumberOfThreads(1);
            
            
            StopWatch swFilter; 
            swFilter.Start();
            cannyFilter->Update();
            swFilter.Stop();
            filterStat.push_back(swFilter.GetElapsedTime());
            processingStat.push_back(StopWatchPool::GetStopWatch(alg)->GetElapsedTime());
            vector<Checkpoint> result = 
              StopWatchPool::GetStopWatch(alg)->GetNotIgnoredCheckpoints();
            checkpoints.insert (checkpoints.end(), result.begin(), result.end());

            
            RescaleFilter::Pointer rescale = RescaleFilter::New();
            rescale->SetOutputMinimum(   0 );
            rescale->SetOutputMaximum( 255 );
            rescale->SetInput( cannyFilter->GetOutput() );
            rescale->Update();
            
            WriterType::Pointer writer = WriterType::New();
            writer->SetFileName( f );
            writer->SetInput( rescale->GetOutput() );
            swTemp = StopWatchPool::GetStopWatch("ImageFileWriter");
            swTemp->StartNew();
            writer->Update(); 
            swTemp->Stop();
            writerStat.push_back(swTemp->GetElapsedTime());
          }
          catch( itk::ExceptionObject & err ) 
          { 
            std::cout << "ExceptionObject caught !" << std::endl; 
            std::cout << err << std::endl; 
            throw err;
          } 
          swIO.Stop();
          ioStat.push_back(swIO.GetElapsedTime()); 
        }
      }    
    }
    swTotal.Stop(); 
    
    cout << string(77, '-') << endl;
    cout << left << alg << " - Dataset " << ds << " - " << nfiles << " files - " << iterations << " iterations" << endl; 
    cout << string(77, '-') << endl;

    map<string, StatisticInfo> info = CheckpointStatistics::GetStatistics(checkpoints); 
    
    bool first = true;
    
    for (map<string, StatisticInfo>::iterator it = info.begin(); it != info.end(); it++ ) { 
      StatisticInfo si = it->second;
      if(first) {
        cout << left << setw(60) << "Checkpoint" << setw(10) << "Mean" << setw(5) << "StDev" << endl;
        cout << left << setw(60) << it->first << setw(10) << si.Mean << setw(5) << si.StDev << endl;
        first = false;  
      }
      else {
        cout << left << setw(60) << it->first << setw(10) << si.Mean << setw(5) << si.StDev << endl;
      }
    }
    
    cout << string(77, '-') << endl;
    cout << left << setw(60) << "Total processing time" << setw(10) << Mean(filterStat) << setw(5) << StDev(filterStat) << endl;
    cout << string(77, '-') << endl;
    cout << left << setw(60) << "Total filter time" << setw(10) << Mean(filterStat) << setw(5) << StDev(filterStat) << endl;
    cout << string(77, '-') << endl;
    cout << left << setw(60) << "Reader conversion" << setw(10) << Mean(readerStat) << setw(5) << StDev(readerStat) << endl;
    cout << left << setw(60) << "Writer conversion" << setw(10) << Mean(writerStat) << setw(5) << StDev(writerStat) << endl;
    cout << string(77, '-') << endl;
    cout << left << setw(60) << "Total filter chain time (including I/O)" << setw(10) << Mean(ioStat) << setw(5) << StDev(ioStat) << endl;
    cout << string(77, '-') << endl;
    cout << left << setw(70) << "Total dataset test time";
    cout << right << swTotal.GetElapsedTime() << endl;
    cout << string(77, '-') << endl;
}


int main (int argc, char *argv[])
{
// itk::MultiThreader::SetGlobalMaximumNumberOfThreads( 1 );
 cout.setf(ios::fixed);  
 cout.precision(4);
 
 bool conformanceTest = false;
 bool performanceTest = false;
 
  int optind=1;
  // decode arguments
  while ((optind < argc) && (argv[optind][0]=='-')) {
      string sw = argv[optind];
      if (sw=="-c") {
          conformanceTest = true; 
      }
      else if (sw=="-p") {
          performanceTest = true; 
      }
      optind++;
  }    

  string line;
  ifstream confFile ("test.cfg");
  map<string,string> config;
  
  if (confFile.is_open()) {
      while ( confFile.good() )
      {
          getline (confFile, line);
          int eqPos = line.find("=");
          string key = line.substr (0, eqPos);
          string value = line.substr (eqPos + 1, line.length() - eqPos + 1);
          config[key] = value;
      }
      confFile.close();
  }
  else {
      cout << "Unable to open test.cfg file" << flush << endl; 
      return 1;
  }
  
  using boost::lexical_cast;
  using boost::bad_lexical_cast;  
  
  float variance = lexical_cast<float>(config["variance"]);
  float upperThreshold = lexical_cast<float>(config["upper_threshold"]);
  float lowerThreshold = lexical_cast<float>(config["lower_threshold"]); 
  int iterations = lexical_cast<int>(config["iterations"]);
  
  vector<string> dataSets = split(config["ds"], ',');  
  
  //for all datasets  
  for ( vector<string>::iterator ds = dataSets.begin(); ds != dataSets.end(); ds++ ) {
    string spath = config["dsfolder"] +  string("/") + *ds;
    
    if(performanceTest) {
      string s = spath + "/result";
      fs::path rpath(s);
      if(!fs::exists(rpath)) 
          fs::create_directory(rpath);
      
      s = s + "/op";
      rpath = fs::path(s);
      if(!fs::exists(rpath)) 
          fs::create_directory(rpath);
  
      s = spath + "/result/native";
      rpath = fs::path(s);
      if(!fs::exists(rpath)) 
          fs::create_directory(rpath);
  
       
      TestDataset("OpCannyEdgeDetectionImageFilter", *ds, spath, iterations, variance, upperThreshold, lowerThreshold);
      cout << endl;
      TestDataset("CannyEdgeDetectionImageFilter", *ds, spath, iterations, variance, upperThreshold, lowerThreshold);
      cout << endl;
      cout << endl;
    }
  
  
    if(conformanceTest) {
      cout << "Running conformance test..." << endl;
      
      map<string, ConformanceInfo> conformanceInfo;
      
      fs::path current_dir(spath); 
      fs::directory_iterator it(current_dir), eod;
      
      foreach (fs::path const & p, std::make_pair(it, eod)) { //for all files 
        if (is_regular_file(p)) { 
        
//          WriterType::Pointer writer = WriterType::New();
          
          ReaderType::Pointer opReader = ReaderType::New();
          ReaderType::Pointer nativeReader = ReaderType::New();
          
          string opFile = p.parent_path().string() + "/result/op/" + p.filename();
          opReader->SetFileName(opFile);
          
          string nativeFile = p.parent_path().string() + "/result/native/" + p.filename();
          nativeReader->SetFileName(nativeFile);
          
          opReader->Update();
          nativeReader->Update();
          
          
//          writer->SetFileName(p.parent_path().string() + "/result/op-" + p.filename());
//          writer->SetInput(opReader->GetOutput());
//          writer->Update();
//          
//          writer->SetFileName(p.parent_path().string() + "/result/native-" + p.filename());
//          writer->SetInput(nativeReader->GetOutput());
//          writer->Update();
          
          CharImageType::Pointer opIm = opReader->GetOutput();
          CharImageType::Pointer nativeIm = nativeReader->GetOutput();
          
          ImageIterator  opIt( opIm, opIm->GetLargestPossibleRegion() );
          ImageIterator  nativeIt( nativeIm, nativeIm->GetLargestPossibleRegion());
          
          typename CharImageType::SizeType regionSize = opIm->GetLargestPossibleRegion().GetSize();
          
          opIt.GoToBegin();
          nativeIt.GoToBegin();
          while( !opIt.IsAtEnd() && !nativeIt.IsAtEnd() )
          {
            bool equal = opIt.Get() == nativeIt.Get();
            if (!equal) {
              ConformanceInfo ci = conformanceInfo[p.filename()];
              if (ci.NotMatchPixels == 0) {
                ci.Width = regionSize[0];
                ci.Height = regionSize[1];
                ci.Image = p.filename();
              }
              ci.NotMatchPixels++;
              conformanceInfo[p.filename()] = ci;
            }
            ++opIt;
            ++nativeIt;
          }    
        }   
      } //conformance test iterations
        cout << left << setw(20) <<  
        "Image" << setw(20) << "total pixels" << setw(20) << "not match pixels" << setw(20) << "%" << endl;
      for ( map<string, ConformanceInfo>::iterator it = conformanceInfo.begin(); 
                                                   it != conformanceInfo.end(); 
                                                   it++ ) { 
        ConformanceInfo ci = it->second;
        cout << left << setw(20) << 
        ci.Image << setw(20) << ci.GetNumberOfPixels() << setw(20) << ci.NotMatchPixels << setw(20) << ci.GetNotMatchPercentage() * 100 << endl;
      }
    }  
  } //dataset iterations
  
       
  return EXIT_SUCCESS;	
}