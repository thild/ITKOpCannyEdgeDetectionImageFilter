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

#include "boost/date_time/gregorian/gregorian.hpp"
#include "boost/date_time/posix_time/posix_time.hpp"
  
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



struct ImageStatInfo {
  public:
  string Image;
  double Time;
  vector <Checkpoint> Checkpoints;
  ImageStatInfo(string image, double time, vector<Checkpoint>& checkpoints) {
    Image = image;
    Time = time;
    Checkpoints = checkpoints; 
  }
};

struct CheckpointStatInfo {
  string Tag;
  double Mean;
  double Sum;
  double StDev; 
  double StDevPercentage; 
};

struct CheckpointInfo {
  string Tag;
  vector<double> Times;
};

struct IterationCheckpointInfo {
  string Tag;
  double Time;
  int Position;
};

struct IterationStatInfo {
  public:
  int Iteration;
  double Time;
  map <string, IterationCheckpointInfo> Checkpoints;
  
  IterationStatInfo(int iteration, vector <ImageStatInfo>& imageStatInfo) {
    Iteration = iteration;
    ProcessImageStat(imageStatInfo);
  }
  
  private:
  void ProcessImageStat(vector <ImageStatInfo>& imageStatInfo) {
    map<string, IterationCheckpointInfo> checkpointTimes;
    double time = 0;
    for (vector<ImageStatInfo>::iterator it = imageStatInfo.begin(); 
         it != imageStatInfo.end(); it++) {
      time += it->Time;
      for (vector<Checkpoint>::iterator checkpoint = it->Checkpoints.begin(); 
           checkpoint != it->Checkpoints.end(); checkpoint++) { 
        IterationCheckpointInfo& cpi = Checkpoints[checkpoint->Tag];     
        cpi.Tag = checkpoint->Tag;
        cpi.Time += checkpoint->Elapsed; 
        cpi.Position = checkpoint->Position; 
      }
    } 
    Time = time;
  }
  
};

struct DatasetStatInfo {
  public:
  
  string Base;
  double TotalTime;
  double MeanTime;
  double StDevTime;
  double StDevTimePercentage;
  
  vector <CheckpointStatInfo> CheckpointStats;
  
  
  DatasetStatInfo(string basee, vector <IterationStatInfo>& iterationStatInfo) {
    Base = basee;
    ProcessIterationStat(iterationStatInfo);
  }
  
  private:
  void ProcessIterationStat(vector <IterationStatInfo>& iterationStatInfo) {
    vector<double> times;
    int size = iterationStatInfo[0].Checkpoints.size();
    CheckpointInfo infos[size]; 
    map<string, vector<double> > checkpointTimes;
    for (vector<IterationStatInfo>::iterator it = iterationStatInfo.begin(); 
         it != iterationStatInfo.end(); it++) {
      times.push_back (it->Time);
      for (map <string, IterationCheckpointInfo>::iterator checkpointIt = it->Checkpoints.begin(); 
           checkpointIt != it->Checkpoints.end(); checkpointIt++) { 
           infos[checkpointIt->second.Position].Tag = checkpointIt->second.Tag;
           infos[checkpointIt->second.Position].Times.push_back(checkpointIt->second.Time);
      } 
    } 
    TotalTime = Sum(times);
    MeanTime = Mean(times);
    StDevTime = StDev(times);
    StDevTimePercentage = StDevTime / MeanTime * 100; 
    for (int i = 0; i < size; ++i) {
        CheckpointInfo& cpi = infos[i];
        CheckpointStatInfo csi;
        csi.Tag = cpi.Tag;
        csi.Mean = Mean(cpi.Times);
        csi.Sum = Sum(cpi.Times);
        csi.StDev = StDev(cpi.Times); 
        csi.StDevPercentage = csi.StDev / csi.Mean * 100;  
        CheckpointStats.push_back(csi);
    } 
  }
};


void TestDataset(string alg, string datasetsPath, string dataSet, int iterations, bool testIO, double variance, double upperThreshold, double lowerThreshold) {

  ofstream log;
  log.setf(ios::fixed,ios::floatfield); 
  try { 
  
    std::ostringstream dtime;
    boost::posix_time::ptime now =
        boost::posix_time::second_clock::local_time();
    boost::posix_time::time_facet*const f=
        new boost::posix_time::time_facet();
    f->set_iso_format();
    dtime.imbue(std::locale(dtime.getloc(),f));
    dtime << "canny-" << now  << ".log";
    
    string filename = dtime.str();
    
    log.open (filename.c_str(), std::ios::out);

    now = boost::posix_time::second_clock::local_time();
    f->set_iso_extended_format();
    dtime.str( std::string() ); dtime.clear();
    dtime.imbue(std::locale(dtime.getloc(),f));
    dtime << now;
     
    log << dataSet << " dataset test initiated at " << dtime.str() << endl;
    log << "time," << "iteration," << "filepath," << "fileidx," << "checkpoint," << "time" << endl;
  
    fs::path current_dir(datasetsPath + dataSet); 
    int nfiles = 0;
    StopWatch swTotal; 
    swTotal.Start();
    vector<IterationStatInfo> iterationStats;
    for (int i = 0; i < iterations; ++i) { 
      vector<ImageStatInfo> imageStats;
      fs::directory_iterator it(current_dir), eod;
      int j = 1;
      foreach (fs::path const & p, std::make_pair(it, eod)) { //for all files 
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
          StopWatch swImage; 
          StopWatchPool::GetStopWatch(alg)->Reset();
          StopWatchPool::GetStopWatch(alg)->StartNew();
          swImage.Start();
          cannyFilter->Update();
          swImage.Stop();
          vector<Checkpoint> result = 
            StopWatchPool::GetStopWatch(alg)->GetNotIgnoredCheckpoints();
            ImageStatInfo a(p.string(), swImage.GetElapsedTime(), result);
          imageStats.push_back(a);  
          
          std::ostringstream itime;
          now = boost::posix_time::second_clock::local_time();
          boost::posix_time::time_facet* const tf=
              new boost::posix_time::time_facet();
          tf->format("%H:%M:%S");
          itime.imbue(std::locale(itime.getloc(),tf));
          itime << now;
          
          for (vector<Checkpoint>::iterator checkpoint = result.begin(); 
               checkpoint != result.end(); checkpoint++) { 
            log << setprecision(0) << itime.str() << "," << i + 1 << "," << p.string() << "," << j << "," << checkpoint->Tag << "," << setprecision(6) << checkpoint->Elapsed << endl;
          }
          log << setprecision(0) << itime.str() << "," << i + 1<< "," << p.string() << "," << j << "," << "Total" << "," << setprecision(6) << swImage.GetElapsedTime() << endl;
          
          ++j;
        }
      }
      iterationStats.push_back(IterationStatInfo(i, imageStats)); 
    } 
    swTotal.Stop();  
    
    now = boost::posix_time::second_clock::local_time();
    f->set_iso_extended_format();
    dtime.str( std::string() ); dtime.clear();
    dtime.imbue(std::locale(dtime.getloc(),f));
    dtime << now;
     
    log << dataSet << " dataset test terminated at " << dtime.str() << endl;
    
    DatasetStatInfo datasetStat(dataSet, iterationStats);
    
//    cout << left << alg << " - " << swTotal.GetElapsedTime()  << endl << "Dataset " << dataSet << " - " << nfiles / iterations << " files - " << 
        
    cout << left << alg << endl;
        
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

    cout << left << "Dataset " << dataSet << " - " << nfiles / iterations << " files - " << 
      iterations << " iterations" << endl;
    
    bool first = true;
    
    for (vector<CheckpointStatInfo>::iterator it = datasetStat.CheckpointStats.begin(); 
                                              it != datasetStat.CheckpointStats.end(); it++ ) { 
      CheckpointStatInfo si = *it;
      if(first) {
        cout << left << setw(60) << "Checkpoint" << setw(15) << 
          "Mean" << setw(15) << "StDev" << setw(15) << "%" <<  setw(15) << "ImgMean" << "TotalDs" << endl;
        first = false;  
      }
      cout << left << 
        setw(60) << si.Tag << 
        setw(15) << setprecision(6) << si.Mean << 
        setw(15) << si.StDev << 
        setw(15) << setprecision(1) <<  si.StDevPercentage << 
        setw(15) << setprecision(6) << si.Mean / (nfiles / iterations) << 
        setprecision(6) << si.Sum << endl;
    }
     
    cout << left << 
      setw(60) << "Total dataset time" << 
      setw(15)  << setprecision(6) << datasetStat.MeanTime <<  
      setw(15) << datasetStat.StDevTime << 
      setw(15) << setprecision(1) << datasetStat.StDevTimePercentage << 
      setw(15) << setprecision(6) << datasetStat.MeanTime / (nfiles / iterations) << 
      setw(15) << setprecision(6) << datasetStat.TotalTime << endl;
    //cout <<  string(125, '-')  <<  endl;
    goto finally;
  }     
  catch( itk::ExceptionObject & err ) 
  { 
    std::cout << "ExceptionObject caught !" << std::endl; 
    std::cout << err << std::endl; 
    throw err; 
    goto finally;
  } 
  
  finally:
  {
    log.close();
  }
    
     
} 


int main (int argc, char *argv[])
{
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