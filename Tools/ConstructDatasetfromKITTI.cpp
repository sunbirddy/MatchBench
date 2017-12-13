// This file construct our datasets from KITTI sequences.
// Input :
// 	1. input directory (one KITTI sequence)
// 	2. output directory
// Author :
//	JiaWang Bian



#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>

#ifdef WIN32
#include <direct.h>
#elif __linux__
#include <sys/stat.h>
#endif

using namespace cv;
using namespace std;


int GetImageNumbers(string TimeStampFile);

void CopyPoses(string srcPose, string resPose, vector<Mat> &vPoses);

int main(int argc, char **argv){

	if(argc != 4)
    {
        cerr << endl << "Usage: ./ConstructDatasetfromKITTI InputDir(..../dataset/) SequenceNumber OutputDir" << endl;
        return 1;
    }

    string ImagePath = string(argv[1]) + "/sequences/" + string(argv[2]) + "/image_0/";
    string TimeStampFile = string(argv[1]) + "/sequences/" + string(argv[2]) + "/times.txt";
    string PoseFile  = string(argv[1]) + "/poses/" + string(argv[2]) + ".txt";
    
    string OutImageDir = string(argv[3]) + "/Images/";
    string OutputPose = OutImageDir + "pose.txt";
    string OutputPairs = string(argv[3]) + "/pairs.txt"; 

#ifdef WIN32
	mkdir(OutImageDir.c_str());
#elif __linux__
	mkdir(OutImageDir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#endif

    // Copy Images
    int nImages = GetImageNumbers(TimeStampFile);
     for (int i = 0; i < nImages; ++i)
     {
         char buffer1[50], buffer2[50];
         sprintf(buffer1, "/%06d.png", i);
         sprintf(buffer2, "/%04d.png", i);
         Mat img = imread(ImagePath + string(buffer1));
         if(img.empty()){
             cout << buffer1 <<endl;
             exit(-1);
         }
         imwrite(OutImageDir + string(buffer2), img);
     }

    // Copy Poses
    vector<Mat> vPoses;
    CopyPoses(PoseFile, OutputPose, vPoses);

    // produce pairs
    ofstream ofs;
    ofs.open(OutputPairs.c_str());
    int npersets = 5;
    int ncount = 0;
    for (int i = 0; i < nImages; ++i)
    {
        if(i % npersets == 0)
            continue;

        int l = i / npersets * npersets;
        int r = i;

        ofs << l << " " << r << " ";

        Mat pose = vPoses[r].inv() * vPoses[l];

        ofs << pose.at<double>(0,0) << " ";
        ofs << pose.at<double>(0,1) << " ";
        ofs << pose.at<double>(0,2) << " ";
        ofs << pose.at<double>(0,3) << " ";
        ofs << pose.at<double>(1,0) << " ";
        ofs << pose.at<double>(1,1) << " ";
        ofs << pose.at<double>(1,2) << " ";
        ofs << pose.at<double>(1,3) << " ";
        ofs << pose.at<double>(2,0) << " ";
        ofs << pose.at<double>(2,1) << " ";
        ofs << pose.at<double>(2,2) << " ";
        ofs << pose.at<double>(2,3) << endl;

        ncount++;
    }

    ofs.close();
    cout << ncount << "pairs are saved."<<endl;

    // write calibrations
    ofs.open((string(argv[3]) + "/camera.txt").c_str());
    ofs << "718.856 0 607.1928 "<<endl;
    ofs << "0 718.856 185.2157 "<<endl;
    ofs << "0 0 1"; 
    ofs.close();
    cout << "camera.txt include camera (sequence 00-02) internal parameters."<<endl;

    return 0;
}



int GetImageNumbers(string TimeStampFile){
    ifstream f;
    f.open(TimeStampFile.c_str());

    int count = 0;
    while(!f.eof())
    {
        string s;
        getline(f,s);
        if(!s.empty())
        {
            count ++;
        }
    }
    f.close();
    return count;
}

void CopyPoses(string srcPose, string resPose, vector<Mat> &vPoses){
    ifstream f;
    f.open(srcPose.c_str());
    vPoses.reserve(10000);

    while(!f.eof())
    {
        Mat temp = Mat::eye(4,4,CV_64FC1);
        double *data = (double *)temp.data;
        for(int i=0; i<12; i++){
            f >> *data++;
        }
        vPoses.push_back(temp);
    }
    f.close();
    vPoses.pop_back();

 
    ofstream ofs;
    ofs.open(resPose.c_str());
    for(int i = 0; i < vPoses.size(); i++){
        double * data = (double *)vPoses[i].data;
        for (int j = 0; j < 11; ++j)
        {
            ofs << data[j] << " ";
        }
        ofs << data[11] << endl;
    }
    ofs.close();
}