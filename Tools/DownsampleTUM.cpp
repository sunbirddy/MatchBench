// This file downsample constructed sequences 01-03 to sequences 05-08 
// Input :
// 	1. input directory (constructed sequences)
// 	2. output directory
// Requirement :
//	You need run ConstructDatasetfromTUM to construct video sequences firstly.
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


void LoadPoses(const string &strFile, vector<Mat> &vPosesAll);


int main(int argc, char **argv){

	if(argc != 3)
    {
        cerr << endl << "Usage: ./DownsampleTUM InputDir OutputDir" << endl;
        return 1;
    }

    string strPoseFile = string(argv[1]) + "/Images/pose.txt";

    vector<Mat> vPoses, vPosesAll;
    LoadPoses(strPoseFile, vPosesAll);

    int nImages = vPosesAll.size();

    string OutputDir = string(argv[2]) + "/Images";

#ifdef WIN32
	mkdir(OutputDir.c_str());
#elif __linux__
	mkdir(OutputDir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#endif

    ofstream ofs;
    ofs.open((OutputDir + "/pose.txt").c_str());


    // downsample
    int stride = 15;
    int count = 0;
    for (int i = 0; i < nImages; ++i)
    {
        if(i % stride != 0)
            continue;

         char buffer1 [50], buffer2[50];
         sprintf(buffer1, "/%04d.png", i);
         sprintf(buffer2, "/%04d.png", count++);
         Mat img = imread(string(argv[1]) + "/Images" + string(buffer1));
         imwrite(OutputDir + buffer2, img);

        vPoses.push_back(vPosesAll[i].clone());

        Mat &pose = vPosesAll[i];
        ofs << pose.at<float>(0,0) << " ";
        ofs << pose.at<float>(0,1) << " ";
        ofs << pose.at<float>(0,2) << " ";
        ofs << pose.at<float>(0,3) << " ";
        ofs << pose.at<float>(1,0) << " ";
        ofs << pose.at<float>(1,1) << " ";
        ofs << pose.at<float>(1,2) << " ";
        ofs << pose.at<float>(1,3) << " ";
        ofs << pose.at<float>(2,0) << " ";
        ofs << pose.at<float>(2,1) << " ";
        ofs << pose.at<float>(2,2) << " ";
        ofs << pose.at<float>(2,3) << endl;
        count ++;
    }

    ofs.close();
    nImages = count;
    cout << nImages << "images and poses have been copyed successfully." << endl;


    // produce pairs
    count = 0;
    ofs.open((string(argv[2]) + "/pairs.txt").c_str());
    for (int i = 0; i < nImages - 1; ++i)
    {
        for(int j = i + 1; j < nImages; j++){
            ofs << i <<" " << j << " ";
            Mat pose = vPoses[j].inv() * vPoses[i];
            ofs << pose.at<float>(0,0) << " ";
            ofs << pose.at<float>(0,1) << " ";
            ofs << pose.at<float>(0,2) << " ";
            ofs << pose.at<float>(0,3) << " ";
            ofs << pose.at<float>(1,0) << " ";
            ofs << pose.at<float>(1,1) << " ";
            ofs << pose.at<float>(1,2) << " ";
            ofs << pose.at<float>(1,3) << " ";
            ofs << pose.at<float>(2,0) << " ";
            ofs << pose.at<float>(2,1) << " ";
            ofs << pose.at<float>(2,2) << " ";
            ofs << pose.at<float>(2,3) << endl;
            count ++;
        }
    }
    ofs.close();
    assert(count == nImages * (nImages - 1) / 2);
    cout << count << "pairs are saved."<<endl;

    // write calibrations
    ofs.open((string(argv[2]) + "/camera.txt").c_str());
    ofs << "535.4 0 320.1 "<<endl;
    ofs << "0 539.2 247.6 "<<endl;
    ofs << "0 0 1"; 
    ofs.close();
    cout << "camera.txt include camera fr3' internal parameters."<<endl;

    return 0;
}




void LoadPoses(const string &strFile, vector<Mat> &vPosesAll){
    ifstream f;
    f.open(strFile.c_str());
    vPosesAll.reserve(10000);

    while(!f.eof())
    {
        Mat temp = Mat::eye(4,4,CV_32FC1);
        float *data = (float *)temp.data;
        for(int i=0; i<12; i++){
            f >> *data++;
        }
        vPosesAll.push_back(temp);
    }
    f.close();

    Mat a = vPosesAll[vPosesAll.size()-1] - Mat::eye(4,4,CV_32FC1);
    if(sum(abs(a))[0] < 0.1){
        vPosesAll.pop_back();
    }
 
}