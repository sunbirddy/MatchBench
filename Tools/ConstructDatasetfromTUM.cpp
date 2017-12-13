// This file construct our datasets from TUM benchmark sequences.
// Input :
// 	1. input directory (one TUM sequence)
// 	2. output directory
// Requirement :
//	You need get associated files "rgb_gt.txt" (rgb.txt + groundtruth.txt) with "associate.py" provided by TUM dataset.
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


void LoadImages(const string &strFile, vector<string> &vstrImageFilenames, vector<vector<float> > &vgts);


// Q : tx ty tz qx qy qz qw
// T : 3 X 4 matrix
void QconvertoT(const vector<float> &Q, Mat &T) {
    T = Mat::eye(4, 4, CV_32FC1);

    Vec3f t(Q[0], Q[1], Q[2]);
    Vec4f q(Q[3], Q[4], Q[5], Q[6]);

    T.at<float>(0, 3) = t[0];
    T.at<float>(1, 3) = t[1];
    T.at<float>(2, 3) = t[2];

    float nq = q.ddot(q);
    if (nq < 0.00000000001)     return;

    q *= sqrt(2.0 / nq);
    Mat q_m = Mat(4, 1, CV_32FC1, &q[0]);
    Mat q_outer = q_m * q_m.t();

    // first row
    T.at<float>(0, 0) = 1.0f - q_outer.at<float>(1, 1) - q_outer.at<float>(2, 2);
    T.at<float>(0, 1) = q_outer.at<float>(0, 1) - q_outer.at<float>(2, 3);
    T.at<float>(0, 2) = q_outer.at<float>(0, 2) + q_outer.at<float>(1, 3);

    // second row
    T.at<float>(1, 0) = q_outer.at<float>(0, 1) + q_outer.at<float>(2, 3);
    T.at<float>(1, 1) = 1.0f - q_outer.at<float>(0, 0) - q_outer.at<float>(2, 2);
    T.at<float>(1, 2) = q_outer.at<float>(1, 2) - q_outer.at<float>(0, 3);

    // third row
    T.at<float>(2, 0) = q_outer.at<float>(0, 2) - q_outer.at<float>(1, 3);
    T.at<float>(2, 1) = q_outer.at<float>(1, 2) + q_outer.at<float>(0, 3);
    T.at<float>(2, 2) = 1.0f - q_outer.at<float>(0, 0) - q_outer.at<float>(1, 1);
}

int main(int argc, char **argv){

	if(argc != 3)
    {
        cerr << endl << "Usage: ./ConstructDatasetfromTUM InputDir OutputDir" << endl;
        return 1;
    }

    // Retrieve image names and groundtruth
    vector<string> vstrImageFilenames;
    vector<vector<float> > vgts;
    string rgbfile = string(argv[1]) + "/rgb_gt.txt";
    LoadImages(rgbfile, vstrImageFilenames, vgts);

    int nImages = vstrImageFilenames.size();

    string output = string(argv[2]) + "/Images";

#ifdef WIN32
	mkdir(output.c_str());
#elif __linux__
	mkdir(output.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#endif

    ofstream ofs;
    ofs.open((output + "/pose.txt").c_str());

    vector<Mat> vPoses(nImages);

	for (int i = 0; i < nImages; ++i)
	{
		Mat img = imread(string(argv[1]) + vstrImageFilenames[i]);
		char buffer[50];
		sprintf(buffer, "/%04d.png", i);
		imwrite(output + buffer, img);

		Mat pose;
		QconvertoT(vgts[i], pose);
		ofs << pose.at<float>(0, 0) << " ";
		ofs << pose.at<float>(0, 1) << " ";
		ofs << pose.at<float>(0, 2) << " ";
		ofs << pose.at<float>(0, 3) << " ";
		ofs << pose.at<float>(1, 0) << " ";
		ofs << pose.at<float>(1, 1) << " ";
		ofs << pose.at<float>(1, 2) << " ";
		ofs << pose.at<float>(1, 3) << " ";
		ofs << pose.at<float>(2, 0) << " ";
		ofs << pose.at<float>(2, 1) << " ";
		ofs << pose.at<float>(2, 2) << " ";
		ofs << pose.at<float>(2, 3) << endl;

		vPoses[i] = pose.clone();
	}
    ofs.close();
    cout << nImages << "images and poses have been copyed successfully." << endl;

    // produce pairs
    ofs.open((string(argv[2]) + "/pairs.txt").c_str());
    int npersets = 15;
    int ncount = 0;
    for (int i = 0; i < nImages; ++i)
    {
        if(i % npersets == 0)
            continue;

        int l = i / npersets * npersets;
        int r = i;

        ofs << l << "   " << r << " ";

        Mat pose = vPoses[r].inv() * vPoses[l];
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

        ncount++;
    }

    ofs.close();
    cout << ncount << "pairs are saved."<<endl;

    // write calibrations
    ofs.open((string(argv[2]) + "/camera.txt").c_str());
    ofs << "535.4 0 320.1 "<<endl;
    ofs << "0 539.2 247.6 "<<endl;
    ofs << "0 0 1"; 
    ofs.close();
    cout << "camera.txt include camera fr3' internal parameters."<<endl;

    return 0;
}


void LoadImages(const string &strFile, vector<string> &vstrImageFilenames, vector<vector<float> > &vgts){
	ifstream f;
    f.open(strFile.c_str());

    vstrImageFilenames.reserve(10000);
    vgts.reserve(10000);

    while(!f.eof())
    {
        string s;
        getline(f,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;

            double t;
            ss >> t;
            
            string filename;
            ss >> filename;
			vstrImageFilenames.push_back(filename);

            ss >> t;

            vector<float> vgt(7);
            for(int i=0; i<7; i++){
            	ss >> vgt[i];
            }
            vgts.push_back(vgt);
        }
    }

    f.close();
}



