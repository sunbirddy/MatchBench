// This file implements SIFT matcher with many RANSAC algorithms
// Input :
// 	1. ransac's name
//  2. Dataset Directory
// Author :
//	JiaWang Bian

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include <numeric> 
#include <algorithm>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace std;


void LoadPairs(const string &strFile, vector<Vec2i> &pairs);

void LoadCalibrations(const string &strCalibration, Mat &K);

int MatchImages(Mat img1, Mat img2, string ransac, Mat K, Mat &pose);



int main(int argc, char **argv){

	if(argc != 4)
    {
        cerr << endl << "Usage: ./opencv_ransac method_name dataset_dir results_dir" << endl;
		cerr << endl << "method_name: ransac_fm lmeds_fm lmeds_em" << endl;
        return 1;
    }

	string ransac = string(argv[1]);
	assert(ransac == string("ransac_fm") || ransac == string("lmeds_fm") || ransac == string("lmeds_em"));


	string dataset = string(argv[2]);
    string strFile = dataset + "/pairs.txt";
    vector<Vec2i> pairs;
    LoadPairs(strFile, pairs);

    Mat K;
    string strCalibration = dataset + "/camera.txt";
    LoadCalibrations(strCalibration, K);

    vector<int> ninliers(pairs.size());
    vector<Mat> vPoses(pairs.size());

    string strImagePath = dataset + "/Images";
    for (size_t i = 0; i < pairs.size(); ++i)
    {
        int l = pairs[i][0];
        int r = pairs[i][1];

        char bufferl[50];  sprintf(bufferl, "/%04d.png", l);
        char bufferr[50];  sprintf(bufferr, "/%04d.png", r);

        Mat img1 = imread(strImagePath + string(bufferl), 0);
        Mat img2 = imread(strImagePath + string(bufferr), 0);

        ninliers[i] = MatchImages(img1, img2, ransac, K, vPoses[i]);
        cout << i << "/"<< pairs.size() << "\r";
    }

    // write results
    string ResultsDir = string(argv[3]);
    string outputfiles =  ResultsDir + "/" +string(argv[1]) + ".txt";
    ofstream ofs;
    ofs.open(outputfiles.c_str());
    for (size_t i = 0; i < vPoses.size(); ++i)
    {
        int l = pairs[i][0];
        int r = pairs[i][1];

        ofs << l <<" "<< r <<" "<< ninliers[i]<<" ";
        double *data = (double *)vPoses[i].data;

        for (int j = 0; j < 11; ++j)
        {
            ofs << data[j] <<" ";
        }
        ofs << data[11] <<endl;
    }
    ofs.close();

    return 0;
}


void LoadPairs(const string &strFile, vector<Vec2i> &pairs){
	ifstream f;
    f.open(strFile.c_str());

    pairs.reserve(10000);
    while(!f.eof())
    {
        string s;
        getline(f,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;

            int l, r;
            ss >> l;
            ss >> r;
            
            Vec2i p;
            p[0] = l;
            p[1] = r;
            pairs.push_back(p);
        }
    }

    f.close();
}

void LoadCalibrations(const string &strCalibration, Mat &K){
    K = Mat::eye(3,3,CV_64FC1);

    ifstream f;
    f.open(strCalibration.c_str());

    double *data = (double *)K.data;

    while(!f.eof())
    {
        string s;
        getline(f,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;

            double c;
            for (int j=0; j<3; ++j){
                ss >> c;
                *data++ = c;
            }
        }
    }
    f.close();
}


int MatchImages(Mat img1, Mat img2, string ransac, Mat K, Mat &pose){

    Ptr<Feature2D> extractor;
    Ptr<DescriptorMatcher> nn_matcher;

    extractor = xfeatures2d::SIFT::create();
    nn_matcher = FlannBasedMatcher::create();
    
    // extract features
	vector<KeyPoint> kp1, kp2;
    Mat d1, d2;

    extractor->detectAndCompute(img1,Mat(),kp1,d1);
    extractor->detectAndCompute(img2,Mat(),kp2,d2);

	// nearest-neighbor matching
	vector<DMatch> vMatches;
	vector<vector<DMatch> > vMatchesKnn;

	// correspondence selection (ratio < 0.8)
	nn_matcher->knnMatch(d1, d2, vMatchesKnn, 2);
	vMatches.reserve(vMatchesKnn.size());

	for (size_t i = 0; i < vMatchesKnn.size(); ++i)
	{
		float ratio = vMatchesKnn[i][0].distance / vMatchesKnn[i][1].distance;
		if (ratio < 0.8) {
			vMatches.push_back(vMatchesKnn[i][0]);
		}
	}

	
	pose = Mat::eye(4, 4, CV_64FC1);
	if (vMatches.size() < 20)
		return 0;

    vector<Point2f> vp1, vp2;
    for (size_t i = 0; i < vMatches.size(); ++i)
    {
		Point2f p1 = kp1[vMatches[i].queryIdx].pt;
		Point2f p2 = kp2[vMatches[i].trainIdx].pt;
		vp1.push_back(p1);
		vp2.push_back(p2);
    }
	

	Mat Fundmatrix;
	Mat E;
	
	try
	{
		if (ransac == string("ransac_fm"))
		{
			Fundmatrix = findFundamentalMat(vp1, vp2, FM_RANSAC);
			E = K.t() * Fundmatrix * K;
		}
		if (ransac == string("lmeds_fm"))
		{
			Fundmatrix = findFundamentalMat(vp1, vp2, FM_LMEDS);
			E = K.t() * Fundmatrix * K;
		}
		if (ransac == string("lmeds_em"))
		{
			E = findEssentialMat(vp1, vp2, K, LMEDS);
		}
	}
	catch (const std::exception&)
	{
		return 0;
	}
	
    if (E.empty())
        return 0;

    Mat R, t;
    int ninliers = recoverPose(E, vp1, vp2, K, R, t); 

    if(ninliers < 10)
        return 0;

    R.copyTo(pose(Rect(0, 0, 3, 3)));
    t.copyTo(pose(Rect(3, 0, 1, 3)));

    return ninliers;
}



