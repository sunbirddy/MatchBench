// This file contains various feature matchers implemented in OpenCV library
// Input :
// 	1. matcher's name
//  2. Dataset Directory
// Author :
//	JiaWang Bian



#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include <sys/stat.h>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace std;


void LoadPairs(const string &strFile, vector<Vec2i> &pairs);

void LoadCalibrations(const string &strCalibration, Mat &K);

int MatchImages(Mat img1, Mat img2, string matcher, Mat K, Mat &pose);


int main(int argc, char **argv){

	if(argc != 4)
    {
        cerr << endl << "Usage: ./fast_matchers matcher_name dataset_dir results_dir" << endl;
        cerr << "matcher_names : sift, surf, orb, akaze, brisk, kaze, dlco, latch, freak, daisy, binboost, msd, star " << endl;
        return 1;
    }

	string dataset = string(argv[2]);
    string strPairs = dataset + "/pairs.txt";
    vector<Vec2i> pairs;
    LoadPairs(strPairs, pairs);

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

        ninliers[i] = MatchImages(img1, img2, string(argv[1]), K, vPoses[i]);
        cout << i << "/"<< pairs.size() << "\r";
    }

    // write results
    string ResultsDir = string(argv[3]);
    string outputfiles =  ResultsDir + "/" + string(argv[1]) + ".txt";
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

int MatchImages(Mat img1, Mat img2, string matcher, Mat K, Mat &pose){

	Ptr<Feature2D> detector, descriptor;
    Ptr<DescriptorMatcher> nn;

	// combined features
    if (matcher == string("sift"))
    {
		detector = xfeatures2d::SIFT::create();
		descriptor = detector;
        nn = FlannBasedMatcher::create();
    }
    if (matcher == string("surf"))
    {
		detector = xfeatures2d::SURF::create();
		descriptor = detector;
        nn = FlannBasedMatcher::create();
    }
    if (matcher == string("orb"))
    {
		detector = ORB::create(100000);
		descriptor = detector;
        nn = BFMatcher::create(NORM_HAMMING);
    }
    if (matcher == string("akaze"))
    {
		detector = AKAZE::create();
		descriptor = detector;
        nn = BFMatcher::create(NORM_HAMMING);
    }
    if (matcher == string("kaze")){
		detector = KAZE::create();
		descriptor = detector;
        nn = FlannBasedMatcher::create();
    }
    if (matcher == string("brisk")){
		detector = BRISK::create();
		descriptor = detector;
        nn = BFMatcher::create(NORM_HAMMING);
    }

	// SIFT + Descriptors
	if (matcher == string("dlco")) {
		detector = xfeatures2d::SIFT::create();
		descriptor = xfeatures2d::VGG::create(xfeatures2d::VGG::VGG_120, 1.4f, true, true, 6.75f, false);
		nn = FlannBasedMatcher::create();
	}
	if (matcher == string("latch")) {
		detector = xfeatures2d::SIFT::create();
		descriptor = cv::xfeatures2d::LATCH::create();
		nn = BFMatcher::create(NORM_HAMMING);
	}
	if (matcher == string("freak")) {
		detector = xfeatures2d::SIFT::create();
		descriptor = cv::xfeatures2d::FREAK::create();
		nn = BFMatcher::create(NORM_HAMMING);
	}
	if (matcher == string("daisy")) {
		detector = xfeatures2d::SIFT::create();
		descriptor = xfeatures2d::DAISY::create();
		nn = FlannBasedMatcher::create();
	}
	if (matcher == string("binboost")) {
		detector = xfeatures2d::SIFT::create();
		descriptor = xfeatures2d::BoostDesc::create(302,true,6.75);
		nn = BFMatcher::create(NORM_HAMMING);
	}

	// detectors 
	if (matcher == string("msd")) {
		detector = cv::xfeatures2d::MSDDetector::create();
		descriptor = ORB::create();
		nn = BFMatcher::create(NORM_HAMMING);
	}
	// star + surf
	if (matcher == string("star")) {
		detector = cv::xfeatures2d::StarDetector::create();
		descriptor = xfeatures2d::SURF::create(100, 4, 3, false, true);
		nn = FlannBasedMatcher::create();
	}

	// extract features
	vector<KeyPoint> kp1, kp2;
	Mat d1, d2;
	detector->detect(img1, kp1);
	descriptor->compute(img1, kp1, d1);
	detector->detect(img2, kp2);
	descriptor->compute(img2, kp2, d2);

    // nearest-neighbor matching
    vector<DMatch> vMatches;
    vector<vector<DMatch> > vMatchesKnn;

    // correspondence selection (ratio < 0.8)
    nn->knnMatch(d1,d2,vMatchesKnn,2);
    vMatches.reserve(vMatchesKnn.size());

    for (size_t i = 0; i < vMatchesKnn.size(); ++i)
    {
        if(vMatchesKnn[i][0].distance < vMatchesKnn[i][1].distance * 0.8){
            vMatches.push_back(vMatchesKnn[i][0]);
        }
    }

    // geometry verification
    pose = Mat::eye(4,4,CV_64FC1);

    if (vMatches.size() < 20)
        return 0;

    vector<Point2f> vp1(vMatches.size()), vp2(vMatches.size());
    for (size_t i = 0; i < vMatches.size(); ++i)
    {
        vp1[i].x = kp1[vMatches[i].queryIdx].pt.x;
        vp1[i].y = kp1[vMatches[i].queryIdx].pt.y;
    
        vp2[i].x = kp2[vMatches[i].trainIdx].pt.x;
        vp2[i].y = kp2[vMatches[i].trainIdx].pt.y;
    }

    Mat E = findEssentialMat(vp1, vp2, K);

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
