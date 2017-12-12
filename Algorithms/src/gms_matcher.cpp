// This file implements gms_matcher
// Input :
// 	1. dataset_dir
//  2. results_dir
//  3. match_type
// Author :
//	JiaWang Bian



#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include <sys/stat.h>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "gms_matcher.h"

using namespace cv;
using namespace std;


void LoadPairs(const string &strFile, vector<Vec2i> &pairs);

void LoadCalibrations(const string &strCalibration, Mat &K);

int MatchImages(Mat img1, Mat img2, Mat K, Mat &pose, string match_type);



int main(int argc, char **argv){

	if(argc != 3)
    {
        cerr << endl << "Usage: ./gms_matcher dataset_dir results_dir match_type" << endl;
        cerr << "match_type: single-scale or multi-scale " << endl;
		return 1;
    }

	// match_type
	string match_type = string(argv[3]);
	if(match_type == string("single-scale") && match_type == string("multi-scale"))
    {
        cerr << endl << "Usage: ./gms_matcher dataset_dir image_type match_type" << endl;
        cerr << "match_type: single-scale or multi-scale " << endl;
		return 1;
    }
	
	// dataset
	string dataset = string(argv[1]);

    string strPair = dataset + "/pairs.txt";
    vector<Vec2i> pairs;
    LoadPairs(strPair, pairs);

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

        ninliers[i] = MatchImages(img1, img2, K, vPoses[i], match_type);
 
        cout << i << "/"<< pairs.size() << "\r";
    }

    // write results
    string ResultsDir = string(argv[2]); 
    string outputfile;
	
	if (match_type == string("single-scale"))
		outputfile = ResultsDir + "/" + "gms.txt";
	else if (match_type == string("multi-scale"))
		outputfile = ResultsDir + "/" + "gms_s.txt";
	else {
		cout << "Please input correct match types: single-scale or multi-scale." << endl;
		exit(-1);
	}

    ofstream ofs;
    ofs.open(outputfile.c_str());
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

int MatchImages(Mat img1, Mat img2, Mat K, Mat &pose, string match_type){

    Ptr<ORB> extractor = ORB::create();
	int resolution = img1.rows * img1.cols;
	if(resolution <= 480 * 640){
        extractor->setMaxFeatures(10000);
        extractor->setFastThreshold(0);
    }else
	{
    	extractor->setMaxFeatures(100000);
        extractor->setFastThreshold(5);
    }

    Ptr<BFMatcher> nnmatcher = BFMatcher::create(NORM_HAMMING);

    // extract features
    vector<KeyPoint> kp1, kp2;
    Mat d1, d2;

    extractor->detectAndCompute(img1,Mat(),kp1,d1);
    extractor->detectAndCompute(img2,Mat(),kp2,d2);

    // nearest-neighbor matching
    vector<DMatch> vMatches, vMatchesAll;
    nnmatcher->match(d1,d2,vMatchesAll);
    vMatches.reserve(vMatchesAll.size());

	bool multi_scale = false;
	if(match_type == string("multi-scale"))
		multi_scale = true;
	
    // GMS filter
    vector<bool> vbInliers;
    gms_matcher gms(kp1,img1.size(), kp2, img2.size(), vMatchesAll);
    int num_inliers = gms.GetInlierMask(vbInliers, multi_scale, false);

    // collect
    for (size_t i = 0; i < vbInliers.size(); ++i)
    {
        if (vbInliers[i] == true)
        {
            vMatches.push_back(vMatchesAll[i]);
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
