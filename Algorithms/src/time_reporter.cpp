// This file evaluate the speed of feature matchers
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
#include <chrono>
#include "gms_matcher.h"
#include <opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace std;


void LoadPairs(const string &strFile, vector<Vec2i> &pairs);

void LoadCalibrations(const string &strCalibration, Mat &K);



int main(int argc, char **argv) {

	if (argc != 3)
	{
		cerr << endl << "Usage: ./time_reporter matchers dataset_dir" << endl;
		cerr << "Matchers : sift, surf, orb, akaze, brisk, kaze, dlco, latch, freak, daisy, binboost, msd, star, gms, gms_s" << endl;
		return 1;
	}

	string strDataset = string(argv[2]);

	string strFile = strDataset + "/pairs.txt";
	vector<Vec2i> pairs;
	LoadPairs(strFile, pairs);

	Mat K;
	string strCalibration = strDataset + "/camera.txt";
	LoadCalibrations(strCalibration, K);

	vector<int> ninliers(pairs.size());
	vector<Mat> vPoses(pairs.size());

	int nPairs = 200;
	assert(pairs.size() > nPairs);

	string strImagePath = strDataset + "/Images";
	string matchers = string(argv[1]);

	double totaltime1 = 0;
	double totaltime2 = 0;
	double totaltime3 = 0;
	double featurenumbers = 0;
	for (size_t i = 0; i < nPairs; ++i)
	{
		int l = pairs[i][0];
		int r = pairs[i][1];

		char bufferl[50];  sprintf(bufferl, "/%04d.png", l);
		char bufferr[50];  sprintf(bufferr, "/%04d.png", r);

		Mat img1 = imread(strImagePath + string(bufferl), 0);
		Mat img2 = imread(strImagePath + string(bufferr), 0);

		Ptr<Feature2D> extractor;
		Ptr<Feature2D> detector, descriptor;
		Ptr<DescriptorMatcher> nn_matcher;
		int flag = 0;

		if (matchers == string("sift"))
		{
			extractor = xfeatures2d::SIFT::create();
			nn_matcher = FlannBasedMatcher::create();
		}
		if (matchers == string("surf"))
		{
			extractor = xfeatures2d::SURF::create();
			nn_matcher = FlannBasedMatcher::create();
		}
		if (matchers == string("orb"))
		{
			extractor = ORB::create(100000);
			nn_matcher = BFMatcher::create(NORM_HAMMING);
		}
		if (matchers == string("akaze"))
		{
			extractor = AKAZE::create();
			nn_matcher = BFMatcher::create(NORM_HAMMING);
		}
		if (matchers == string("brisk"))
		{
			extractor = BRISK::create();
			nn_matcher = BFMatcher::create(NORM_HAMMING);
		}
		if (matchers == string("kaze"))
		{
			extractor = KAZE::create();
			nn_matcher = FlannBasedMatcher::create();
		}
		if (matchers == string("gms") || matchers == string("gms_s"))
		{
			extractor = ORB::create(10000, 1.2f, 8, 31, 0, 2, 0, 31, 0);
			nn_matcher = BFMatcher::create(NORM_HAMMING);
		}

		// Descriptors
		if (matchers == string("dlco")) {
			detector = xfeatures2d::SIFT::create();
			descriptor = xfeatures2d::VGG::create(xfeatures2d::VGG::VGG_120, 1.4f, true, true, 6.75f, false);
			nn_matcher = FlannBasedMatcher::create();
			flag = 1;
		}
		if (matchers == string("latch")) {
			detector = xfeatures2d::SIFT::create();
			descriptor = cv::xfeatures2d::LATCH::create();
			nn_matcher = BFMatcher::create(NORM_HAMMING);
			flag = 1;
		}
		if (matchers == string("freak")) {
			detector = xfeatures2d::SURF::create();
			descriptor = cv::xfeatures2d::FREAK::create();
			nn_matcher = BFMatcher::create(NORM_HAMMING);
			flag = 1;
		}
		if (matchers == string("daisy")) {
			detector = xfeatures2d::SIFT::create();
			descriptor = xfeatures2d::DAISY::create();
			nn_matcher = FlannBasedMatcher::create();
			flag = 1;
		}
		if (matchers == string("binboost")) {
			detector = xfeatures2d::SIFT::create();
			descriptor = xfeatures2d::BoostDesc::create(302, true, 6.75);
			nn_matcher = BFMatcher::create(NORM_HAMMING);
			flag = 1;
		}
		// detectors
		if (matchers == string("msd")) {
			detector = cv::xfeatures2d::MSDDetector::create();
			descriptor = ORB::create();
			nn_matcher = BFMatcher::create(NORM_HAMMING);
			flag = 1;
		}
		if (matchers == string("star")) {
			detector = cv::xfeatures2d::StarDetector::create();
			descriptor = xfeatures2d::SURF::create(100, 4, 3, false, true);
			nn_matcher = FlannBasedMatcher::create();
			flag = 1;
		}


		// extract features
		vector<KeyPoint> kp1, kp2;
		Mat d1, d2;

		std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

		if (!flag) {
			extractor->detectAndCompute(img1, Mat(), kp1, d1);
			extractor->detectAndCompute(img2, Mat(), kp2, d2);
		}
		else {
			detector->detect(img1, kp1);
			descriptor->compute(img1, kp1, d1);
			detector->detect(img2, kp2);
			descriptor->compute(img2, kp2, d2);
		}
		std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

		double ttrack1 = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
		totaltime1 += ttrack1;

		featurenumbers = featurenumbers + kp1.size() + kp2.size();

		// nearest-neighbor matching
		vector<DMatch> vMatches, vMatchesAll;
		vector<vector<DMatch> > vMatchesKnn;

		std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();

		// correspondence selection
		if (matchers == string("gms") || matchers == string("gms_s")) {
			nn_matcher->match(d1, d2, vMatchesAll);
			vMatches.reserve(vMatchesAll.size());

			// GMS filter
			vector<bool> vbInliers;
			gms_matcher gms(kp1, img1.size(), kp2, img2.size(), vMatchesAll);
			int num_inliers = 0;

			if (matchers == string("gms")) {
				num_inliers = gms.GetInlierMask(vbInliers, false, false);
			}
			else
			{
				num_inliers = gms.GetInlierMask(vbInliers, true, false);
			}

			// collect
			for (size_t i = 0; i < vbInliers.size(); ++i)
			{
				if (vbInliers[i] == true)
				{
					vMatches.push_back(vMatchesAll[i]);
				}
			}
		}
		else
		{
			// ratio 
			nn_matcher->knnMatch(d1, d2, vMatchesKnn, 2);
			vMatches.reserve(vMatchesKnn.size());

			for (size_t i = 0; i < vMatchesKnn.size(); ++i)
			{
				if (vMatchesKnn[i][0].distance < vMatchesKnn[i][1].distance * 0.8) {
					vMatches.push_back(vMatchesKnn[i][0]);
				}
			}
		}

		std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
		double ttrack2 = std::chrono::duration_cast<std::chrono::duration<double> >(t4 - t3).count();
		totaltime2 += ttrack2;

		// geometry verification
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

		std::chrono::steady_clock::time_point t5 = std::chrono::steady_clock::now();

		Mat E = findEssentialMat(vp1, vp2, K);

		std::chrono::steady_clock::time_point t6 = std::chrono::steady_clock::now();
		double ttrack3 = std::chrono::duration_cast<std::chrono::duration<double> >(t6 - t5).count();
		totaltime3 += ttrack3;

		if (E.empty())
			return 0;

		Mat R, t;
		int ninliers = recoverPose(E, vp1, vp2, K, R, t);
	}

	cout << "Feature Numbers : " << featurenumbers / nPairs / 2 << endl;
	cout << "Feature Extraction: " << (totaltime1 / nPairs / 2) << endl;
	cout << "Full Matching: " << (totaltime2 / nPairs) << endl;
	cout << "RANSAC : " << (totaltime3 / nPairs) << endl;

	return 0;
}


void LoadPairs(const string &strFile, vector<Vec2i> &pairs) {
	ifstream f;
	f.open(strFile.c_str());

	pairs.reserve(10000);
	while (!f.eof())
	{
		string s;
		getline(f, s);
		if (!s.empty())
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

void LoadCalibrations(const string &strCalibration, Mat &K) {
	K = Mat::eye(3, 3, CV_64FC1);

	ifstream f;
	f.open(strCalibration.c_str());

	double *data = (double *)K.data;

	while (!f.eof())
	{
		string s;
		getline(f, s);
		if (!s.empty())
		{
			stringstream ss;
			ss << s;

			double c;
			for (int j = 0; j<3; ++j) {
				ss >> c;
				*data++ = c;
			}
		}
	}
	f.close();
}

