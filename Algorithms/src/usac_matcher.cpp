// This file implements SIFT with USAC matcher
// Input :
// 	1. matcher's name
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

template <typename T>
vector<size_t> sort_indexes(const vector<T> &v) {

	// initialize original index locations
	vector<size_t> idx(v.size());
	iota(idx.begin(), idx.end(), 0);

	// sort indexes based on comparing values in v
	std::sort(idx.begin(), idx.end(),
		[&v](size_t i1, size_t i2) {return v[i1] < v[i2]; });

	return idx;
}

void LoadPairs(const string &strFile, vector<Vec2i> &pairs);

void LoadCalibrations(const string &strCalibration, Mat &K);

int MatchImages(Mat img1, Mat img2, string usac_dir, Mat K, Mat &pose);



int main(int argc, char **argv){

	if(argc != 4)
    {
        cerr << endl << "Usage: ./usac_matcher usac_dir dataset_dir, results_dir" << endl;
        return 1;
    }

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

        ninliers[i] = MatchImages(img1, img2, string(argv[1]), K, vPoses[i]);
        cout << i << "/"<< pairs.size() << "\r";
    }

    // write results
    string ResultsDir = string(argv[3]);
    string outputfiles =  ResultsDir + "/usac_fm.txt";
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


int MatchImages(Mat img1, Mat img2, string usac_dir, Mat K, Mat &pose){

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

	//vector<DMatch> vMatches;
	//nn_matcher->match(d1,d2,vMatches);
	//vector<float> vDistance(vMatches.size());
	//for (size_t i = 0; i < vMatches.size(); i++)
	//{
	//	vDistance[i] = vMatches[i].distance;
	//}
	//vector<size_t> indexes = sort_indexes(vDistance);

	// nearest-neighbor matching
	vector<DMatch> vMatches;
	vector<vector<DMatch> > vMatchesKnn;
	vector<float> vRatio;
	vRatio.reserve(kp1.size());

	// correspondence selection (ratio < 0.8)
	nn_matcher->knnMatch(d1, d2, vMatchesKnn, 2);
	vMatches.reserve(vMatchesKnn.size());

	for (size_t i = 0; i < vMatchesKnn.size(); ++i)
	{
		float ratio = vMatchesKnn[i][0].distance / vMatchesKnn[i][1].distance;
		if (ratio < 0.8) {
			vMatches.push_back(vMatchesKnn[i][0]);
			vRatio.push_back(ratio);
		}
	}
	vector<size_t> indexes = sort_indexes(vRatio);

	pose = Mat::eye(4, 4, CV_64FC1);
	if (vMatches.size() < 20)
		return 0;

	// write matching
	string output1 = usac_dir + "orig_pts.txt";
	string output2 = usac_dir + "sorting.txt";

	ofstream ofs;
	ofs.open(output1.c_str(), ofstream::out);
	ofs << (int) vMatches.size() << endl;
	for(int i = 0; i < vMatches.size(); ++i){
		Point2f &p1 = kp1[vMatches[i].queryIdx].pt;
		Point2f &p2 = kp2[vMatches[i].trainIdx].pt;
		ofs << p1.x << " " << p1.y << " " << p2.x << " " << p2.y << endl;
	}
	ofs.close();
	
	ofs.open(output2.c_str(), ofstream::out);
	for(int i = 0; i < vMatches.size(); ++i){
		ofs << indexes[i] << endl;
	}
	ofs.close();
	
	// excute usac
	string cmd = usac_dir + "RunSingleTest.exe 0 " + usac_dir + "fundmatrix.cfg";
	system(cmd.c_str());
	

	// read fundamental matrix and inliers
	string maskfile = usac_dir + "inliers.txt";
	string fundfile = usac_dir + "F.txt";
	
	Mat Fundmatrix = Mat::zeros(3,3,CV_64FC1);
	double *dataF = (double*)Fundmatrix.data;
	vector<int> mask(vMatches.size());
	int ninliers = 0;
	
	ifstream ifs;
	ifs.open(maskfile);
	for(int i = 0 ;i < vMatches.size(); ++i){
		ifs >> mask[i];
		if(mask[i] == 1){
			ninliers++;
		}
	}
	ifs.close();
	
	if (ninliers < 10)
		return 0;

	ifs.open(fundfile);
	for(int i = 0 ;i < 9; ++i){
		ifs >> dataF[i];
	}
	ifs.close();
 
	Mat E = K.t() * Fundmatrix * K; 
 
    vector<Point2f> vp1, vp2;
    for (size_t i = 0; i < vMatches.size(); ++i)
    {
		if(mask[i] == 1){
			Point2f p1 = kp1[vMatches[i].queryIdx].pt;
			Point2f p2 = kp2[vMatches[i].trainIdx].pt;
			vp1.push_back(p1);
			vp2.push_back(p2);
		}
    }

    if (E.empty())
        return 0;

    Mat R, t;
    ninliers = recoverPose(E, vp1, vp2, K, R, t); 

    if(ninliers < 10)
        return 0;

    R.copyTo(pose(Rect(0, 0, 3, 3)));
    t.copyTo(pose(Rect(3, 0, 1, 3)));

    return ninliers;
}



