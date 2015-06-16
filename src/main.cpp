#include "main.h"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/video/background_segm.hpp>

using namespace std;
using namespace cv;

/*****************************************/
/*********** GLOBAL VARIABLES ************/
/*****************************************/

int DIC_SIZE = 500;
static const string INTERACTION_TYPES[4] = {"Kiss", "HandShake", "HighFive", "Hug"};

Ptr<FeatureDetector> siftDetector = FeatureDetector::create("SIFT");
Ptr<DescriptorExtractor> siftExtractor = DescriptorExtractor::create("SIFT");
Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
BOWKMeansTrainer bowTrainer(DIC_SIZE, TermCriteria(CV_TERMCRIT_ITER, 100, 0.001), 1, KMEANS_PP_CENTERS);
BOWImgDescriptorExtractor bowDE(siftExtractor, matcher);

/*****************************************/
/*************** UTILITIES ***************/
/*****************************************/

string type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

void printMat(cv::Mat mat)
{
	string ty = type2str(mat.type());
	printf("Matrix: %s %dx%d \n", ty.c_str(), mat.cols, mat.rows);
}

string createPath(const string& path, int type, int num)
{
    string dirname = to_string(type+1) + "_" + INTERACTION_TYPES[type];

    stringstream ss;
    ss << setw(3) << setfill('0') << num+1;
    string numstr = ss.str();
    string filename = INTERACTION_TYPES[type] + "_" + numstr + ".avi";
    string filepath = path + PATH_SEPARATOR + dirname + PATH_SEPARATOR + filename;
    
    return filepath;
}

/*****************************************/
/************ DATA STRUCTURES ************/
/*****************************************/

struct Classifier {
    unique_ptr<CvSVM> svm;
};

/*****************************************/
/************* SAVE / LOAD ***************/
/*****************************************/

bool persistClassifier(const Classifier& c, const string& filepath) {
    string p = filepath + PATH_SEPARATOR + "svm.xml";
    c.svm->save(p.c_str());
    return true;
}

Classifier loadClassifier(const string& path, const string& filepath) {
    Classifier c;
    c.svm = unique_ptr<CvSVM>(new CvSVM());
    string p = filepath + PATH_SEPARATOR + "svm.xml";
    c.svm->load(p.c_str());
    return c;
}

/*****************************************/
/************** PROCESSING ***************/
/*****************************************/

void processVideo(const string& filepath, const function<void (const Mat&)>& f)
{
    cout << "Processing " << filepath << endl;

    Size targetSize(320,240);
    VideoCapture video;
    video.open(filepath);
    if (!video.isOpened()) {
        throw "Video not opened!";
    }
    
    Mat frame;
    Mat features;
    
    int framecounter = 0;
    int fps = video.get(CV_CAP_PROP_FPS);
    for(int i = 0 ; i < 50*fps ; i = i+(fps*0.5))
    {
        video.set(CV_CAP_PROP_POS_FRAMES, i);
        video >> frame;
        if (frame.empty()) {
            break;
        }
        resize(frame, frame, targetSize);
        cvtColor(frame, frame, CV_BGR2GRAY);
        
        f(frame);
        framecounter++;
    }
    //cout << framecounter << endl;
}

/*****************************************/
/***************** CORE ******************/
/*****************************************/

Classifier train(const Mat& data, const Mat& labels)
{
    CvSVMParams params;
    params.svm_type    = SVM::C_SVC;
    params.C           = 312.5;
    params.kernel_type = SVM::RBF;
    params.gamma       = 0.50625;
    params.term_crit   = TermCriteria(CV_TERMCRIT_ITER, 100, 0.000001);

    Classifier c;
    c.svm = unique_ptr<CvSVM>(new CvSVM());

    c.svm->train(data, labels, Mat(), Mat(), params);
    
    return c;
}

int classify(const Classifier& c, const string& filepath)
{
    map<int, int> classVoting;
    
    function<void (const Mat&)> f = [&c, &classVoting](const Mat& frame)
    {
        vector<KeyPoint> keypoints;
        siftDetector->detect(frame, keypoints);
        Mat bowDescriptor;
        bowDE.compute(frame, keypoints, bowDescriptor);
        
        if (!bowDescriptor.empty())
        {
            float l = c.svm->predict(bowDescriptor);
            classVoting[(int)l]++;
        }
    };
    processVideo(filepath, f);
    
    int maxValue = 0;
    int maxLabel = 0;
    for (auto it = classVoting.begin(); it != classVoting.end(); it++) {
        if (it->second > maxValue) {
            maxLabel = it->first;
            maxValue = it->second;
        }
        cout << "Label " << it->first << " has " << maxValue << " votes." << endl;
    }
    return maxLabel;
}


float performCrossValidation(string path, int numLeaveOut)
{
    Mat trainingData(0, DIC_SIZE, CV_32FC1);
    Mat trainingLabels(0, 1, CV_32FC1);

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 45-numLeaveOut; j++)
        {
            string filepath = createPath(path, i, j);
            
            function<void (const Mat&)> f = [&i, &trainingData, &trainingLabels](const Mat& frame) {
                vector<KeyPoint> keypoints;
                siftDetector->detect(frame, keypoints);
                Mat bowDescriptor;
                bowDE.compute(frame, keypoints, bowDescriptor);
                
                if (!bowDescriptor.empty()) {
                    trainingLabels.push_back((float)i);
                    trainingData.push_back(bowDescriptor);
                }
            };
            processVideo(filepath, f);
        }
    
    Classifier c = train(trainingData, trainingLabels);
    
    int correct_guesses = 0;
    
    for (int i = 0; i < 4; i++)
        for (int j = 45-numLeaveOut; j < 45; j++)
        {
            string filepath = createPath(path, i, j);
            
            int l = classify(c, filepath);
            if (l == i) {
                correct_guesses++;
            }
        }
    cout << "Precision: " << (float)correct_guesses/numLeaveOut/4 << endl;
    
    return -1;
}

void collectCentroidsForVideo(string filepath)
{
    function<void (const Mat&)> f = [](const Mat& frame)
    {
        vector<KeyPoint> keypoints;
        siftDetector->detect(frame, keypoints);
        Mat descriptors;
        siftExtractor->compute(frame, keypoints, descriptors);
        if (!descriptors.empty()) {
            bowTrainer.add(descriptors);
        }
    };
    processVideo(filepath, f);
}

void collectCentroids(string path)
{
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
        {
            string filepath = createPath(path, i, j);
            collectCentroidsForVideo(filepath);
        }
    
    vector<Mat> descriptors = bowTrainer.getDescriptors();
    
    Mat dictionary = bowTrainer.cluster();
    bowDE.setVocabulary(dictionary);
}

int main(int argc, char** argv)
{
    collectCentroids(argv[1]);
    performCrossValidation(argv[1], 5);

	return -1;
}









