#include "main.h"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/video/background_segm.hpp>

using namespace std;
using namespace cv;

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

/*****************************************/
/************ DATA STRUCTURES ************/
/*****************************************/

struct Classifier {
    unique_ptr<CvSVM> svm;
};

/*****************************************/
/************* SAVE / LOAD ***************/
/*****************************************/

bool persistClassifier(const Classifier& c, const string& path, const string& filename) {
    return false;
}

Classifier loadClassifier(const string& path, const string& filename) {
    Classifier c;
    return c;
}

/*****************************************/
/***************** CORE ******************/
/*****************************************/

Mat extractFeaturesFromVideo(string filepath, int label)
{
    cout << "Extracting features from: " << filepath << endl;

    Size targetSize(320,240);

    VideoCapture video;
    video.open(filepath);
    
    if (!video.isOpened()) {
        throw "Video not opened!";
    }
    
    Mat frame;
    video >> frame;
    if (frame.empty()) {
        throw "Empty video?!";
    }
    resize(frame, frame, targetSize);
    frame.convertTo(frame, CV_32FC3);

    Mat f = Mat::zeros(2, 1, CV_32FC1);
    f.at<float>(0) = (float) (label%2);
    f.at<float>(1) = (float) (label/2);
    
	return f;
}

Classifier train(const Mat& data, const Mat& labels)
{
    CvSVMParams params;
    params.svm_type    = SVM::C_SVC;
    params.C           = 0.1;
    params.kernel_type = SVM::RBF;
    params.term_crit   = TermCriteria(CV_TERMCRIT_ITER, (int)1e7, 1e-6);

    Classifier c;
    c.svm = unique_ptr<CvSVM>(new CvSVM());

    cout << "Starting training process" << endl;
    c.svm->train(data, labels, Mat(), Mat(), params);
    cout << "Finished training process" << endl;

    return c;
}

int classify(const Classifier& c, const Mat& f)
{
    float l = c.svm->predict(f);
    cout << "Predicting " << l << endl;
    return l;
}

static const string INTERACTION_TYPES[4] = {"Kiss", "HandShake", "HighFive", "Hug"};

float performCrossValidation(string path, int numLeaveOut)
{
    Mat trainingData = Mat::zeros((45-numLeaveOut) * 4, 2, CV_32FC1);
    Mat trainingLabels = Mat::zeros((45-numLeaveOut) * 4, 1, CV_32FC1);

    for (int i = 0; i < 4; i++)
    {
        string dirname = to_string(i+1) + "_" + INTERACTION_TYPES[i];
        
        for (int j = 0; j < 45-numLeaveOut; j++) {
            stringstream ss;
            ss << setw(3) << setfill('0') << j+1;
            string numstr = ss.str();
            string filename = INTERACTION_TYPES[i] + "_" + numstr + ".avi";
            string filepath = path + PATH_SEPARATOR + dirname + PATH_SEPARATOR + filename;
            
            Mat f = extractFeaturesFromVideo(filepath, i);
            trainingData.at<float>(i*(45-numLeaveOut)+j, 0) = f.at<float>(0);
            trainingData.at<float>(i*(45-numLeaveOut)+j, 1) = f.at<float>(1);
            trainingLabels.at<float>(i*(45-numLeaveOut)+j) = (float)i;
        }
    }
    
    cout << trainingData << endl;
    Classifier c = train(trainingData, trainingLabels);
    
    int correct_guesses = 0;
    
    for (int i = 0; i < 4; i++)
    {
        string dirname = to_string(i+1) + "_" + INTERACTION_TYPES[i];
        
        for (int j = 45-numLeaveOut; j < 45; j++) {
            stringstream ss;
            ss << setw(3) << setfill('0') << j+1;
            string numstr = ss.str();
            string filename = INTERACTION_TYPES[i] + "_" + numstr + ".avi";
            string filepath = path + PATH_SEPARATOR + dirname + PATH_SEPARATOR + filename;
            
            Mat f = extractFeaturesFromVideo(filepath, i);
            int l = classify(c, f);
            if (l == i) {
                correct_guesses++;
            }
        }
    }
    cout << "Precision: " << (float)correct_guesses/numLeaveOut/4 << endl;
    
    return -1;
}

int main(int argc, char** argv)
{
    performCrossValidation("/Users/markomlinaric/Desktop/training", 40);

    // Suggested usage:
    // videoanalysis [path/to/test/dir] [-train path/to/training/dir]
	return -1;
}









