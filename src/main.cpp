#include "main.h"

#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <fstream>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/video/background_segm.hpp>

using namespace std;
using namespace cv;

/*****************************************/
/*********** GLOBAL VARIABLES ************/
/*****************************************/


Ptr<FeatureDetector> siftDetector = FeatureDetector::create("SIFT");
Ptr<DescriptorExtractor> siftExtractor = DescriptorExtractor::create("SIFT");
Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
BOWKMeansTrainer bowTrainer(DIC_SIZE, TermCriteria(CV_TERMCRIT_ITER, 100, 0.001), 1, KMEANS_PP_CENTERS);
BOWImgDescriptorExtractor bowDE(siftExtractor, matcher);
CvSVM svm;

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
	string dirname = to_string(type + 1) + "_" + getTextForEnum(type);

	stringstream ss;
	ss << setw(3) << setfill('0') << num + 1;
	string numstr = ss.str();
	string filename = INTERACTION_TYPES[type] + "_" + numstr + ".avi";
	string filepath = path + PATH_SEPARATOR + dirname + PATH_SEPARATOR + filename;

	return filepath;
}

/*****************************************/
/************** PROCESSING ***************/
/*****************************************/

void processVideo(const string& filepath, const function<void(const Mat&)>& f,
        int featurefps = FEATURE_FRAMES_PER_SECOND)
{
    cout << "Processing " << filepath << endl;

	Size TARGET_SIZE(320, 240);
    VideoCapture video;
    video.open(filepath);
    if (!video.isOpened()) {
        throw "Video not opened!";
    }
    
    Mat frame;
    Mat features;
    
    int framecounter = 0;
    int fps = video.get(CV_CAP_PROP_FPS);
    for(int i = 0; i < FEATURE_MAX_FRAMES; i++)
    {
        video.set(CV_CAP_PROP_POS_FRAMES, i*(float)fps/(float)featurefps);
        video >> frame;
        if (frame.empty()) {
            break;
        }
        resize(frame, frame, TARGET_SIZE);
        cvtColor(frame, frame, CV_BGR2GRAY);
        
        f(frame);
        framecounter++;
    }
    //cout << framecounter << endl;
}

/*****************************************/
/***************** CORE ******************/
/*****************************************/

void train(const Mat& data, const Mat& labels)
{
	CvSVMParams params;
	params.svm_type = SVM::C_SVC;
	//params.C = 312.5;
	params.kernel_type = SVM::RBF;
	//params.gamma = 0.50625;
	params.term_crit = TermCriteria(CV_TERMCRIT_ITER, 100, 0.000001);

	svm.train_auto(data, labels, Mat(), Mat(), params);
	svm.save(SVM_PATH);
}

int classify(const string& filepath)
{
	map<int, int> classVoting;

	function<void(const Mat&)> f = [&classVoting](const Mat& frame)
	{
		vector<KeyPoint> keypoints;
		siftDetector->detect(frame, keypoints);
		Mat bowDescriptor;
		bowDE.compute(frame, keypoints, bowDescriptor);

		if (!bowDescriptor.empty())
		{
			float l = svm.predict(bowDescriptor);
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

mutex training_mutex;

void collectTrainData(const string& path, int i, Mat* trainingData, Mat* trainingLabels)
{

    for (int j = 0; j < 45-NUM_CROSS_VALID_LEAVE_OUT; j++)
    {
        string filepath = createPath(path, i, j);
        
        function<void (const Mat&)> f = [&i, &trainingData, &trainingLabels](const Mat& frame) {
            vector<KeyPoint> keypoints;
            siftDetector->detect(frame, keypoints);
            Mat bowDescriptor;
            bowDE.compute(frame, keypoints, bowDescriptor);
            
            if (!bowDescriptor.empty()) {
                training_mutex.lock();
                trainingLabels->push_back((float)i);
                trainingData->push_back(bowDescriptor);
                training_mutex.unlock();
            }
        };
        processVideo(filepath, f);
    }
}

float performCrossValidation(string path)
{
    Mat trainingData(0, DIC_SIZE, CV_32FC1);
    Mat trainingLabels(0, 1, CV_32FC1);

    vector<thread> ts;
    for (int i = 0; i < 4; i++)
        ts.push_back(thread(collectTrainData, path, i, &trainingData, &trainingLabels));
    
    for (auto &t : ts)
        t.join();
    
    cout << "Start training SVM" << endl;
    train(trainingData, trainingLabels);
    cout << "Finished training SVM" << endl;
    
    int correct_guesses = 0;
    
    for (int i = 0; i < 4; i++)
        for (int j =  0 /*45-NUM_CROSS_VALID_LEAVE_OUT*/; j < 45; j++)
        {
            string filepath = createPath(path, i, j);
            
            int l = classify(filepath);
            if (l == i) {
                correct_guesses++;
            }
        }
    cout << "Precision: " << (float)correct_guesses/ /*NUM_CROSS_VALID_LEAVE_OUT*/ (float)45 / (float)4 << endl;
    
    return -1;
}

void collectCentroidsForVideo(string filepath)
{
	function<void(const Mat&)> f = [](const Mat& frame)
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
        for (int j = 0; j < NUM_CLUSTER_VIDEOS; j++)
        {
            string filepath = createPath(path, i, j);
            collectCentroidsForVideo(filepath);
        }
    
    vector<Mat> descriptors = bowTrainer.getDescriptors();
    
    cout << "Start clustering ..." << endl;
    Mat dictionary = bowTrainer.cluster();
    cout << "Finished clustering ..." << endl;
    bowDE.setVocabulary(dictionary);
    
	cv::FileStorage file(DICT_PATH, cv::FileStorage::WRITE);
	file << "vocabulary" << dictionary;
    file.release();
}

String writeResult(string filename, int label)
{
	String line = filename + '\t' + getTextForEnum(label) + '\n';
	return line;
}

void evaluateTestSet(){
	ofstream myfile;
	String filepath = TEST_PATH;
	myfile.open("result.txt");
	for (int i = 1; i < 30; i++)
	{
		ostringstream ss;
		ss << setw(3) << setfill('0') << i;
		string str_i(ss.str());
		filepath = TEST_PATH + String(PATH_SEPARATOR) +  str_i + ".avi";
		if (FILE *file = fopen(filepath.c_str(), "r")) {
			int l = classify(filepath);
			String res = writeResult(filepath, l);
			myfile << res;
		}
		else
		{
			break;
		}
	}
	myfile.close();
}
int main(int argc, char** argv)
{
	cv::initModule_nonfree();
	cv::initModule_features2d();
	cv::initModule_ml();

	if (TRAIN_MODE)
	{
		collectCentroids(argv[1]);
		performCrossValidation(argv[1]);
	}
	else{
		svm.load(SVM_PATH);
		evaluateTestSet();
	}
	waitKey();
}







