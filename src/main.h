#if defined(WIN32) || defined(_WIN32)
#define PATH_SEPARATOR "\\" 
#else 
#define PATH_SEPARATOR "/" 
#endif
#define TRAIN_MODE 1
#define SVM_PATH "svm.xml"
#define DICT_PATH "dictionary.xml"
#include <string>
#define TEST_PATH "test"
#define DEBUG 0

enum Enum{ Kiss, HandShake, HighFive, Hug };
static const std::string INTERACTION_TYPES[4] = {"Kiss", "HandShake", "HighFive", "Hug"};

const char* getTextForEnum(int enumVal)
{
	return INTERACTION_TYPES[enumVal].c_str();
}


int DIC_SIZE = 500;
int NUM_CLUSTER_VIDEOS = 2;
int NUM_CROSS_VALID_LEAVE_OUT = 5;
int FEATURE_FRAMES_PER_SECOND = 4;
int FEATURE_MAX_FRAMES = 20;

