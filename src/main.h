#if defined(WIN32) || defined(_WIN32)
#define PATH_SEPARATOR "\\" 
#else 
#define PATH_SEPARATOR "/" 
#endif
#define TRAIN_MODE 0
#define SVM_PATH "svm.xml"
#define DICT_PATH "dictionary.xml"
#include <string>
#define TEST_PATH "test"
#define DEBUG 0
enum Enum{ Kiss, HandShake, HighFive, Hug };
static const char * EnumStrings[] = { "Kiss",  "HandShake", "HighFive", "Hug"};

const char* getTextForEnum(int enumVal)
{
	return EnumStrings[enumVal];
}


int DIC_SIZE = 1000;
int NUM_CLUSTER_VIDEOS = 2;
int NUM_CROSS_VALID_LEAVE_OUT = 0;
int FEATURE_FRAMES_PER_SECOND = 4;
int FEATURE_MAX_FRAMES = 20;
static const std::string INTERACTION_TYPES[4] = {"Kiss", "HandShake", "HighFive", "Hug"};
