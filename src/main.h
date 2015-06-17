#if defined(WIN32) || defined(_WIN32)
#define PATH_SEPARATOR "\\" 
#else 
#define PATH_SEPARATOR "/" 
#endif

#define DEBUG 1

#include <string>

int DIC_SIZE = 500;
int NUM_CLUSTER_VIDEOS = 2;
int NUM_CROSS_VALID_LEAVE_OUT = 5;
int FEATURE_FRAMES_PER_SECOND = 4;
int FEATURE_MAX_FRAMES = 20;

static const std::string INTERACTION_TYPES[4] = {"Kiss", "HandShake", "HighFive", "Hug"};
