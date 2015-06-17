#if defined(WIN32) || defined(_WIN32)
#define PATH_SEPARATOR "\\" 
#else 
#define PATH_SEPARATOR "/" 
#endif
#define TRAIN_MODE 0
#define SVM_PATH "svm.xml"
#define DICT_PATH "dictionary.xml"

#define TEST_PATH "test"
#define DEBUG 1
enum Enum{ Kiss, HandShake, HighFive, Hug };
static const char * EnumStrings[] = { "Kiss",  "HandShake", "HighFive", "Hug"};

const char* getTextForEnum(int enumVal)
{
	return EnumStrings[enumVal];
}