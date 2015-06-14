#if defined(WIN32) || defined(_WIN32)
#define PATH_SEPARATOR "\\" 
#else 
#define PATH_SEPARATOR "/" 
#endif

#define DEBUG 1

#define NTRAINING_SAMPLES   100
#define FRAC_LINEAR_SEP     0.9f