import os

import cffi

lib_file = os.path.join(os.path.dirname(__file__), 'lib', 'distclus.so')
header_file = os.path.join(os.path.dirname(__file__), 'lib', 'distclus.so')

ffi = cffi.FFI()
ffi.cdef("""
typedef enum {I_RANDOM, I_GIVEN, I_KMEANSPP, I_OC} initializer;
typedef enum {S_VECTORS, S_COSINUS, S_SERIES} space;
typedef enum {O_KMEANS, O_MCMC, O_KNN, O_STREAMING} oc;
typedef enum {
    F_ITERATIONS, F_ACCEPTATIONS, F_MAX_DISTANCE, F_PUSHED_DATA,
    F_LAST_ITERATIONS, F_DURATION, F_LAST_DURATION
} figure;

extern void FreeRealArray(double* p0);

extern void FreeIntArray(long* p0);

struct Algo {
    int descr;
    const char* err;
};

extern struct Algo KMeans(
    space space, double* data, size_t l1, size_t l2, size_t l3,
    int par, initializer init, int initdescr, long seed,
    int k, int iter, int framesize,
    float iterFreq, int dataPerIter, int timeout, int numCPU,
    space innerSpace, int window
);

extern struct Algo MCMC(
    space space, double* data, size_t l1, size_t l2, size_t l3,
    int par, initializer init, int initdescr, long seed,
    size_t dim, int initK, int maxK, int iter, int framesize, double b,
    double amp, double norm, double nu,
    float iterFreq, int dataPerIter, int timeout, int numCPU,
    space innerSpace, int window
);

extern struct Algo Streaming(
    space space, double* data, size_t l1, size_t l2, size_t l3,
    long seed, int bufsize,
    double mu, double sigma,
    double outRatio , int outAfter,
    int iter, float iterFreq, int dataPerIter, int timeout,
    space innerSpace, int window
);

struct CombineResult {
    double* combined;
    size_t l1;
    size_t l2;
    size_t l3;
    const char* err;
};

extern struct CombineResult Combine(
    int descr,
    double* data1, size_t l11, size_t l21, size_t l31, int weight1,
    double* data2, size_t l12, size_t l22, size_t l32, int weight2
);

struct DistResult {
    double dist;
    const char* err;
};

extern struct DistResult Dist(
    int descr,
    double* data1, size_t l11, size_t l21, size_t l31,
    double* data2, size_t l12, size_t l22, size_t l32
);

extern const char* Push(
    int descr, double* data, size_t l1, size_t l2, size_t l3
);

extern const char* Play(int descr, int iter, int duration);

extern const char* Wait(int descr, int iter, int duration);

extern const char* Pause(int descr);

extern const char* Stop(int descr);

extern const int Alive(int descr);

extern const char* Status(int descr);

extern const char* Batch(int descr, int iter, int duration);

extern const char* Init(int descr);

struct CentroidsResult {
    double* centroids;
    size_t l1;
    size_t l2;
    size_t l3;
    const char* err;
};

extern struct CentroidsResult Centroids(int descr);

struct PredictResult {
    long* labels;
    size_t n1;
    double* centroids;
    size_t l1;
    size_t l2;
    size_t l3;
    const char* err;
};

extern struct PredictResult Predict(
    int descr, double* data, size_t l1, size_t l2, size_t l3
);

struct FigureResult {
    double value;
    const char *err;
};

extern struct FigureResult RuntimeFigure(int descr, figure fig);

extern const char* Close(int descr);

extern void Free(int descr);
""")
lib = ffi.dlopen(lib_file)
