import os

import cffi

lib_file = os.path.join(os.path.dirname(__file__), 'lib', 'distclus.so')
header_file = os.path.join(os.path.dirname(__file__), 'lib', 'distclus.so')

ffi = cffi.FFI()
ffi.cdef("""
typedef enum {I_RANDOM, I_GIVEN, I_KMEANSPP} initializer;
typedef enum {S_VECTORS, S_COSINUS, S_SERIES} space;
typedef enum {O_KMEANS, O_MCMC, O_KNN, O_STREAMING} oc;
typedef enum {F_ITERATIONS, F_ACCEPTATIONS} figure;

extern void FreeRealArray(double* p0);

extern void FreeIntArray(long* p0);

struct Algo {
    int descr;
    const char* err;
};

extern struct Algo KMeans(
    space space, double* data, size_t l1, size_t l2, size_t l3,
    int par, initializer init, long seed,
    int k, int iter, int framesize,
    space innerSpace, int window
);

extern struct Algo MCMC(
    space space, double* data, size_t l1, size_t l2, size_t l3,
    int par, initializer init, long seed,
    size_t dim, int initK, int maxK, int iter, int framesize, double b,
    double amp, double norm, double nu,
    space innerSpace, int window
);

extern struct Algo Streaming(
    space space, double* data, size_t l1, size_t l2, size_t l3,
    long seed, int bufsize,
    double b, double lambda,
    space innerSpace, int window
);

extern const char* Push(int descr, double* data, size_t l1, size_t l2, size_t l3);

extern const char* Run(int descr, int async);

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

extern struct PredictResult Predict(int descr, double* data, size_t l1, size_t l2, size_t l3);

struct FigureResult {
    double value;
    const char *err;
};

extern struct FigureResult RuntimeFigure(int descr, figure fig);

extern void Close(int descr);

extern void Free(int descr);
""")
lib = ffi.dlopen(lib_file)
