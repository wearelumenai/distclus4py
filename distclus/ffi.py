import os

import cffi

lib_file = os.path.join(os.path.dirname(__file__), 'lib', 'distclus.so')
header_file = os.path.join(os.path.dirname(__file__), 'lib', 'distclus.so')

ffi = cffi.FFI()
ffi.cdef("""
typedef enum {I_RANDOM, I_GIVEN, I_KMEANSPP} initializer;
typedef enum {S_VECTORS, S_COSINUS, S_SERIES} space;
typedef enum {O_KMEANS, O_MCMC, O_KNN, O_STREAMING} oc;
typedef enum {F_ITERATIONS} figure;

struct IntArray {
    long* addr;
    size_t l1;
    const char* err;
};

struct RealArray {
    double* addr;
    size_t l1;
    size_t l2;
    size_t l3;
    const char* err;
};

struct Algo {
    int descr;
    const char* err;
};

struct Figure {
    double value;
    const char *err;
};

extern void FreeRealArray(double* p0);

extern void FreeIntArray(long* p0);

extern struct Algo KMeans(
    space space, double* data, size_t l1, size_t l2,
    int par, initializer init, long seed,
    int k, int iter, int framesize,
    space innerSpace, int window
);

extern struct Algo MCMC(
    space space, double* data, size_t l1, size_t l2,
    int par, initializer init, long seed,
    size_t dim, int initK, int maxK, int mcmcIter, int framesize, double b,
    double amp, double norm, double nu, int initIter,
    space innerSpace, int window
);

extern struct Algo Streaming(
	space space, double* data, size_t l1, size_t l2,
    long seed, int bufsize,
	double b, double lambda,
	space innerSpace, int window
);

extern const char* Push(int descr, double* data, size_t l1, size_t l2);

extern const char* Run(int descr, int async);

extern struct RealArray Centroids(int descr);

extern struct IntArray Predict(int descr, double* data, size_t l1, size_t l2);

extern struct Figure RuntimeFigure(int descr, figure fig);

extern void Close(int descr);

extern void Free(int descr);
""")
lib = ffi.dlopen(lib_file)
