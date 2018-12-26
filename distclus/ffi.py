import os

import cffi

lib_file = os.path.join(os.path.dirname(__file__), 'lib', 'distclus.so')
header_file = os.path.join(os.path.dirname(__file__), 'lib', 'distclus.so')

ffi = cffi.FFI()
ffi.cdef("""
typedef enum {I_RANDOM, I_GIVEN, I_KMEANSPP} initializer;
typedef enum {S_VECTORS, S_SERIES} space;
typedef enum {O_KMEANS, O_MCMC, O_KNN, O_STREAMING} oc;
typedef enum {F_ITERATIONS} figure;

struct IntArray1D {
    long* addr;
    size_t l1;
};

struct RealArray2D {
    double* addr;
    size_t l1;
    size_t l2;
};

extern void FreeRealArray(double* p0);

extern void FreeIntArray(long* p0);

extern int KMEANS(
    space space, int par, initializer init, long seed,
    double* data, size_t l1, size_t l2,
    int k, int iter, int framesize,
    space innerSpace, int window
);

extern int MCMC(
    space space, int par, initializer init, long seed,
    double* data, size_t l1, size_t l2,
    size_t dim, int initK, int maxK, int mcmcIter, int framesize, double b,
    double amp, double norm, double nu, int initIter,
    space innerSpace, int window
);

extern void Push(int descr, double* data, size_t l1, size_t l2);

extern int Run(int descr, int async);

extern struct RealArray2D RealCentroids(int descr);

extern struct IntArray1D Predict(int descr, double* data, size_t l1, size_t l2);

extern void Close(int descr);

extern void Free(int descr);

extern double RuntimeFigure(int descr, figure fig);
""")
lib = ffi.dlopen(lib_file)
