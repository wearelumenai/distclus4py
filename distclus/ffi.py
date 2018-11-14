import os

import cffi

lib_file = os.path.join(os.path.dirname(__file__), 'lib', 'distclus.so')
header_file = os.path.join(os.path.dirname(__file__), 'lib', 'distclus.so')

ffi = cffi.FFI()
ffi.cdef("""
typedef enum {I_RANDOM, I_GIVEN, I_KMEANSPP} initializer;

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

extern int MCMC(size_t dim, int initK, int mcmcIter, int framesize, double b, double amp, double norm, double nu, int initIter, initializer init, long seed);

extern void Push(int descr, double* data, size_t l1, size_t l2);

extern void Run(int descr, int async);

extern struct RealArray2D RealCentroids(int descr);

extern struct IntArray1D Predict(int descr, double* data, size_t l1, size_t l2, int push);

extern void Close(int descr);

extern void Free(int descr);
""")
lib = ffi.dlopen(lib_file)
