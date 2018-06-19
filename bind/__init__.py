import cffi
import os

lib_file = os.path.join(os.path.dirname(__file__), 'build', 'distclus.so')

ffi = cffi.FFI()
ffi.cdef("""
typedef enum {I_RANDOM, I_GIVIEN, I_KMEANSPP} initializer;
struct Array2D {
	double* data;
	size_t l1;
	size_t l2;
};

extern struct Array2D MCMC(double* data, size_t l1, size_t l2, int framesize, int initK, int mcmcIter, int initIter, double b, double amp, double norm, double nu, initializer init);

extern struct Array2D Kmeans(double* data, size_t l1, size_t l2, int k, int iter, initializer init);
""")
lib = ffi.dlopen(lib_file)
