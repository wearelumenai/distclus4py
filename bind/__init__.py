import cffi
import os

lib_file = os.path.join(os.path.dirname(__file__), 'build', 'distclus.so')

ffi = cffi.FFI()
ffi.cdef("""
 typedef enum {I_RANDOM, I_GIVIEN, I_KMEANSPP} initializer;
struct Kmeans_return {
	double* data;
	size_t l1;
	size_t l2;
};

extern struct Kmeans_return Kmeans(double* data, size_t l1, size_t l2, int k, int iter, initializer init);
""")
lib = ffi.dlopen(lib_file)
