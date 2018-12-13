import cffi

tffi = cffi.FFI()
tffi.cdef("""
extern void *malloc(size_t size);
""")
C = tffi.dlopen(None)