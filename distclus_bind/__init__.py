import numpy as np

from bind import ffi, lib

TYPE_MAP = {}

for prefix in ('int', 'uint'):
    for log_bytes in range(4):
        ctype = '%s%d_t' % (prefix, 8 * (2 ** log_bytes))
        dtype = '%s%d' % (prefix[0], 2 ** log_bytes)
        TYPE_MAP[ctype] = np.dtype(dtype)

TYPE_MAP['float'] = np.dtype('f4')
TYPE_MAP['double'] = np.dtype('f8')


def as_array(ptr, l):
    """
    Convert a C pointer to a numpy 1D array
    :param ptr: cdata pointer to the array
    :param l: dimension size
    :return: numpy 1D array
    """
    _type = ffi.getctype(ffi.typeof(ptr).item)
    if _type not in TYPE_MAP:
        raise RuntimeError("Cannot create an array for element type: %s" % _type)
    return np.frombuffer(ffi.buffer(ptr, l * ffi.sizeof(_type)), TYPE_MAP[_type])


def as_array2(ptr, l1, l2):
    """
    Convert a C pointer to a numpy 2D array
    :param ptr: cdata pointer to the array
    :param l1: first dimension size
    :param l2: second dimension size
    :return: numpy 2D array
    """
    arr = as_array(ptr, l1 * l2)
    arr.shape = (l1, l2)
    return arr


def cast_initializer(initializer):
    """
    convert a string to a CFFI initializer enum type
    :param initializer: initializer identifier
    :return: CFFI cdata for c enum type
    """
    if initializer in ['random', 'rand']:
        return lib.I_RANDOM
    elif initializer == 'given':
        return lib.I_GIVEN
    elif initializer in ['kmeanspp', 'kmeans++']:
        return lib.I_KMEANSPP


def kmeans(data, k, iter, initializer="random"):
    """
    Compute an MCMC clustering
    :param data: numpy 2-D array representing a data set
    :param k: number of centroids
    :param initializer: initialisation method name:
        random: take init_k value as centers
        given: take init_k first values of the data set as centers
        kmeans++: cleaver initialisation
    :return: centers in a numpy 2-D array
    """
    arr = ffi.cast("double*", data.ctypes.data)
    l1 = ffi.cast("size_t", data.shape[0])
    l2 = ffi.cast("size_t", data.shape[1])
    res = lib.Kmeans(arr, l1, l2, k, iter, cast_initializer(initializer))
    gc_data = ffi.gc(res.data, lib.FreeArrPtr)
    centers = as_array2(gc_data, res.l1, res.l2)
    print(centers)
    return centers


def mcmc(data, framesize=None, init_k=8, mcmc_iter=30, init_iter=1, b=100, amp=1, norm=2, nu=1, initializer="random"):
    """
    Compute an MCMC clustering
    :param data: numpy 2-D array representing a data set
    :param framesize: history of data to consider during computation(default: len(data))
    :param init_k: initial number of centroids
    :param mcmc_iter: number of mcmc iterations
    :param init_iter: number of initialisation iteration
    :param b: mcmc b parameter
    :param amp: mcmc amp parameter
    :param norm: distance normalisation coefficient
    :param nu: degrees of freedom for the student alteration
    :param initializer: initialisation method name:
        random: take init_k value as centers
        given: take init_k first values of the data set as centers
        kmeans++: cleaver initialisation
    :return: centers in a numpy 2-D array
    """
    arr = ffi.cast("double*", data.ctypes.data)
    l1 = ffi.cast("size_t", data.shape[0])
    l2 = ffi.cast("size_t", data.shape[1])
    framesize = framesize or data.shape[0]
    res = lib.MCMC(arr, l1, l2, framesize, init_k, mcmc_iter, init_iter, b, amp, norm, nu, cast_initializer(initializer))
    gc_data = ffi.gc(res.data, lib.FreeArrPtr)
    centers = as_array2(gc_data, res.l1, res.l2)
    return centers
