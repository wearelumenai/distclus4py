from collections import namedtuple

import numpy as np

from distclus.ffi import ffi
from .ffi import ffi, lib

TYPE_MAP = {
    'float': np.dtype('f4'),
    'double': np.dtype('f8'),
    'int': np.dtype('i4'),
    'long': np.dtype('i8')
}

Array1D = namedtuple("Array1D", "addr l1")
Array2D = namedtuple("Array2D", "addr l1 l2")


def to_c_2d_array(data):
    """
    Convert a numpy 2D array to C pointer
    """
    arr = ffi.cast("double*", data.ctypes.data)
    l1 = ffi.cast("size_t", data.shape[0])
    l2 = ffi.cast("size_t", data.shape[1])
    return Array2D(addr=arr, l1=l1, l2=l2)


def to_managed_2d_array(ptr):
    """Convert a C pointer to a numpy 2D array and ensure it will be freed when
    array is garbage collected
    """
    ptr_1d = Array1D(addr=ptr.addr, l1=ptr.l1 * ptr.l2)
    arr = to_managed_1d_array(ptr_1d)
    arr.shape = (ptr.l1, ptr.l2)
    return arr


def to_managed_1d_array(ptr):
    """Convert a C pointer to a numpy 1D array and ensure it will be freed when
    array is garbage collected
    """
    _type = get_type(ptr.addr)
    gc_data = finalize(ptr, _type)
    return np.frombuffer(
        ffi.buffer(gc_data, ptr.l1 * ffi.sizeof(_type)), TYPE_MAP[_type]
    )


def finalize(ptr, _type):
    """
    Wrap a given pointer in a managed object in order to free it
    """
    if _type == 'int' or _type == 'long':
        gc_data = ffi.gc(ptr.addr, lib.FreeIntArray)
    else:
        gc_data = ffi.gc(ptr.addr, lib.FreeRealArray)
    return gc_data


def get_type(cdata):
    """
    Get the C type of the given cdata
    """
    _type = ffi.getctype(ffi.typeof(cdata).item)
    if _type not in TYPE_MAP:
        raise RuntimeError("unknown type: {}".format(_type))
    return _type


def initializer(name):
    """
    convert a string to a CFFI initializer enum type
    """
    if name in ['random', 'rand']:
        return lib.I_RANDOM
    elif name == 'given':
        return lib.I_GIVEN
    elif name in ['kmeanspp', 'kmeans++']:
        return lib.I_KMEANSPP


def space(name):
    """convert a string to a CFFI space enum type"""
    return getattr(lib, 'S_{0}'.format(name.upper()))


def oc(name):
    """convert a string to a CFFI oc enum type"""
    return getattr(lib, 'O_{0}'.format(name.upper()))


def figure(name):
    """convert a string to a CFFI oc enum type"""
    return getattr(lib, 'F_{0}'.format(name.upper()))


def handle_error(err):
    if err:
        raise RuntimeError(ffi.string(err))