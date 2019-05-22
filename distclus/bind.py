from collections import namedtuple

import numpy as np

from .ffi import ffi, lib

TYPE_MAP = {
    'float': np.dtype('f4'),
    'double': np.dtype('f8'),
    'int': np.dtype('i4'),
    'long': np.dtype('i8')
}

Array = namedtuple("Array", "addr l1 l2 l3")
Array.__new__.__defaults__ = (None, 0, 0, 0)


def to_c_array(data):
    """
    Convert a numpy 2D array to C pointer
    """
    if data is None:
        return Array(addr=ffi.NULL, l1=0, l2=0, l3=0)
    arr = ffi.cast("double*", data.ctypes.data)
    zero = ffi.cast("size_t", 0)
    l1, l2, l3 = ffi.cast("size_t", data.shape[0]), zero, zero
    if len(data.shape) > 1:
        l2 = ffi.cast("size_t", data.shape[1])
    if len(data.shape) > 2:
        l3 = ffi.cast("size_t", data.shape[2])
    return Array(addr=arr, l1=l1, l2=l2, l3=l3)


def to_managed_array(ptr):
    """Convert a C pointer to a numpy 2D array and ensure it will be freed when
    array is garbage collected
    """
    length = ptr.l1
    shape = (ptr.l1,)
    if hasattr(ptr, "l2") and ptr.l2 > 0:
        length *= ptr.l2
        shape = (ptr.l1, ptr.l2)
    if hasattr(ptr, "l3") and ptr.l3 > 0:
        length *= ptr.l3
        shape = (ptr.l1, ptr.l2, ptr.l3)
    _type = get_type(ptr.addr)
    gc_data = finalize(ptr, _type)
    arr = np.frombuffer(ffi.buffer(gc_data, length * ffi.sizeof(_type)), TYPE_MAP[_type])
    arr.shape = shape
    return arr


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


def par(par):
    return 1 if par else 0


def handle_error(err):
    if err:
        raise RuntimeError(ffi.string(err))


def seed(seed):
    return 0 if seed is None else seed