import os
from ctypes import *

if os.name == 'nt':
    libc = windll.msvcrt

    fopen = libc._wfopen
    fopen.argtypes = c_wchar_p, c_wchar_p
    fopen.restype = c_void_p
else:
    libc = CDLL("libc.so.6")

    fopen = libc.fopen
    fopen.argtypes = c_char_p, c_char_p
    fopen.restype = c_void_p

fwrite = libc.fwrite
fwrite.argtypes = c_void_p, c_size_t, c_size_t, c_void_p
fwrite.restype = c_size_t

fclose = libc.fclose
fclose.argtypes = c_void_p,
fclose.restype = c_int

ferror = libc.ferror
ferror.argtypes = c_void_p,
ferror.restype = c_int


def _get_native_error_str():
    if os.name == 'nt':
        return FormatError(GetLastError())
    else:
        return os.strerror(get_errno())


def native_write(ptr: c_void_p, size: int, path: str):
    if os.name == 'nt':
        f = fopen(path, 'wb')
    else:
        f = fopen(path.encode('utf-8'), b'wb')
    assert f is not None, _get_native_error_str()

    try:
        if fwrite(ptr, 1, size, f) != size:
            raise IOError(_get_native_error_str())
    finally:
        fclose(f)
