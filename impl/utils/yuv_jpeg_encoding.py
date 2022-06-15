from ctypes import *
from ctypes.util import find_library
import warnings
import numpy as np

TJSAMP_444 = 0
TJSAMP_422 = 1
TJSAMP_420 = 2
TJSAMP_GRAY = 3
TJSAMP_440 = 4
TJSAMP_411 = 5

TJERR_WARNING = 0
TJERR_FATAL = 1


def _get_ndarray_address(array: np.ndarray):
    return cast(array.__array_interface__['data'][0], POINTER(c_ubyte))


class JPEGEncoded:
    def __init__(self, jpeg_buf, jpeg_size, free_fn):
        self.jpeg_buf = jpeg_buf
        self.jpeg_size = jpeg_size
        self.free_fn = free_fn

    def __del__(self):
        self.dispose()

    def get_ptr(self):
        return self.jpeg_buf

    def get_size(self):
        return self.jpeg_size

    def dispose(self):
        if self.jpeg_buf is not None:
            self.free_fn(self.jpeg_buf)
            self.jpeg_buf = None


class YUVJpegEncoder:
    def __init__(self, turbojpeg_dll_path=None):
        if turbojpeg_dll_path is None:
            turbojpeg_dll_path = find_library('turbojpeg')
        assert turbojpeg_dll_path is not None
        turbojpeg_dll = cdll.LoadLibrary(turbojpeg_dll_path)

        _tj_init_compress = turbojpeg_dll.tjInitCompress
        _tj_init_compress.restype = c_void_p

        _tj_destroy = turbojpeg_dll.tjDestroy
        _tj_destroy.argtypes = c_void_p,
        _tj_destroy.restype = c_int

        _tj_compressFromYUV = turbojpeg_dll.tjCompressFromYUV
        _tj_compressFromYUV.argtypes = (
            c_void_p, POINTER(c_ubyte), c_int, c_int, c_int, c_int,
            POINTER(c_void_p), POINTER(c_ulong), c_int, c_int)
        _tj_compressFromYUV.restype = c_int

        _tj_free = turbojpeg_dll.tjFree
        _tj_free.argtypes = c_void_p,
        _tj_free.restype = None

        _tj_get_error_code = turbojpeg_dll.tjGetErrorCode
        _tj_get_error_code.argtypes = c_void_p,
        _tj_get_error_code.restype = c_int

        _tj_get_error_str = turbojpeg_dll.tjGetErrorStr2
        _tj_get_error_str.argtypes = c_void_p,
        _tj_get_error_str.restype = c_char_p

        self._tj_init_compress = _tj_init_compress
        self._tj_destroy = _tj_destroy
        self._tj_compressFromYUV = _tj_compressFromYUV
        self._tj_free = _tj_free
        self._tj_get_error_code = _tj_get_error_code
        self._tj_get_error_str = _tj_get_error_str

        self.handle = _tj_init_compress()

    def compress(self, data: np.ndarray, width: int, height: int, subsample=TJSAMP_420, quality=85, flags=0):
        yuv_data_ptr = _get_ndarray_address(data)

        jpeg_buf = c_void_p()
        jpeg_size = c_ulong()
        status = self._tj_compressFromYUV(
            self.handle, yuv_data_ptr, width, 1, height, subsample, byref(jpeg_buf),
            byref(jpeg_size), quality, flags)

        assert status == 0, self._get_tj_error_str()
        jpeg_size = jpeg_size.value
        assert jpeg_size > 0
        return JPEGEncoded(jpeg_buf, jpeg_size, self._tj_free)

    def __del__(self):
        self._tj_destroy(self.handle)

    def _get_tj_error_str(self):
        """reports error while error occurred"""
        if self._tj_get_error_code(self.handle) == TJERR_WARNING:
            warnings.warn(self._tj_get_error_str(self.handle).decode())
            return
        return self._tj_get_error_str(self.handle).decode()
