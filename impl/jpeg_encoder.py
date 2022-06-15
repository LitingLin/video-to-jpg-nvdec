from .round_robin_worker_threads import RoundRobinWorkerThreads, BaseWorkerThreadHandler
from impl.utils.yuv_jpeg_encoding import YUVJpegEncoder, JPEGEncoded
from .utils.native_file_ops import native_write


class IOWorkerThread(BaseWorkerThreadHandler):
    def __call__(self, compressed: JPEGEncoded, path: str):
        native_write(compressed.get_ptr(), compressed.get_size(), path)
        compressed.dispose()


class JPEGEncoderWorkerThread(BaseWorkerThreadHandler):
    def __init__(self, quality, io_threads, libjpegturbo_path):
        self.quality = quality
        self.io_threads = io_threads
        self.libjpegturbo_path = libjpegturbo_path

    def __enter__(self):
        self.jpeg_encoder = YUVJpegEncoder(self.libjpegturbo_path)

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.jpeg_encoder

    def __call__(self, data, width, height, path):
        compressed = self.jpeg_encoder.compress(data, width, height, quality=self.quality)
        self.io_threads.put(compressed, path)


class JpegEncoder:
    def __init__(self, num_encoder_threads, num_io_threads, quality, thread_max_queue, libjpegturbo_path):
        self.io_threads = RoundRobinWorkerThreads(num_io_threads, IOWorkerThread, max_queue=thread_max_queue)
        self.encode_workers = RoundRobinWorkerThreads(num_encoder_threads, JPEGEncoderWorkerThread, (quality, self.io_threads, libjpegturbo_path), max_queue=thread_max_queue)

    def __enter__(self):
        self.io_threads.__enter__()
        self.encode_workers.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.encode_workers.__exit__(exc_type, exc_val, exc_tb)
        self.io_threads.__exit__(exc_type, exc_val, exc_tb)

    def encode(self, data, width, height, path):
        self.encode_workers.put(data, width, height, path)

    def join(self):
        self.encode_workers.join()
        self.io_threads.join()
