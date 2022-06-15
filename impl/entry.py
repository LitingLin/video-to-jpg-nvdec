import os
import sys


def touch(fname):
    try:
        os.utime(fname, None)
    except OSError:
        open(fname, 'a').close()


def entry(gpu_id, video_file, output_dir, log_dir, num_jpeg_encoding_threads, num_io_threads, jpeg_encoding_quality, interval, thread_max_queue, libjpegturbo_path):
    if log_dir is not None:
        sys.stdout = TeeStdOut(os.path.join(log_dir, 'stdout'))
        sys.stderr = TeeStdErr(os.path.join(log_dir, 'stdout'))
        success_file = os.path.join(log_dir, 'success')
        if os.path.exists(success_file):
            os.remove(success_file)

    from .jpeg_encoder import JpegEncoder
    from impl.nv_vpf_decoder import nv_vpf_decode_video_with_ffmpeg_demuxer

    jpeg_encoder = JpegEncoder(num_jpeg_encoding_threads, num_io_threads, jpeg_encoding_quality, thread_max_queue, libjpegturbo_path)

    with jpeg_encoder:
        nv_vpf_decode_video_with_ffmpeg_demuxer(gpu_id, video_file, output_dir, jpeg_encoder, thread_max_queue * num_jpeg_encoding_threads, interval)

    if log_dir is not None:
        touch(success_file)


class TeeStdOut:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.logfile = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()


class TeeStdErr:
    def __init__(self, filename):
        self.terminal = sys.stderr
        self.logfile = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()
