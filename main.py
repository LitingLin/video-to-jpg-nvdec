import argparse
import multiprocessing
import pycuda.driver as cuda
import os
import tqdm
from concurrent.futures import ThreadPoolExecutor
import itertools


def _get_arg_parser():
    arg_parser = argparse.ArgumentParser(description='High-performance Video to JPG (image sequence) converter, NVDEC accelerated', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument('input_video_list', type=str, help="Path to the input video list file")
    arg_parser.add_argument('output_dir', type=str, help="Output path")
    arg_parser.add_argument('--log_dir', type=str, help="Logging path")
    arg_parser.add_argument('--device_ids', default='all', type=str, help="Select the CUDA devices by indices (e.g. '0,1') or 'all'")
    arg_parser.add_argument('--num_enc_threads', default='4', type=int, help="Number of jpeg image encoding threads (per GPU)")
    arg_parser.add_argument('--num_io_threads', default='4', type=int, help="Number of jpeg image write threads (per GPU)")
    arg_parser.add_argument('--jpeg_enc_quality', default='85', type=int, help="JPEG encoding quality (1 = worst, 100 = best)")
    arg_parser.add_argument('--extract_interval', default=1, type=int, help="Frame extraction interval")
    arg_parser.add_argument('--timeout', default=60*60, type=int, help="Max wait time for a single video decoding task (in seconds)")
    arg_parser.add_argument('--vpf_path', type=str, help="Path to the Video Processing Framework installation path")
    arg_parser.add_argument('--libturbojpeg_path', type=str, help="Override the system default path to the turbojpeg shared library, e.g. libturbojpeg.so or turbojpeg.dll")
    arg_parser.add_argument('--thread_max_queue', default=4, type=int, help="Adjust the max queue size for worker threads")
    return arg_parser


class Runner:
    def __init__(self,
                 output_dir, log_dir,
                 num_jpeg_encoding_threads, num_io_threads,
                 jpeg_encoding_quality, extract_interval,
                 timeout, thread_max_queue,
                 libturbojpeg_path):
        self.output_dir = output_dir
        self.log_dir = log_dir
        self.num_jpeg_encoding_threads = num_jpeg_encoding_threads
        self.num_io_threads = num_io_threads
        self.jpeg_encoding_quality = jpeg_encoding_quality
        self.extract_interval = extract_interval
        self.timeout = timeout
        self.thread_max_queue = thread_max_queue
        self.libturbojpeg_path = libturbojpeg_path

    def __call__(self, video_file_path, device_index):
        from impl.launch_worker import launch_worker
        from impl.entry import entry
        video_file_name = os.path.basename(video_file_path)
        output_dir = os.path.join(self.output_dir, video_file_name)
        os.makedirs(output_dir, exist_ok=True)

        if self.log_dir is not None:
            log_dir = os.path.join(self.log_dir, video_file_name)
            os.makedirs(log_dir, exist_ok=True)
        else:
            log_dir = None

        is_success = launch_worker(entry, args=(
                                   device_index, video_file_path, output_dir, log_dir,
                                   self.num_jpeg_encoding_threads, self.num_io_threads,
                                   self.jpeg_encoding_quality, self.extract_interval, self.thread_max_queue,
                                   self.libturbojpeg_path),
                                   timeout=self.timeout)
        return video_file_path, is_success


def main():
    args = _get_arg_parser().parse_args()
    multiprocessing.set_start_method('spawn', force=True)
    cuda.init()
    num_devices = cuda.Device.count()
    if args.device_ids == 'all':
        device_indices = tuple(range(num_devices))
    else:
        device_indices = tuple(int(idx) for idx in args.device_ids.split(','))
        assert all(idx in range(num_devices) for idx in device_indices)

    assert args.extract_interval > 0
    assert 1 <= args.jpeg_enc_quality <= 100
    assert args.timeout >= 0
    if args.timeout == 0:
        args.timeout = None
    assert args.num_enc_threads > 0
    assert args.num_io_threads > 0
    assert args.thread_max_queue > 0
    if args.vpf_path is not None:
        assert os.path.isdir(args.vpf_path)
    if args.libturbojpeg_path is not None:
        assert os.path.isfile(args.libturbojpeg_path)

    print(f'Device found: {list(range(num_devices))}, using: {list(device_indices)}')

    if args.vpf_path is not None:
        import sys
        sys.path.insert(0, args.vpf_path)
        if os.name == 'nt':
            os.environ['PATH'] = args.vpf_path + ';' + os.environ['PATH']
        else:
            os.environ['LD_LIBRARY_PATH'] = args.vpf_path + ':' + os.environ['LD_LIBRARY_PATH']

    if args.libturbojpeg_path is not None:
        libturbojpeg_dir_path = os.path.dirname(args.libturbojpeg_path)
        if os.name == 'nt':
            os.environ['PATH'] = libturbojpeg_dir_path + ';' + os.environ['PATH']
        else:
            os.environ['LD_LIBRARY_PATH'] = libturbojpeg_dir_path + ':' + os.environ['LD_LIBRARY_PATH']

    vid_files = tuple(l.strip() for l in open(args.input_video_list, 'r', encoding='utf-8'))
    vid_files = tuple(l[1: -1] if l.startswith('"') and l.endswith('"') else l for l in vid_files)
    vid_files = tuple(l for l in vid_files if len(l) > 0)
    vid_files = set(vid_files)
    vid_files = tuple(vid_files)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    log_dir = args.log_dir
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)

    runner = Runner(output_dir, log_dir,
                    args.num_enc_threads, args.num_io_threads,
                    args.jpeg_enc_quality, args.extract_interval,
                    args.timeout, args.thread_max_queue,
                    args.libturbojpeg_path)

    with ThreadPoolExecutor(max_workers=len(device_indices)) as ex, tqdm.tqdm(total=len(vid_files)) as progress_bar:
        success_count = 0
        failure_count = 0
        progress_bar.set_description('Processing', refresh=False)
        progress_bar.set_postfix({'success': success_count, 'fail': failure_count}, refresh=True)
        for vid_file, success_flag in ex.map(runner, vid_files, itertools.cycle(device_indices)):
            progress_bar.set_postfix({'last': os.path.basename(vid_file), 'success': success_count, 'fail': failure_count}, refresh=False)

            if success_flag:
                success_count += 1
                if log_dir is not None:
                    with open(os.path.join(log_dir, 'success'), 'a') as f:
                        f.write(vid_file)
            else:
                failure_count += 1
                if log_dir is not None:
                    with open(os.path.join(log_dir, 'fail'), 'a') as f:
                        f.write(vid_file)
            progress_bar.update()


if __name__ == "__main__":
    main()
