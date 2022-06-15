# High-performance NVDEC Video to JPG (image sequence) converter
This tool aims to convert a list of videos into jpg image sequences. NVDEC and multithread JPEG encoding provide incredible high-performance.

## Features
* GPU-accelerated video decoding and color space conversion
* Multi-GPU support
* Multithread JPEG encoding
* Fault-tolerance

## Usage
```
High-performance Video to JPG (image sequence) converter, NVDEC accelerated

positional arguments:
  input_video_list      Path to the input video list file
  output_dir            Output path

optional arguments:
  -h, --help            show this help message and exit
  --log_dir LOG_DIR     Logging path (default: None)
  --device_ids DEVICE_IDS
                        Select the CUDA devices by indices (e.g. '0,1') or 'all' (default: all)
  --num_enc_threads NUM_ENC_THREADS
                        Number of jpeg image encoding threads (per GPU) (default: 4)
  --num_io_threads NUM_IO_THREADS
                        Number of jpeg image write threads (per GPU) (default: 4)
  --jpeg_enc_quality JPEG_ENC_QUALITY
                        JPEG encoding quality (1 = worst, 100 = best) (default: 85)
  --extract_interval EXTRACT_INTERVAL
                        Frame extraction interval (default: 1)
  --timeout TIMEOUT     Max wait time for a single video decoding task (in seconds) (default: 3600)
  --vpf_path VPF_PATH   Path to the Video Processing Framework installation path (default: None)
  --libturbojpeg_path LIBTURBOJPEG_PATH
                        Override the system default path to the turbojpeg shared library, e.g. libturbojpeg.so or turbojpeg.dll (default: None)
  --thread_max_queue THREAD_MAX_QUEUE
                        Adjust the max queue size for worker threads (default: 4)
```
```input_file_list``` should contain video files line-by-line, like:
```
\path\to\vid_a.mp4
\path\to\vid_b.mp4
```
A typical usage with CUDA ID ```0,1``` enabled:
```shell
python main.py /path/to/video_file_list /path/to/output --vpf_path /path/to/vpf/ --device_ids 0,1
```
## Prerequisites
### libraries
#### Video Decoding
[VideoProcessingFramework](https://github.com/NVIDIA/VideoProcessingFramework) is adopted to provide GPU-accelerated video decoding and color space conversion. Follow the instruction [here](https://github.com/NVIDIA/VideoProcessingFramework/wiki/Building-from-source) to build from source. Install ffmpeg and Nvidia Video Codec SDK first.   
#### JPEG
TurboJPEG C API is adopted to encode YUV420 raw frames from GPU to jpeg. [libjpeg-turbo](https://libjpeg-turbo.org/) or [mozjpeg](https://github.com/mozilla/mozjpeg) is compatible. ```libjpeg-turbo``` is much faster while ```mozjpeg``` can generate a smaller file in similar quality. 
### python
Install following packages:
```
numpy
pycuda
tqdm
```
