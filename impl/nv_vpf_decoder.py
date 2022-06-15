import os
import pycuda.driver as cuda
import PyNvCodec as nvc
import numpy as np


class _RingBuffer:
    def __init__(self, size, num):
        self.buffers = tuple(np.empty(size, dtype=np.uint8) for _ in range(num))
        self.index = 0

    def get(self):
        buffer = self.buffers[self.index]
        return buffer

    def forward(self):
        self.index += 1
        if self.index >= len(self.buffers):
            self.index = 0


def nv_vpf_decode_video_with_ffmpeg_demuxer(gpu_id, source_file_path, destination_folder_path, encoder, raw_frame_buffer_size, interval=1):
    cuda.init()
    cuda_ctx = cuda.Device(gpu_id).retain_primary_context()
    cuda_ctx.push()
    cuda_stream = cuda.Stream()
    cuda_ctx.pop()

    nvDmx = nvc.PyFFmpegDemuxer(source_file_path)
    nvDec = nvc.PyNvDecoder(nvDmx.Width(), nvDmx.Height(), nvDmx.Format(), nvDmx.Codec(), cuda_ctx.handle, cuda_stream.handle)
    nvCvt = nvc.PySurfaceConverter(nvDmx.Width(), nvDmx.Height(), nvDmx.Format(), nvc.PixelFormat.YUV420, cuda_ctx.handle, cuda_stream.handle)
    nvDwn = nvc.PySurfaceDownloader(nvDmx.Width(), nvDmx.Height(), nvCvt.Format(), cuda_ctx.handle, cuda_stream.handle)

    packet = np.ndarray(shape=(0), dtype=np.uint8)
    frameSize = int(nvDmx.Width() * nvDmx.Height() * 3 / 2)

    raw_frames = _RingBuffer(frameSize, raw_frame_buffer_size)

    pdata_in, pdata_out = nvc.PacketData(), nvc.PacketData()

    # Determine colorspace conversion parameters.
    # Some video streams don't specify these parameters so default values
    # are most widespread bt601 and mpeg.
    cspace, crange = nvDmx.ColorSpace(), nvDmx.ColorRange()
    if nvc.ColorSpace.UNSPEC == cspace:
        cspace = nvc.ColorSpace.BT_601
    if nvc.ColorRange.UDEF == crange:
        crange = nvc.ColorRange.MPEG
    cc_ctx = nvc.ColorspaceConversionContext(cspace, crange)

    count = 0

    while True:
        # Demuxer has sync design, it returns packet every time it's called.
        # If demuxer can't return packet it usually means EOF.
        if not nvDmx.DemuxSinglePacket(packet):
            break

        # Get last packet data to obtain frame timestamp
        nvDmx.LastPacketData(pdata_in)

        # Decoder is async by design.
        # As it consumes packets from demuxer one at a time it may not return
        # decoded surface every time the decoding function is called.
        surface_nv12 = nvDec.DecodeSurfaceFromPacket(pdata_in, packet, pdata_out)
        if not surface_nv12.Empty():
            count += 1
            if (count - 1) % interval != 0:
                continue
            surface_yuv420 = nvCvt.Execute(surface_nv12, cc_ctx)
            if surface_yuv420.Empty():
                break
            if not nvDwn.DownloadSingleSurface(surface_yuv420, raw_frames.get()):
                break

            encoder.encode(raw_frames.get(), nvDmx.Width(), nvDmx.Height(), os.path.join(destination_folder_path, f'{count:06d}.jpg'))
            raw_frames.forward()

    # Now we flush decoder to emtpy decoded frames queue.
    while True:
        count += 1
        if (count - 1) % interval != 0:
            continue
        surface_nv12 = nvDec.FlushSingleSurface()
        if surface_nv12.Empty():
            break
        surface_yuv420 = nvCvt.Execute(surface_nv12, cc_ctx)
        if surface_yuv420.Empty():
            break
        if not nvDwn.DownloadSingleSurface(surface_yuv420, raw_frames.get()):
            break

        encoder.encode(raw_frames.get(), nvDmx.Width(), nvDmx.Height(), os.path.join(destination_folder_path, f'{count:06d}.jpg'))
        raw_frames.forward()
