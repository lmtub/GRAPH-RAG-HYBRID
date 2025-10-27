static int vda_h264_uninit(AVCodecContext *avctx)

{

    VDAContext *vda = avctx->internal->hwaccel_priv_data;

    av_freep(&vda->bitstream);

    if (vda->frame)

        CVPixelBufferRelease(vda->frame);

    return 0;

}
