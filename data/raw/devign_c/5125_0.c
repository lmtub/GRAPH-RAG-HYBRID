static void vdadec_flush(AVCodecContext *avctx)

{

    return ff_h264_decoder.flush(avctx);

}
