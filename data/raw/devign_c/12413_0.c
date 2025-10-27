static av_cold int ffv1_encode_close(AVCodecContext *avctx)

{

    av_frame_free(&avctx->coded_frame);

    ffv1_close(avctx);

    return 0;

}
