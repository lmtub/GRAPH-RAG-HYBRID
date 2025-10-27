static av_cold int v410_encode_init(AVCodecContext *avctx)

{

    if (avctx->width & 1) {

        av_log(avctx, AV_LOG_ERROR, "v410 requires even width.\n");

        return AVERROR_INVALIDDATA;

    }



    avctx->coded_frame = av_frame_alloc();



    if (!avctx->coded_frame) {

        av_log(avctx, AV_LOG_ERROR, "Could not allocate frame.\n");

        return AVERROR(ENOMEM);

    }



    return 0;

}
