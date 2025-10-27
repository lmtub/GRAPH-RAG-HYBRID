static av_cold int v410_decode_init(AVCodecContext *avctx)

{

    avctx->pix_fmt             = PIX_FMT_YUV444P10;

    avctx->bits_per_raw_sample = 10;



    if (avctx->width & 1) {

        av_log(avctx, AV_LOG_ERROR, "v410 requires width to be even.\n");

        return AVERROR_INVALIDDATA;

    }



    avctx->coded_frame = avcodec_alloc_frame();



    if (!avctx->coded_frame) {

        av_log(avctx, AV_LOG_ERROR, "Could not allocate frame.\n");

        return AVERROR(ENOMEM);

    }



    return 0;

}
