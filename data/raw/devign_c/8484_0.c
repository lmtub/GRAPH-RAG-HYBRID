static av_cold int msrle_decode_init(AVCodecContext *avctx)

{

    MsrleContext *s = avctx->priv_data;



    s->avctx = avctx;



    switch (avctx->bits_per_coded_sample) {

    case 4:

    case 8:

        avctx->pix_fmt = AV_PIX_FMT_PAL8;

        break;

    case 24:

        avctx->pix_fmt = AV_PIX_FMT_BGR24;

        break;

    default:

        av_log(avctx, AV_LOG_ERROR, "unsupported bits per sample\n");

        return AVERROR_INVALIDDATA;

    }



    s->frame.data[0] = NULL;



    return 0;

}
