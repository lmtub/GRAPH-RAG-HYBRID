static av_cold int flashsv_decode_init(AVCodecContext *avctx)

{

    FlashSVContext *s = avctx->priv_data;

    int zret; // Zlib return code



    s->avctx          = avctx;

    s->zstream.zalloc = Z_NULL;

    s->zstream.zfree  = Z_NULL;

    s->zstream.opaque = Z_NULL;

    zret = inflateInit(&s->zstream);

    if (zret != Z_OK) {

        av_log(avctx, AV_LOG_ERROR, "Inflate init error: %d\n", zret);

        return 1;

    }

    avctx->pix_fmt = AV_PIX_FMT_BGR24;

    s->frame.data[0] = NULL;



    return 0;

}
