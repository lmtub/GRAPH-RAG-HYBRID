static av_cold int cfhd_decode_init(AVCodecContext *avctx)

{

    CFHDContext *s = avctx->priv_data;



    avctx->bits_per_raw_sample = 10;

    s->avctx                   = avctx;

    avctx->width               = 0;

    avctx->height              = 0;



    return ff_cfhd_init_vlcs(s);

}
