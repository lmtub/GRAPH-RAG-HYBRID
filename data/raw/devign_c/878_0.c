static int rpza_decode_init(AVCodecContext *avctx)

{

    RpzaContext *s = avctx->priv_data;



    s->avctx = avctx;

    avctx->pix_fmt = PIX_FMT_RGB555;

    dsputil_init(&s->dsp, avctx);



    s->frame.data[0] = NULL;



    return 0;

}
