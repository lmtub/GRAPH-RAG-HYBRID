static int cinepak_decode_init(AVCodecContext *avctx)

{

    CinepakContext *s = avctx->priv_data;



    s->avctx = avctx;

    s->width = (avctx->width + 3) & ~3;

    s->height = (avctx->height + 3) & ~3;

    s->sega_film_skip_bytes = -1;  /* uninitialized state */



    // check for paletted data

    if ((avctx->palctrl == NULL) || (avctx->bits_per_sample == 40)) {

        s->palette_video = 0;

        avctx->pix_fmt = PIX_FMT_YUV420P;

    } else {

        s->palette_video = 1;

        avctx->pix_fmt = PIX_FMT_PAL8;

    }



    dsputil_init(&s->dsp, avctx);



    s->frame.data[0] = NULL;



    return 0;

}
