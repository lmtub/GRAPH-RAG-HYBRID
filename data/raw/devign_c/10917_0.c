static av_cold int msvideo1_decode_init(AVCodecContext *avctx)

{

    Msvideo1Context *s = avctx->priv_data;



    s->avctx = avctx;



    /* figure out the colorspace based on the presence of a palette */

    if (s->avctx->bits_per_coded_sample == 8) {

        s->mode_8bit = 1;

        avctx->pix_fmt = AV_PIX_FMT_PAL8;

    } else {

        s->mode_8bit = 0;

        avctx->pix_fmt = AV_PIX_FMT_RGB555;

    }



    s->frame.data[0] = NULL;



    return 0;

}
