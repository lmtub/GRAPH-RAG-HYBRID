static int qtrle_decode_init(AVCodecContext *avctx)

{

    QtrleContext *s = avctx->priv_data;



    s->avctx = avctx;

    switch (avctx->bits_per_sample) {

    case 1:

    case 2:

    case 4:

    case 8:

    case 33:

    case 34:

    case 36:

    case 40:

        avctx->pix_fmt = PIX_FMT_PAL8;

        break;



    case 16:

        avctx->pix_fmt = PIX_FMT_RGB555;

        break;



    case 24:

        avctx->pix_fmt = PIX_FMT_RGB24;

        break;



    case 32:

        avctx->pix_fmt = PIX_FMT_RGB32;

        break;



    default:

        av_log (avctx, AV_LOG_ERROR, "Unsupported colorspace: %d bits/sample?\n",

            avctx->bits_per_sample);

        break;

    }

    dsputil_init(&s->dsp, avctx);



    s->frame.data[0] = NULL;



    return 0;

}
