static av_cold int j2kenc_init(AVCodecContext *avctx)

{

    int i, ret;

    Jpeg2000EncoderContext *s = avctx->priv_data;

    Jpeg2000CodingStyle *codsty = &s->codsty;

    Jpeg2000QuantStyle  *qntsty = &s->qntsty;



    s->avctx = avctx;

    av_log(s->avctx, AV_LOG_DEBUG, "init\n");



    // defaults:

    // TODO: implement setting non-standard precinct size

    memset(codsty->log2_prec_widths , 15, sizeof(codsty->log2_prec_widths ));

    memset(codsty->log2_prec_heights, 15, sizeof(codsty->log2_prec_heights));

    codsty->nreslevels2decode=

    codsty->nreslevels       = 7;

    codsty->log2_cblk_width  = 4;

    codsty->log2_cblk_height = 4;

    codsty->transform        = avctx->prediction_method ? FF_DWT53 : FF_DWT97_INT;



    qntsty->nguardbits       = 1;



    s->tile_width            = 256;

    s->tile_height           = 256;



    if (codsty->transform == FF_DWT53)

        qntsty->quantsty = JPEG2000_QSTY_NONE;

    else

        qntsty->quantsty = JPEG2000_QSTY_SE;



    s->width = avctx->width;

    s->height = avctx->height;



    for (i = 0; i < 3; i++)

        s->cbps[i] = 8;



    if (avctx->pix_fmt == AV_PIX_FMT_RGB24){

        s->ncomponents = 3;

    } else if (avctx->pix_fmt == AV_PIX_FMT_GRAY8){

        s->ncomponents = 1;

    } else{ // planar YUV

        s->planar = 1;

        s->ncomponents = 3;

        avcodec_get_chroma_sub_sample(avctx->pix_fmt,

                s->chroma_shift, s->chroma_shift + 1);

    }



    ff_jpeg2000_init_tier1_luts();



    init_luts();



    init_quantization(s);

    if (ret=init_tiles(s))

        return ret;



    av_log(s->avctx, AV_LOG_DEBUG, "after init\n");



    return 0;

}
