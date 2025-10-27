static av_cold int truemotion1_decode_init(AVCodecContext *avctx)

{

    TrueMotion1Context *s = avctx->priv_data;



    s->avctx = avctx;



    // FIXME: it may change ?

//    if (avctx->bits_per_sample == 24)

//        avctx->pix_fmt = AV_PIX_FMT_RGB24;

//    else

//        avctx->pix_fmt = AV_PIX_FMT_RGB555;



    s->frame = av_frame_alloc();

    if (!s->frame)

        return AVERROR(ENOMEM);



    /* there is a vertical predictor for each pixel in a line; each vertical

     * predictor is 0 to start with */

    av_fast_malloc(&s->vert_pred, &s->vert_pred_size, s->avctx->width * sizeof(unsigned int));

    if (!s->vert_pred)

        return AVERROR(ENOMEM);



    return 0;

}
