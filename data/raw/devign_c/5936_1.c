static av_cold int mss1_decode_init(AVCodecContext *avctx)

{

    MSS1Context * const c = avctx->priv_data;

    int ret;



    c->ctx.avctx       = avctx;



    c->pic = av_frame_alloc();

    if (!c->pic)

        return AVERROR(ENOMEM);



    ret = ff_mss12_decode_init(&c->ctx, 0, &c->sc, NULL);





    avctx->pix_fmt = AV_PIX_FMT_PAL8;



    return ret;

}