static av_cold int xma_decode_init(AVCodecContext *avctx)

{

    XMADecodeCtx *s = avctx->priv_data;

    int i, ret;



    for (i = 0; i < avctx->channels / 2; i++) {

        ret = decode_init(&s->xma[i], avctx);

        s->frames[i] = av_frame_alloc();

        if (!s->frames[i])

            return AVERROR(ENOMEM);

        s->frames[i]->nb_samples = 512;

        if ((ret = ff_get_buffer(avctx, s->frames[i], 0)) < 0) {

            return AVERROR(ENOMEM);

        }



    }



    return ret;

}
