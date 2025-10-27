static av_cold int atrac1_decode_end(AVCodecContext * avctx) {

    AT1Ctx *q = avctx->priv_data;



    av_freep(&q->out_samples[0]);



    ff_mdct_end(&q->mdct_ctx[0]);

    ff_mdct_end(&q->mdct_ctx[1]);

    ff_mdct_end(&q->mdct_ctx[2]);

    return 0;

}
