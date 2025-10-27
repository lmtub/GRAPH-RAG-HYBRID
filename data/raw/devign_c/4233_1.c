int ff_rv34_decode_init_thread_copy(AVCodecContext *avctx)

{

    int err;

    RV34DecContext *r = avctx->priv_data;



    r->s.avctx = avctx;



    if (avctx->internal->is_copy) {

        r->tmp_b_block_base = NULL;

        if ((err = ff_MPV_common_init(&r->s)) < 0)

            return err;

        if ((err = rv34_decoder_alloc(r)) < 0)

            return err;

    }



    return 0;

}
