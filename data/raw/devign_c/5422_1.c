static av_cold int dnxhd_decode_init_thread_copy(AVCodecContext *avctx)

{

    DNXHDContext *ctx = avctx->priv_data;




    // make sure VLC tables will be loaded when cid is parsed

    ctx->cid = -1;



    ctx->rows = av_mallocz_array(avctx->thread_count, sizeof(RowContext));

    if (!ctx->rows)

        return AVERROR(ENOMEM);



    return 0;

}