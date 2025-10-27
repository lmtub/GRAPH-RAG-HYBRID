av_cold int ff_nvenc_encode_close(AVCodecContext *avctx)
{
    NVENCContext *ctx               = avctx->priv_data;
    NV_ENCODE_API_FUNCTION_LIST *nv = &ctx->nvel.nvenc_funcs;
    int i;
    av_fifo_free(ctx->timestamps);
    av_fifo_free(ctx->pending);
    av_fifo_free(ctx->ready);
    if (ctx->in) {
        for (i = 0; i < ctx->nb_surfaces; ++i) {
            nv->nvEncDestroyInputBuffer(ctx->nvenc_ctx, ctx->in[i].in);
            nv->nvEncDestroyBitstreamBuffer(ctx->nvenc_ctx, ctx->out[i].out);
    av_freep(&ctx->in);
    av_freep(&ctx->out);
    if (ctx->nvenc_ctx)
        nv->nvEncDestroyEncoder(ctx->nvenc_ctx);
    if (ctx->cu_context)
        ctx->nvel.cu_ctx_destroy(ctx->cu_context);
    if (ctx->nvel.nvenc)
        dlclose(ctx->nvel.nvenc);
    if (ctx->nvel.cuda)
        dlclose(ctx->nvel.cuda);
    return 0;