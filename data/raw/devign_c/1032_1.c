static av_cold int init_buffers(SANMVideoContext *ctx)

{

    av_fast_padded_malloc(&ctx->frm0, &ctx->frm0_size, ctx->buf_size);

    av_fast_padded_malloc(&ctx->frm1, &ctx->frm1_size, ctx->buf_size);

    av_fast_padded_malloc(&ctx->frm2, &ctx->frm2_size, ctx->buf_size);

    if (!ctx->version)

        av_fast_padded_malloc(&ctx->stored_frame,

                              &ctx->stored_frame_size, ctx->buf_size);



    if (!ctx->frm0 || !ctx->frm1 || !ctx->frm2 ||

        (!ctx->stored_frame && !ctx->version)) {

        destroy_buffers(ctx);

        return AVERROR(ENOMEM);

    }



    return 0;

}
