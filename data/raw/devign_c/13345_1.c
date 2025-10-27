static int append_to_cached_buf(AVCodecContext *avctx,

                                const uint8_t *buf, int buf_size)

{

    DVDSubContext *ctx = avctx->priv_data;



    if (ctx->buf_size > 0xffff - buf_size) {

        av_log(avctx, AV_LOG_WARNING, "Attempt to reconstruct "

               "too large SPU packets aborted.\n");

        av_freep(&ctx->buf);

        return AVERROR_INVALIDDATA;

    }

    ctx->buf = av_realloc(ctx->buf, ctx->buf_size + buf_size);

    if (!ctx->buf)

        return AVERROR(ENOMEM);

    memcpy(ctx->buf + ctx->buf_size, buf, buf_size);

    ctx->buf_size += buf_size;

    return 0;

}
