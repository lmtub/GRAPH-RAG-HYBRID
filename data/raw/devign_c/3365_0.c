LinuxAioState *aio_get_linux_aio(AioContext *ctx)

{

    if (!ctx->linux_aio) {

        ctx->linux_aio = laio_init();

        laio_attach_aio_context(ctx->linux_aio, ctx);

    }

    return ctx->linux_aio;

}
