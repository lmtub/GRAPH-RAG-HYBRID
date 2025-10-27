void aio_context_acquire(AioContext *ctx)

{

    qemu_rec_mutex_lock(&ctx->lock);

}
