aio_ctx_finalize(GSource     *source)
{
    AioContext *ctx = (AioContext *) source;
    thread_pool_free(ctx->thread_pool);
    aio_set_event_notifier(ctx, &ctx->notifier, NULL);
    event_notifier_cleanup(&ctx->notifier);
    rfifolock_destroy(&ctx->lock);
    qemu_mutex_destroy(&ctx->bh_lock);
    timerlistgroup_deinit(&ctx->tlg);