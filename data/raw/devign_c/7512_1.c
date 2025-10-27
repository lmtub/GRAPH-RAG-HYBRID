AioContext *aio_context_new(void)

{

    AioContext *ctx;

    ctx = (AioContext *) g_source_new(&aio_source_funcs, sizeof(AioContext));

    ctx->pollfds = g_array_new(FALSE, FALSE, sizeof(GPollFD));

    ctx->thread_pool = NULL;

    qemu_mutex_init(&ctx->bh_lock);

    rfifolock_init(&ctx->lock, aio_rfifolock_cb, ctx);

    event_notifier_init(&ctx->notifier, false);

    aio_set_event_notifier(ctx, &ctx->notifier, 

                           (EventNotifierHandler *)

                           event_notifier_test_and_clear);

    timerlistgroup_init(&ctx->tlg, aio_timerlist_notify, ctx);



    return ctx;

}
