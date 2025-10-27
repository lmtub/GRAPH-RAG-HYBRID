static void co_schedule_bh_cb(void *opaque)
{
    AioContext *ctx = opaque;
    QSLIST_HEAD(, Coroutine) straight, reversed;
    QSLIST_MOVE_ATOMIC(&reversed, &ctx->scheduled_coroutines);
    QSLIST_INIT(&straight);
    while (!QSLIST_EMPTY(&reversed)) {
        Coroutine *co = QSLIST_FIRST(&reversed);
        QSLIST_REMOVE_HEAD(&reversed, co_scheduled_next);
        QSLIST_INSERT_HEAD(&straight, co, co_scheduled_next);
    }
    while (!QSLIST_EMPTY(&straight)) {
        Coroutine *co = QSLIST_FIRST(&straight);
        QSLIST_REMOVE_HEAD(&straight, co_scheduled_next);
        trace_aio_co_schedule_bh_cb(ctx, co);
        aio_context_acquire(ctx);
        qemu_coroutine_enter(co);
        aio_context_release(ctx);
    }
}