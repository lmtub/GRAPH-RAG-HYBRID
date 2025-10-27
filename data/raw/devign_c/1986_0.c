static bool run_poll_handlers(AioContext *ctx, int64_t max_ns)

{

    bool progress;

    int64_t end_time;



    assert(ctx->notify_me);

    assert(qemu_lockcnt_count(&ctx->list_lock) > 0);

    assert(ctx->poll_disable_cnt == 0);



    trace_run_poll_handlers_begin(ctx, max_ns);



    end_time = qemu_clock_get_ns(QEMU_CLOCK_REALTIME) + max_ns;



    do {

        progress = run_poll_handlers_once(ctx);

    } while (!progress && qemu_clock_get_ns(QEMU_CLOCK_REALTIME) < end_time);



    trace_run_poll_handlers_end(ctx, progress);



    return progress;

}
