bool aio_dispatch(AioContext *ctx, bool dispatch_fds)

{

    bool progress;



    progress = aio_bh_poll(ctx);

    if (dispatch_fds) {

        progress |= aio_dispatch_handlers(ctx, INVALID_HANDLE_VALUE);

    }

    progress |= timerlistgroup_run_timers(&ctx->tlg);

    return progress;

}
