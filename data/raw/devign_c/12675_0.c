aio_ctx_check(GSource *source)

{

    AioContext *ctx = (AioContext *) source;

    QEMUBH *bh;



    atomic_and(&ctx->notify_me, ~1);

    aio_notify_accept(ctx);



    for (bh = ctx->first_bh; bh; bh = bh->next) {

        if (!bh->deleted && bh->scheduled) {

            return true;

        }

    }

    return aio_pending(ctx) || (timerlistgroup_deadline_ns(&ctx->tlg) == 0);

}
