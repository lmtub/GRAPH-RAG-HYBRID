void qemu_bh_schedule(QEMUBH *bh)

{

    AioContext *ctx;



    if (bh->scheduled)

        return;

    ctx = bh->ctx;

    bh->idle = 0;

    /* Make sure that:

     * 1. idle & any writes needed by the callback are done before the

     *    locations are read in the aio_bh_poll.

     * 2. ctx is loaded before scheduled is set and the callback has a chance

     *    to execute.

     */

    smp_mb();

    bh->scheduled = 1;

    aio_notify(ctx);

}
