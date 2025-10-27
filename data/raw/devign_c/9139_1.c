int aio_bh_poll(AioContext *ctx)

{

    QEMUBH *bh, **bhp, *next;

    int ret;



    ctx->walking_bh++;



    ret = 0;

    for (bh = ctx->first_bh; bh; bh = next) {

        /* Make sure that fetching bh happens before accessing its members */

        smp_read_barrier_depends();

        next = bh->next;

        /* The atomic_xchg is paired with the one in qemu_bh_schedule.  The

         * implicit memory barrier ensures that the callback sees all writes

         * done by the scheduling thread.  It also ensures that the scheduling

         * thread sees the zero before bh->cb has run, and thus will call

         * aio_notify again if necessary.

         */

        if (!bh->deleted && atomic_xchg(&bh->scheduled, 0)) {

            if (!bh->idle)

                ret = 1;

            bh->idle = 0;

            bh->cb(bh->opaque);

        }

    }



    ctx->walking_bh--;



    /* remove deleted bhs */

    if (!ctx->walking_bh) {

        qemu_mutex_lock(&ctx->bh_lock);

        bhp = &ctx->first_bh;

        while (*bhp) {

            bh = *bhp;

            if (bh->deleted) {

                *bhp = bh->next;

                g_free(bh);

            } else {

                bhp = &bh->next;

            }

        }

        qemu_mutex_unlock(&ctx->bh_lock);

    }



    return ret;

}
