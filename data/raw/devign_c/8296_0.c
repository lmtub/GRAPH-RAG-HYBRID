aio_compute_timeout(AioContext *ctx)

{

    int64_t deadline;

    int timeout = -1;

    QEMUBH *bh;



    for (bh = atomic_rcu_read(&ctx->first_bh); bh;

         bh = atomic_rcu_read(&bh->next)) {

        if (bh->scheduled) {

            if (bh->idle) {

                /* idle bottom halves will be polled at least

                 * every 10ms */

                timeout = 10000000;

            } else {

                /* non-idle bottom halves will be executed

                 * immediately */

                return 0;

            }

        }

    }



    deadline = timerlistgroup_deadline_ns(&ctx->tlg);

    if (deadline == 0) {

        return 0;

    } else {

        return qemu_soonest_timeout(timeout, deadline);

    }

}
