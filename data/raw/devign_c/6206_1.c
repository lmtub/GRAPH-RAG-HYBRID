static bool try_poll_mode(AioContext *ctx, bool blocking)

{

    if (blocking && ctx->poll_max_ns && ctx->poll_disable_cnt == 0) {

        /* See qemu_soonest_timeout() uint64_t hack */

        int64_t max_ns = MIN((uint64_t)aio_compute_timeout(ctx),

                             (uint64_t)ctx->poll_max_ns);



        if (max_ns) {

            if (run_poll_handlers(ctx, max_ns)) {

                return true;

            }

        }

    }



    return false;

}
