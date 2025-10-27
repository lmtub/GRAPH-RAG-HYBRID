static void bdrv_io_limits_intercept(BlockDriverState *bs,

                                     unsigned int bytes,

                                     bool is_write)

{

    /* does this io must wait */

    bool must_wait = throttle_schedule_timer(&bs->throttle_state, is_write);



    /* if must wait or any request of this type throttled queue the IO */

    if (must_wait ||

        !qemu_co_queue_empty(&bs->throttled_reqs[is_write])) {

        qemu_co_queue_wait(&bs->throttled_reqs[is_write]);

    }



    /* the IO will be executed, do the accounting */

    throttle_account(&bs->throttle_state, is_write, bytes);





    /* if the next request must wait -> do nothing */

    if (throttle_schedule_timer(&bs->throttle_state, is_write)) {

        return;

    }



    /* else queue next request for execution */

    qemu_co_queue_next(&bs->throttled_reqs[is_write]);

}
