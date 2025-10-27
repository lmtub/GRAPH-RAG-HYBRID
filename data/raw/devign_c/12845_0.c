void block_job_sleep_ns(BlockJob *job, QEMUClockType type, int64_t ns)

{

    assert(job->busy);



    /* Check cancellation *before* setting busy = false, too!  */

    if (block_job_is_cancelled(job)) {

        return;

    }



    job->busy = false;

    if (block_job_is_paused(job)) {

        qemu_coroutine_yield();

    } else {

        co_aio_sleep_ns(blk_get_aio_context(job->blk), type, ns);

    }

    job->busy = true;

}
