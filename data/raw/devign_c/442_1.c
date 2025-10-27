void block_job_yield(BlockJob *job)

{

    assert(job->busy);



    /* Check cancellation *before* setting busy = false, too!  */

    if (block_job_is_cancelled(job)) {

        return;

    }



    job->busy = false;

    if (!block_job_should_pause(job)) {

        qemu_coroutine_yield();

    }

    job->busy = true;



    block_job_pause_point(job);

}
