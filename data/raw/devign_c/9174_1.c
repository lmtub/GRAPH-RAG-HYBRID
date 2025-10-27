void coroutine_fn block_job_pause_point(BlockJob *job)

{

    assert(job && block_job_started(job));



    if (!block_job_should_pause(job)) {

        return;

    }

    if (block_job_is_cancelled(job)) {

        return;

    }



    if (job->driver->pause) {

        job->driver->pause(job);

    }



    if (block_job_should_pause(job) && !block_job_is_cancelled(job)) {

        job->paused = true;

        job->busy = false;

        qemu_coroutine_yield(); /* wait for block_job_resume() */

        job->busy = true;

        job->paused = false;

    }



    if (job->driver->resume) {

        job->driver->resume(job);

    }

}
