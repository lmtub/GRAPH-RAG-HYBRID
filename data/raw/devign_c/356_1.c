void block_job_enter(BlockJob *job)

{

    if (!block_job_started(job)) {

        return;

    }

    if (job->deferred_to_main_loop) {

        return;

    }



    if (!job->busy) {

        bdrv_coroutine_enter(blk_bs(job->blk), job->co);

    }

}
