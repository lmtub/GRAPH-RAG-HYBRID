void block_job_enter(BlockJob *job)

{

    if (job->co && !job->busy) {

        bdrv_coroutine_enter(blk_bs(job->blk), job->co);

    }

}
