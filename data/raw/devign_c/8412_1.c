void block_job_resume_all(void)

{

    BlockJob *job = NULL;

    while ((job = block_job_next(job))) {

        AioContext *aio_context = blk_get_aio_context(job->blk);



        aio_context_acquire(aio_context);

        block_job_resume(job);

        aio_context_release(aio_context);

    }

}
