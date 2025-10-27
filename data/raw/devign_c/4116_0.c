static AioContext *block_job_get_aio_context(BlockJob *job)

{

    return job->deferred_to_main_loop ?

           qemu_get_aio_context() :

           blk_get_aio_context(job->blk);

}
