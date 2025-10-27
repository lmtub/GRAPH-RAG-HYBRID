static void run_block_job(BlockJob *job, Error **errp)

{

    AioContext *aio_context = blk_get_aio_context(job->blk);



    /* FIXME In error cases, the job simply goes away and we access a dangling

     * pointer below. */

    aio_context_acquire(aio_context);

    do {

        aio_poll(aio_context, true);

        qemu_progress_print(job->len ?

                            ((float)job->offset / job->len * 100.f) : 0.0f, 0);

    } while (!job->ready);



    block_job_complete_sync(job, errp);

    aio_context_release(aio_context);



    /* A block job may finish instantaneously without publishing any progress,

     * so just signal completion here */

    qemu_progress_print(100.f, 0);

}
