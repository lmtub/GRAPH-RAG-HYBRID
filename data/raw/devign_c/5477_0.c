static void block_job_defer_to_main_loop_bh(void *opaque)

{

    BlockJobDeferToMainLoopData *data = opaque;

    AioContext *aio_context;



    /* Prevent race with block_job_defer_to_main_loop() */

    aio_context_acquire(data->aio_context);



    /* Fetch BDS AioContext again, in case it has changed */

    aio_context = blk_get_aio_context(data->job->blk);

    if (aio_context != data->aio_context) {

        aio_context_acquire(aio_context);

    }



    data->job->deferred_to_main_loop = false;

    data->fn(data->job, data->opaque);



    if (aio_context != data->aio_context) {

        aio_context_release(aio_context);

    }



    aio_context_release(data->aio_context);



    g_free(data);

}
