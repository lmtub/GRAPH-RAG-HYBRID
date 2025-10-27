void bdrv_drain_all_end(void)

{

    BlockDriverState *bs;

    BdrvNextIterator it;

    BlockJob *job = NULL;



    for (bs = bdrv_first(&it); bs; bs = bdrv_next(&it)) {

        AioContext *aio_context = bdrv_get_aio_context(bs);



        aio_context_acquire(aio_context);

        aio_enable_external(aio_context);

        bdrv_io_unplugged_end(bs);

        bdrv_parent_drained_end(bs);

        aio_context_release(aio_context);

    }



    while ((job = block_job_next(job))) {

        AioContext *aio_context = blk_get_aio_context(job->blk);



        aio_context_acquire(aio_context);

        block_job_resume(job);

        aio_context_release(aio_context);

    }

}
