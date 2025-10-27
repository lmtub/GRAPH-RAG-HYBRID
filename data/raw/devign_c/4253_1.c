void bdrv_invalidate_cache_all(Error **errp)

{

    BlockDriverState *bs;

    Error *local_err = NULL;

    BdrvNextIterator *it = NULL;



    while ((it = bdrv_next(it, &bs)) != NULL) {

        AioContext *aio_context = bdrv_get_aio_context(bs);



        aio_context_acquire(aio_context);

        bdrv_invalidate_cache(bs, &local_err);

        aio_context_release(aio_context);

        if (local_err) {

            error_propagate(errp, local_err);

            return;

        }

    }

}
