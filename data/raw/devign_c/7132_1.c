void bdrv_set_aio_context(BlockDriverState *bs, AioContext *new_context)
{
    bdrv_drain(bs); /* ensure there are no in-flight requests */
    bdrv_detach_aio_context(bs);
    /* This function executes in the old AioContext so acquire the new one in
     * case it runs in a different thread.
     */
    aio_context_acquire(new_context);
    bdrv_attach_aio_context(bs, new_context);
    aio_context_release(new_context);