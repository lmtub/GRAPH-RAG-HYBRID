static void bdrv_drain_poll(BlockDriverState *bs)

{

    while (bdrv_requests_pending(bs)) {

        /* Keep iterating */

        aio_poll(bdrv_get_aio_context(bs), true);

    }

}
