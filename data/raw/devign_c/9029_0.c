static bool bdrv_drain_poll(BlockDriverState *bs)

{

    bool waited = false;



    while (atomic_read(&bs->in_flight) > 0) {

        aio_poll(bdrv_get_aio_context(bs), true);

        waited = true;

    }

    return waited;

}
