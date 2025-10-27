AioContext *bdrv_get_aio_context(BlockDriverState *bs)

{

    /* Currently BlockDriverState always uses the main loop AioContext */

    return qemu_get_aio_context();

}
