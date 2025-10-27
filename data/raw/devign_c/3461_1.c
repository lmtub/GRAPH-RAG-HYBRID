void bdrv_flush(BlockDriverState *bs)
{
    if (bs->drv && bs->drv->bdrv_flush)
        bs->drv->bdrv_flush(bs);