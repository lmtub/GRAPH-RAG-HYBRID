int64_t bdrv_getlength(BlockDriverState *bs)

{

    BlockDriver *drv = bs->drv;

    if (!drv)

        return -ENOMEDIUM;



    if (bs->growable || bs->removable) {

        if (drv->bdrv_getlength) {

            return drv->bdrv_getlength(bs);

        }

    }

    return bs->total_sectors * BDRV_SECTOR_SIZE;

}
