int bdrv_can_snapshot(BlockDriverState *bs)

{

    BlockDriver *drv = bs->drv;

    if (!drv || bdrv_is_removable(bs) || bdrv_is_read_only(bs)) {

        return 0;

    }



    if (!drv->bdrv_snapshot_create) {

        if (bs->file != NULL) {

            return bdrv_can_snapshot(bs->file);

        }

        return 0;

    }



    return 1;

}
