int bdrv_snapshot_delete(BlockDriverState *bs, const char *snapshot_id)

{

    BlockDriver *drv = bs->drv;

    if (!drv) {

        return -ENOMEDIUM;

    }

    if (drv->bdrv_snapshot_delete) {

        return drv->bdrv_snapshot_delete(bs, snapshot_id);

    }

    if (bs->file) {

        return bdrv_snapshot_delete(bs->file, snapshot_id);

    }

    return -ENOTSUP;

}
