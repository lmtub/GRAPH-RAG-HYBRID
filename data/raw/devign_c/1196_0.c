int bdrv_snapshot_load_tmp(BlockDriverState *bs,

        const char *snapshot_name)

{

    BlockDriver *drv = bs->drv;

    if (!drv) {

        return -ENOMEDIUM;

    }

    if (!bs->read_only) {

        return -EINVAL;

    }

    if (drv->bdrv_snapshot_load_tmp) {

        return drv->bdrv_snapshot_load_tmp(bs, snapshot_name);

    }

    return -ENOTSUP;

}
