int bdrv_snapshot_goto(BlockDriverState *bs,

                       const char *snapshot_id)

{

    BlockDriver *drv = bs->drv;

    int ret, open_ret;



    if (!drv)

        return -ENOMEDIUM;

    if (drv->bdrv_snapshot_goto)

        return drv->bdrv_snapshot_goto(bs, snapshot_id);



    if (bs->file) {

        drv->bdrv_close(bs);

        ret = bdrv_snapshot_goto(bs->file, snapshot_id);

        open_ret = drv->bdrv_open(bs, NULL, bs->open_flags);

        if (open_ret < 0) {

            bdrv_delete(bs->file);

            bs->drv = NULL;

            return open_ret;

        }

        return ret;

    }



    return -ENOTSUP;

}
