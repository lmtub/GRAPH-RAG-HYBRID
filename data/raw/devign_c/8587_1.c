int bdrv_truncate(BlockDriverState *bs, int64_t offset)

{

    BlockDriver *drv = bs->drv;

    int ret;

    if (!drv)

        return -ENOMEDIUM;

    if (!drv->bdrv_truncate)

        return -ENOTSUP;

    if (bs->read_only)

        return -EACCES;



    ret = drv->bdrv_truncate(bs, offset);

    if (ret == 0) {

        ret = refresh_total_sectors(bs, offset >> BDRV_SECTOR_BITS);

        bdrv_dirty_bitmap_truncate(bs);

        bdrv_parent_cb_resize(bs);


    }

    return ret;

}