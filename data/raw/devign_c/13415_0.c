int bdrv_write_compressed(BlockDriverState *bs, int64_t sector_num,

                          const uint8_t *buf, int nb_sectors)

{

    BlockDriver *drv = bs->drv;

    int ret;



    if (!drv) {

        return -ENOMEDIUM;

    }

    if (!drv->bdrv_write_compressed) {

        return -ENOTSUP;

    }

    ret = bdrv_check_request(bs, sector_num, nb_sectors);

    if (ret < 0) {

        return ret;

    }



    assert(QLIST_EMPTY(&bs->dirty_bitmaps));



    return drv->bdrv_write_compressed(bs, sector_num, buf, nb_sectors);

}
