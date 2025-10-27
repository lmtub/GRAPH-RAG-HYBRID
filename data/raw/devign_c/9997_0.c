int bdrv_set_key(BlockDriverState *bs, const char *key)

{

    int ret;

    if (bs->backing_hd && bs->backing_hd->encrypted) {

        ret = bdrv_set_key(bs->backing_hd, key);

        if (ret < 0)

            return ret;

        if (!bs->encrypted)

            return 0;

    }

    if (!bs->encrypted) {

        return -EINVAL;

    } else if (!bs->drv || !bs->drv->bdrv_set_key) {

        return -ENOMEDIUM;

    }

    ret = bs->drv->bdrv_set_key(bs, key);

    if (ret < 0) {

        bs->valid_key = 0;

    } else if (!bs->valid_key) {

        bs->valid_key = 1;

        if (bs->blk) {

            /* call the change callback now, we skipped it on open */

            blk_dev_change_media_cb(bs->blk, true);

        }

    }

    return ret;

}
