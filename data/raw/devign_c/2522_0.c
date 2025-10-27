int bdrv_commit(BlockDriverState *bs)

{

    BlockDriver *drv = bs->drv;

    int64_t sector, total_sectors, length, backing_length;

    int n, ro, open_flags;

    int ret = 0;

    uint8_t *buf = NULL;

    char filename[PATH_MAX];



    if (!drv)

        return -ENOMEDIUM;

    

    if (!bs->backing_hd) {

        return -ENOTSUP;

    }



    if (bdrv_in_use(bs) || bdrv_in_use(bs->backing_hd)) {

        return -EBUSY;

    }



    ro = bs->backing_hd->read_only;

    /* Use pstrcpy (not strncpy): filename must be NUL-terminated. */

    pstrcpy(filename, sizeof(filename), bs->backing_hd->filename);

    open_flags =  bs->backing_hd->open_flags;



    if (ro) {

        if (bdrv_reopen(bs->backing_hd, open_flags | BDRV_O_RDWR, NULL)) {

            return -EACCES;

        }

    }



    length = bdrv_getlength(bs);

    if (length < 0) {

        ret = length;

        goto ro_cleanup;

    }



    backing_length = bdrv_getlength(bs->backing_hd);

    if (backing_length < 0) {

        ret = backing_length;

        goto ro_cleanup;

    }



    /* If our top snapshot is larger than the backing file image,

     * grow the backing file image if possible.  If not possible,

     * we must return an error */

    if (length > backing_length) {

        ret = bdrv_truncate(bs->backing_hd, length);

        if (ret < 0) {

            goto ro_cleanup;

        }

    }



    total_sectors = length >> BDRV_SECTOR_BITS;

    buf = g_malloc(COMMIT_BUF_SECTORS * BDRV_SECTOR_SIZE);



    for (sector = 0; sector < total_sectors; sector += n) {

        ret = bdrv_is_allocated(bs, sector, COMMIT_BUF_SECTORS, &n);

        if (ret < 0) {

            goto ro_cleanup;

        }

        if (ret) {

            ret = bdrv_read(bs, sector, buf, n);

            if (ret < 0) {

                goto ro_cleanup;

            }



            ret = bdrv_write(bs->backing_hd, sector, buf, n);

            if (ret < 0) {

                goto ro_cleanup;

            }

        }

    }



    if (drv->bdrv_make_empty) {

        ret = drv->bdrv_make_empty(bs);

        if (ret < 0) {

            goto ro_cleanup;

        }

        bdrv_flush(bs);

    }



    /*

     * Make sure all data we wrote to the backing device is actually

     * stable on disk.

     */

    if (bs->backing_hd) {

        bdrv_flush(bs->backing_hd);

    }



    ret = 0;

ro_cleanup:

    g_free(buf);



    if (ro) {

        /* ignoring error return here */

        bdrv_reopen(bs->backing_hd, open_flags & ~BDRV_O_RDWR, NULL);

    }



    return ret;

}
