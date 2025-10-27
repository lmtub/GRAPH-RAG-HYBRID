static int bdrv_open_common(BlockDriverState *bs, const char *filename,

    int flags, BlockDriver *drv)

{

    int ret, open_flags;



    assert(drv != NULL);



    bs->file = NULL;

    bs->total_sectors = 0;

    bs->is_temporary = 0;

    bs->encrypted = 0;

    bs->valid_key = 0;

    bs->open_flags = flags;

    /* buffer_alignment defaulted to 512, drivers can change this value */

    bs->buffer_alignment = 512;



    pstrcpy(bs->filename, sizeof(bs->filename), filename);



    if (use_bdrv_whitelist && !bdrv_is_whitelisted(drv)) {

        return -ENOTSUP;

    }



    bs->drv = drv;

    bs->opaque = qemu_mallocz(drv->instance_size);



    /*

     * Yes, BDRV_O_NOCACHE aka O_DIRECT means we have to present a

     * write cache to the guest.  We do need the fdatasync to flush

     * out transactions for block allocations, and we maybe have a

     * volatile write cache in our backing device to deal with.

     */

    if (flags & (BDRV_O_CACHE_WB|BDRV_O_NOCACHE))

        bs->enable_write_cache = 1;



    /*

     * Clear flags that are internal to the block layer before opening the

     * image.

     */

    open_flags = flags & ~(BDRV_O_SNAPSHOT | BDRV_O_NO_BACKING);



    /*

     * Snapshots should be writeable.

     */

    if (bs->is_temporary) {

        open_flags |= BDRV_O_RDWR;

    }



    /* Open the image, either directly or using a protocol */

    if (drv->bdrv_file_open) {

        ret = drv->bdrv_file_open(bs, filename, open_flags);

    } else {

        ret = bdrv_file_open(&bs->file, filename, open_flags);

        if (ret >= 0) {

            ret = drv->bdrv_open(bs, open_flags);

        }

    }



    if (ret < 0) {

        goto free_and_fail;

    }



    bs->keep_read_only = bs->read_only = !(open_flags & BDRV_O_RDWR);



    ret = refresh_total_sectors(bs, bs->total_sectors);

    if (ret < 0) {

        goto free_and_fail;

    }



#ifndef _WIN32

    if (bs->is_temporary) {

        unlink(filename);

    }

#endif

    return 0;



free_and_fail:

    if (bs->file) {

        bdrv_delete(bs->file);

        bs->file = NULL;

    }

    qemu_free(bs->opaque);

    bs->opaque = NULL;

    bs->drv = NULL;

    return ret;

}
