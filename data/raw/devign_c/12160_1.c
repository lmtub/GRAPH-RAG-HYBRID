static int64_t try_fiemap(BlockDriverState *bs, off_t start, off_t *data,

                          off_t *hole, int nb_sectors, int *pnum)

{

#ifdef CONFIG_FIEMAP

    BDRVRawState *s = bs->opaque;

    int64_t ret = BDRV_BLOCK_DATA | BDRV_BLOCK_OFFSET_VALID | start;

    struct {

        struct fiemap fm;

        struct fiemap_extent fe;

    } f;



    if (s->skip_fiemap) {

        return -ENOTSUP;

    }



    f.fm.fm_start = start;

    f.fm.fm_length = (int64_t)nb_sectors * BDRV_SECTOR_SIZE;

    f.fm.fm_flags = 0;

    f.fm.fm_extent_count = 1;

    f.fm.fm_reserved = 0;

    if (ioctl(s->fd, FS_IOC_FIEMAP, &f) == -1) {

        s->skip_fiemap = true;

        return -errno;

    }



    if (f.fm.fm_mapped_extents == 0) {

        /* No extents found, data is beyond f.fm.fm_start + f.fm.fm_length.

         * f.fm.fm_start + f.fm.fm_length must be clamped to the file size!

         */

        off_t length = lseek(s->fd, 0, SEEK_END);

        *hole = f.fm.fm_start;

        *data = MIN(f.fm.fm_start + f.fm.fm_length, length);

    } else {

        *data = f.fe.fe_logical;

        *hole = f.fe.fe_logical + f.fe.fe_length;

        if (f.fe.fe_flags & FIEMAP_EXTENT_UNWRITTEN) {

            ret |= BDRV_BLOCK_ZERO;

        }

    }



    return ret;

#else

    return -ENOTSUP;

#endif

}
