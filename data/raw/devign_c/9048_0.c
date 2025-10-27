raw_co_writev_flags(BlockDriverState *bs, int64_t sector_num, int nb_sectors,

                    QEMUIOVector *qiov, int flags)

{

    void *buf = NULL;

    BlockDriver *drv;

    QEMUIOVector local_qiov;

    int ret;



    if (bs->probed && sector_num == 0) {

        /* As long as these conditions are true, we can't get partial writes to

         * the probe buffer and can just directly check the request. */

        QEMU_BUILD_BUG_ON(BLOCK_PROBE_BUF_SIZE != 512);

        QEMU_BUILD_BUG_ON(BDRV_SECTOR_SIZE != 512);



        if (nb_sectors == 0) {

            /* qemu_iovec_to_buf() would fail, but we want to return success

             * instead of -EINVAL in this case. */

            return 0;

        }



        buf = qemu_try_blockalign(bs->file->bs, 512);

        if (!buf) {

            ret = -ENOMEM;

            goto fail;

        }



        ret = qemu_iovec_to_buf(qiov, 0, buf, 512);

        if (ret != 512) {

            ret = -EINVAL;

            goto fail;

        }



        drv = bdrv_probe_all(buf, 512, NULL);

        if (drv != bs->drv) {

            ret = -EPERM;

            goto fail;

        }



        /* Use the checked buffer, a malicious guest might be overwriting its

         * original buffer in the background. */

        qemu_iovec_init(&local_qiov, qiov->niov + 1);

        qemu_iovec_add(&local_qiov, buf, 512);

        qemu_iovec_concat(&local_qiov, qiov, 512, qiov->size - 512);

        qiov = &local_qiov;

    }



    BLKDBG_EVENT(bs->file, BLKDBG_WRITE_AIO);

    ret = bdrv_co_pwritev(bs->file->bs, sector_num * BDRV_SECTOR_SIZE,

                          nb_sectors * BDRV_SECTOR_SIZE, qiov, flags);



fail:

    if (qiov == &local_qiov) {

        qemu_iovec_destroy(&local_qiov);

    }

    qemu_vfree(buf);

    return ret;

}
