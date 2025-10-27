static int coroutine_fn raw_co_pwritev(BlockDriverState *bs, uint64_t offset,

                                       uint64_t bytes, QEMUIOVector *qiov,

                                       int flags)

{

    BDRVRawState *s = bs->opaque;

    void *buf = NULL;

    BlockDriver *drv;

    QEMUIOVector local_qiov;

    int ret;



    if (s->has_size && (offset > s->size || bytes > (s->size - offset))) {

        /* There's not enough space for the data. Don't write anything and just

         * fail to prevent leaking out of the size specified in options. */

        return -ENOSPC;

    }



    if (offset > UINT64_MAX - s->offset) {

        ret = -EINVAL;

        goto fail;

    }



    if (bs->probed && offset < BLOCK_PROBE_BUF_SIZE && bytes) {

        /* Handling partial writes would be a pain - so we just

         * require that guests have 512-byte request alignment if

         * probing occurred */

        QEMU_BUILD_BUG_ON(BLOCK_PROBE_BUF_SIZE != 512);

        QEMU_BUILD_BUG_ON(BDRV_SECTOR_SIZE != 512);

        assert(offset == 0 && bytes >= BLOCK_PROBE_BUF_SIZE);



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



    offset += s->offset;



    BLKDBG_EVENT(bs->file, BLKDBG_WRITE_AIO);

    ret = bdrv_co_pwritev(bs->file, offset, bytes, qiov, flags);



fail:

    if (qiov == &local_qiov) {

        qemu_iovec_destroy(&local_qiov);

    }

    qemu_vfree(buf);

    return ret;

}
