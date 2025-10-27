static int coroutine_fn bdrv_co_do_copy_on_readv(BdrvChild *child,

        int64_t offset, unsigned int bytes, QEMUIOVector *qiov)

{

    BlockDriverState *bs = child->bs;



    /* Perform I/O through a temporary buffer so that users who scribble over

     * their read buffer while the operation is in progress do not end up

     * modifying the image file.  This is critical for zero-copy guest I/O

     * where anything might happen inside guest memory.

     */

    void *bounce_buffer;



    BlockDriver *drv = bs->drv;

    struct iovec iov;

    QEMUIOVector bounce_qiov;

    int64_t cluster_offset;

    unsigned int cluster_bytes;

    size_t skip_bytes;

    int ret;



    assert(child->perm & (BLK_PERM_WRITE_UNCHANGED | BLK_PERM_WRITE));



    /* Cover entire cluster so no additional backing file I/O is required when

     * allocating cluster in the image file.

     */

    bdrv_round_to_clusters(bs, offset, bytes, &cluster_offset, &cluster_bytes);



    trace_bdrv_co_do_copy_on_readv(bs, offset, bytes,

                                   cluster_offset, cluster_bytes);



    iov.iov_len = cluster_bytes;

    iov.iov_base = bounce_buffer = qemu_try_blockalign(bs, iov.iov_len);

    if (bounce_buffer == NULL) {

        ret = -ENOMEM;

        goto err;

    }



    qemu_iovec_init_external(&bounce_qiov, &iov, 1);



    ret = bdrv_driver_preadv(bs, cluster_offset, cluster_bytes,

                             &bounce_qiov, 0);

    if (ret < 0) {

        goto err;

    }



    if (drv->bdrv_co_pwrite_zeroes &&

        buffer_is_zero(bounce_buffer, iov.iov_len)) {

        /* FIXME: Should we (perhaps conditionally) be setting

         * BDRV_REQ_MAY_UNMAP, if it will allow for a sparser copy

         * that still correctly reads as zero? */

        ret = bdrv_co_do_pwrite_zeroes(bs, cluster_offset, cluster_bytes, 0);

    } else {

        /* This does not change the data on the disk, it is not necessary

         * to flush even in cache=writethrough mode.

         */

        ret = bdrv_driver_pwritev(bs, cluster_offset, cluster_bytes,

                                  &bounce_qiov, 0);

    }



    if (ret < 0) {

        /* It might be okay to ignore write errors for guest requests.  If this

         * is a deliberate copy-on-read then we don't want to ignore the error.

         * Simply report it in all cases.

         */

        goto err;

    }



    skip_bytes = offset - cluster_offset;

    qemu_iovec_from_buf(qiov, 0, bounce_buffer + skip_bytes, bytes);



err:

    qemu_vfree(bounce_buffer);

    return ret;

}
