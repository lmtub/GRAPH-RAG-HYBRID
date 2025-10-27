int bdrv_pwrite(BlockDriverState *bs, int64_t offset,

                const void *buf, int bytes)

{

    QEMUIOVector qiov;

    struct iovec iov = {

        .iov_base   = (void *) buf,

        .iov_len    = bytes,

    };



    if (bytes < 0) {

        return -EINVAL;

    }



    qemu_iovec_init_external(&qiov, &iov, 1);

    return bdrv_pwritev(bs, offset, &qiov);

}
