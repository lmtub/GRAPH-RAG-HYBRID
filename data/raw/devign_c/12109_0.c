int bdrv_save_vmstate(BlockDriverState *bs, const uint8_t *buf,

                      int64_t pos, int size)

{

    QEMUIOVector qiov;

    struct iovec iov = {

        .iov_base   = (void *) buf,

        .iov_len    = size,

    };



    qemu_iovec_init_external(&qiov, &iov, 1);

    return bdrv_writev_vmstate(bs, &qiov, pos);

}
