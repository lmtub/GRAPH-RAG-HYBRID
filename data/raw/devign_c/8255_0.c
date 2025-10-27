static int bdrv_write_em(BlockDriverState *bs, int64_t sector_num,

                         const uint8_t *buf, int nb_sectors)

{

    int async_ret;

    BlockDriverAIOCB *acb;

    struct iovec iov;

    QEMUIOVector qiov;



    async_ret = NOT_DONE;

    iov.iov_base = (void *)buf;

    iov.iov_len = nb_sectors * BDRV_SECTOR_SIZE;

    qemu_iovec_init_external(&qiov, &iov, 1);



    acb = bs->drv->bdrv_aio_writev(bs, sector_num, &qiov, nb_sectors,

                                   bdrv_rw_em_cb, &async_ret);

    if (acb == NULL) {

        async_ret = -1;

        goto fail;

    }

    while (async_ret == NOT_DONE) {

        qemu_aio_wait();

    }



fail:

    return async_ret;

}
