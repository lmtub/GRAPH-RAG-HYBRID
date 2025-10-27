static int bdrv_read_em(BlockDriverState *bs, int64_t sector_num,

                        uint8_t *buf, int nb_sectors)

{

    int async_ret;

    BlockDriverAIOCB *acb;



    async_ret = NOT_DONE;

    qemu_aio_wait_start();

    acb = bdrv_aio_read(bs, sector_num, buf, nb_sectors,

                        bdrv_rw_em_cb, &async_ret);

    if (acb == NULL) {

        qemu_aio_wait_end();

        return -1;

    }

    while (async_ret == NOT_DONE) {

        qemu_aio_wait();

    }

    qemu_aio_wait_end();

    return async_ret;

}
