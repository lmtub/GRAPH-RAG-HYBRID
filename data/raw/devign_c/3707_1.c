static int qcow_write(BlockDriverState *bs, int64_t sector_num,

                      const uint8_t *buf, int nb_sectors)

{

    Coroutine *co;

    AioContext *aio_context = bdrv_get_aio_context(bs);

    QcowWriteCo data = {

        .bs         = bs,

        .sector_num = sector_num,

        .buf        = buf,

        .nb_sectors = nb_sectors,

        .ret        = -EINPROGRESS,

    };

    co = qemu_coroutine_create(qcow_write_co_entry);

    qemu_coroutine_enter(co, &data);

    while (data.ret == -EINPROGRESS) {

        aio_poll(aio_context, true);

    }

    return data.ret;

}
