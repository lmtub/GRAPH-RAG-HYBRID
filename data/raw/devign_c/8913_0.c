int64_t bdrv_get_block_status(BlockDriverState *bs, int64_t sector_num,

                              int nb_sectors, int *pnum)

{

    Coroutine *co;

    BdrvCoGetBlockStatusData data = {

        .bs = bs,

        .sector_num = sector_num,

        .nb_sectors = nb_sectors,

        .pnum = pnum,

        .done = false,

    };



    if (qemu_in_coroutine()) {

        /* Fast-path if already in coroutine context */

        bdrv_get_block_status_co_entry(&data);

    } else {

        co = qemu_coroutine_create(bdrv_get_block_status_co_entry);

        qemu_coroutine_enter(co, &data);

        while (!data.done) {

            qemu_aio_wait();

        }

    }

    return data.ret;

}
