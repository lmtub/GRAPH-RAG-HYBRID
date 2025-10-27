int bdrv_discard(BlockDriverState *bs, int64_t sector_num, int nb_sectors)

{

    Coroutine *co;

    DiscardCo rwco = {

        .bs = bs,

        .sector_num = sector_num,

        .nb_sectors = nb_sectors,

        .ret = NOT_DONE,

    };



    if (qemu_in_coroutine()) {

        /* Fast-path if already in coroutine context */

        bdrv_discard_co_entry(&rwco);

    } else {

        co = qemu_coroutine_create(bdrv_discard_co_entry);

        qemu_coroutine_enter(co, &rwco);

        while (rwco.ret == NOT_DONE) {

            qemu_aio_wait();

        }

    }



    return rwco.ret;

}
