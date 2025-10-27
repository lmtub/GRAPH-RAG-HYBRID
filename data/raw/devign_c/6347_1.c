int bdrv_flush(BlockDriverState *bs)

{

    Coroutine *co;

    FlushCo flush_co = {

        .bs = bs,

        .ret = NOT_DONE,

    };



    if (qemu_in_coroutine()) {

        /* Fast-path if already in coroutine context */

        bdrv_flush_co_entry(&flush_co);

    } else {

        co = qemu_coroutine_create(bdrv_flush_co_entry, &flush_co);

        qemu_coroutine_enter(co);

        BDRV_POLL_WHILE(bs, flush_co.ret == NOT_DONE);

    }



    return flush_co.ret;

}
