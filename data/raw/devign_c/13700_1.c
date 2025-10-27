static int bdrv_prwv_co(BdrvChild *child, int64_t offset,

                        QEMUIOVector *qiov, bool is_write,

                        BdrvRequestFlags flags)

{

    Coroutine *co;

    RwCo rwco = {

        .child = child,

        .offset = offset,

        .qiov = qiov,

        .is_write = is_write,

        .ret = NOT_DONE,

        .flags = flags,

    };



    if (qemu_in_coroutine()) {

        /* Fast-path if already in coroutine context */

        bdrv_rw_co_entry(&rwco);

    } else {

        AioContext *aio_context = bdrv_get_aio_context(child->bs);



        co = qemu_coroutine_create(bdrv_rw_co_entry);

        qemu_coroutine_enter(co, &rwco);

        while (rwco.ret == NOT_DONE) {

            aio_poll(aio_context, true);

        }

    }

    return rwco.ret;

}
