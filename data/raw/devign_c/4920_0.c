static int bdrv_prwv_co(BlockDriverState *bs, int64_t offset,

                        QEMUIOVector *qiov, bool is_write,

                        BdrvRequestFlags flags)

{

    Coroutine *co;

    RwCo rwco = {

        .bs = bs,

        .offset = offset,

        .qiov = qiov,

        .is_write = is_write,

        .ret = NOT_DONE,

        .flags = flags,

    };



    /**

     * In sync call context, when the vcpu is blocked, this throttling timer

     * will not fire; so the I/O throttling function has to be disabled here

     * if it has been enabled.

     */

    if (bs->io_limits_enabled) {

        fprintf(stderr, "Disabling I/O throttling on '%s' due "

                        "to synchronous I/O.\n", bdrv_get_device_name(bs));

        bdrv_io_limits_disable(bs);

    }



    if (qemu_in_coroutine()) {

        /* Fast-path if already in coroutine context */

        bdrv_rw_co_entry(&rwco);

    } else {

        AioContext *aio_context = bdrv_get_aio_context(bs);



        co = qemu_coroutine_create(bdrv_rw_co_entry);

        qemu_coroutine_enter(co, &rwco);

        while (rwco.ret == NOT_DONE) {

            aio_poll(aio_context, true);

        }

    }

    return rwco.ret;

}
