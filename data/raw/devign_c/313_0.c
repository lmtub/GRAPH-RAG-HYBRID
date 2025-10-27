void bdrv_aio_cancel(BlockAIOCB *acb)

{

    qemu_aio_ref(acb);

    bdrv_aio_cancel_async(acb);

    while (acb->refcnt > 1) {

        if (acb->aiocb_info->get_aio_context) {

            aio_poll(acb->aiocb_info->get_aio_context(acb), true);

        } else if (acb->bs) {

            aio_poll(bdrv_get_aio_context(acb->bs), true);

        } else {

            abort();

        }

    }

    qemu_aio_unref(acb);

}
