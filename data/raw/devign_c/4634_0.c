static void bdrv_co_complete(BlockAIOCBCoroutine *acb)

{

    if (!acb->need_bh) {

        acb->common.cb(acb->common.opaque, acb->req.error);

        qemu_aio_unref(acb);

    }

}
