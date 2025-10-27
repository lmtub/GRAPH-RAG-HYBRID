static void bdrv_co_em_bh(void *opaque)

{

    BlockAIOCBCoroutine *acb = opaque;



    acb->common.cb(acb->common.opaque, acb->req.error);



    qemu_bh_delete(acb->bh);

    qemu_aio_unref(acb);

}
