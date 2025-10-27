static void ide_trim_bh_cb(void *opaque)

{

    TrimAIOCB *iocb = opaque;



    iocb->common.cb(iocb->common.opaque, iocb->ret);



    qemu_bh_delete(iocb->bh);

    iocb->bh = NULL;

    qemu_aio_unref(iocb);

}
