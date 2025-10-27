static void bdrv_aio_bh_cb(void *opaque)

{

    BlockAIOCBSync *acb = opaque;



    if (!acb->is_write && acb->ret >= 0) {

        qemu_iovec_from_buf(acb->qiov, 0, acb->bounce, acb->qiov->size);

    }

    qemu_vfree(acb->bounce);

    acb->common.cb(acb->common.opaque, acb->ret);

    qemu_bh_delete(acb->bh);

    acb->bh = NULL;

    qemu_aio_unref(acb);

}
