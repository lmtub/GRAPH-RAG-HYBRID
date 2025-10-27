static void dma_complete(DMAAIOCB *dbs, int ret)

{

    trace_dma_complete(dbs, ret, dbs->common.cb);



    dma_bdrv_unmap(dbs);

    if (dbs->common.cb) {

        dbs->common.cb(dbs->common.opaque, ret);

    }

    qemu_iovec_destroy(&dbs->iov);

    if (dbs->bh) {

        qemu_bh_delete(dbs->bh);

        dbs->bh = NULL;

    }

    qemu_aio_unref(dbs);

}
