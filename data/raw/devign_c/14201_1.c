BlockDriverAIOCB *dma_bdrv_io(

    BlockDriverState *bs, QEMUSGList *sg, uint64_t sector_num,

    DMAIOFunc *io_func, BlockDriverCompletionFunc *cb,

    void *opaque, bool to_dev)

{

    DMAAIOCB *dbs = qemu_aio_get(&dma_aio_pool, bs, cb, opaque);



    dbs->acb = NULL;

    dbs->bs = bs;

    dbs->sg = sg;

    dbs->sector_num = sector_num;

    dbs->sg_cur_index = 0;

    dbs->sg_cur_byte = 0;

    dbs->to_dev = to_dev;

    dbs->io_func = io_func;

    dbs->bh = NULL;

    qemu_iovec_init(&dbs->iov, sg->nsg);

    dma_bdrv_cb(dbs, 0);

    if (!dbs->acb) {

        qemu_aio_release(dbs);

        return NULL;

    }

    return &dbs->common;

}
