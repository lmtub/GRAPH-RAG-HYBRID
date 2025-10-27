static void reschedule_dma(void *opaque)

{

    DMAAIOCB *dbs = (DMAAIOCB *)opaque;



    qemu_bh_delete(dbs->bh);

    dbs->bh = NULL;

    dma_bdrv_cb(opaque, 0);

}
