static void dma_aio_cancel(BlockAIOCB *acb)

{

    DMAAIOCB *dbs = container_of(acb, DMAAIOCB, common);



    trace_dma_aio_cancel(dbs);



    if (dbs->acb) {

        bdrv_aio_cancel_async(dbs->acb);

    }

}
