static void dma_aio_cancel(BlockDriverAIOCB *acb)

{

    DMAAIOCB *dbs = container_of(acb, DMAAIOCB, common);



    if (dbs->acb) {

        bdrv_aio_cancel(dbs->acb);

    }

}
