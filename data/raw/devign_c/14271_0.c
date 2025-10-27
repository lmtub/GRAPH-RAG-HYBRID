void ide_dma_cancel(BMDMAState *bm)

{

    if (bm->status & BM_STATUS_DMAING) {

        bm->status &= ~BM_STATUS_DMAING;

        /* cancel DMA request */

        bm->unit = -1;

        bm->dma_cb = NULL;

        if (bm->aiocb) {

#ifdef DEBUG_AIO

            printf("aio_cancel\n");

#endif

            bdrv_aio_cancel(bm->aiocb);

            bm->aiocb = NULL;

        }

    }

}
