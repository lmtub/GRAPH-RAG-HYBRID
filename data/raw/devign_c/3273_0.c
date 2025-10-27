static void ide_dma_restart_cb(void *opaque, int running, int reason)

{

    BMDMAState *bm = opaque;

    if (!running)

        return;

    if (bm->status & BM_STATUS_DMA_RETRY) {

        bm->status &= ~BM_STATUS_DMA_RETRY;

        ide_dma_restart(bm->ide_if);

    } else if (bm->status & BM_STATUS_PIO_RETRY) {

        bm->status &= ~BM_STATUS_PIO_RETRY;

        ide_sector_write(bm->ide_if);

    }

}
