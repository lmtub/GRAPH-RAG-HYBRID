static void ide_sector_start_dma(IDEState *s, enum ide_dma_cmd dma_cmd)

{

    s->status = READY_STAT | SEEK_STAT | DRQ_STAT | BUSY_STAT;

    s->io_buffer_index = 0;

    s->io_buffer_size = 0;

    s->dma_cmd = dma_cmd;



    switch (dma_cmd) {

    case IDE_DMA_READ:

        block_acct_start(bdrv_get_stats(s->bs), &s->acct,

                         s->nsector * BDRV_SECTOR_SIZE, BLOCK_ACCT_READ);

        break;

    case IDE_DMA_WRITE:

        block_acct_start(bdrv_get_stats(s->bs), &s->acct,

                         s->nsector * BDRV_SECTOR_SIZE, BLOCK_ACCT_WRITE);

        break;

    default:

        break;

    }



    ide_start_dma(s, ide_dma_cb);

}
