BlockAIOCB *dma_bdrv_read(BlockDriverState *bs,

                          QEMUSGList *sg, uint64_t sector,

                          void (*cb)(void *opaque, int ret), void *opaque)

{

    return dma_bdrv_io(bs, sg, sector, bdrv_aio_readv, cb, opaque,

                       DMA_DIRECTION_FROM_DEVICE);

}
