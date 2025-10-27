int iommu_dma_memory_set(DMAContext *dma, dma_addr_t addr, uint8_t c,

                         dma_addr_t len)

{

    target_phys_addr_t paddr, plen;

    int err;



#ifdef DEBUG_IOMMU

    fprintf(stderr, "dma_memory_set context=%p addr=0x" DMA_ADDR_FMT

            " len=0x" DMA_ADDR_FMT "\n", dma, addr, len);

#endif



    while (len) {

        err = dma->translate(dma, addr, &paddr, &plen,

                             DMA_DIRECTION_FROM_DEVICE);

        if (err) {

            return err;

        }



        /* The translation might be valid for larger regions. */

        if (plen > len) {

            plen = len;

        }



        do_dma_memory_set(dma->as, paddr, c, plen);



        len -= plen;

        addr += plen;

    }



    return 0;

}
