bool iommu_dma_memory_valid(DMAContext *dma, dma_addr_t addr, dma_addr_t len,

                            DMADirection dir)

{

    target_phys_addr_t paddr, plen;



#ifdef DEBUG_IOMMU

    fprintf(stderr, "dma_memory_check context=%p addr=0x" DMA_ADDR_FMT

            " len=0x" DMA_ADDR_FMT " dir=%d\n", dma, addr, len, dir);

#endif



    while (len) {

        if (dma->translate(dma, addr, &paddr, &plen, dir) != 0) {

            return false;

        }



        /* The translation might be valid for larger regions. */

        if (plen > len) {

            plen = len;

        }



        len -= plen;

        addr += plen;

    }



    return true;

}
