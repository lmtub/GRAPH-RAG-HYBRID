void *iommu_dma_memory_map(DMAContext *dma, dma_addr_t addr, dma_addr_t *len,

                           DMADirection dir)

{

    int err;

    target_phys_addr_t paddr, plen;

    void *buf;



    if (dma->map) {

        return dma->map(dma, addr, len, dir);

    }



    plen = *len;

    err = dma->translate(dma, addr, &paddr, &plen, dir);

    if (err) {

        return NULL;

    }



    /*

     * If this is true, the virtual region is contiguous,

     * but the translated physical region isn't. We just

     * clamp *len, much like address_space_map() does.

     */

    if (plen < *len) {

        *len = plen;

    }



    buf = address_space_map(dma->as, paddr, &plen, dir == DMA_DIRECTION_FROM_DEVICE);

    *len = plen;



    return buf;

}
