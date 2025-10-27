int dma_memory_set(DMAContext *dma, dma_addr_t addr, uint8_t c, dma_addr_t len)
{
    if (dma_has_iommu(dma)) {
        return iommu_dma_memory_set(dma, addr, c, len);
    }
    do_dma_memory_set(addr, c, len);
    return 0;
}