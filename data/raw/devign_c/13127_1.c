int spapr_tce_dma_read(VIOsPAPRDevice *dev, uint64_t taddr, void *buf,
                       uint32_t size)
{
#ifdef DEBUG_TCE
    fprintf(stderr, "spapr_tce_dma_write taddr=0x%llx size=0x%x\n",
            (unsigned long long)taddr, size);
#endif
    while (size) {
        uint64_t tce;
        uint32_t lsize;
        uint64_t txaddr;
        /* Check if we are in bound */
        if (taddr >= dev->rtce_window_size) {
#ifdef DEBUG_TCE
            fprintf(stderr, "spapr_tce_dma_read out of bounds\n");
#endif
            return H_DEST_PARM;
        tce = dev->rtce_table[taddr >> SPAPR_VIO_TCE_PAGE_SHIFT].tce;
        /* How much til end of page ? */
        lsize = MIN(size, ((~taddr) & SPAPR_VIO_TCE_PAGE_MASK) + 1);
        /* Check TCE */
        if (!(tce & 1)) {
            return H_DEST_PARM;
        /* Translate */
        txaddr = (tce & ~SPAPR_VIO_TCE_PAGE_MASK) |
            (taddr & SPAPR_VIO_TCE_PAGE_MASK);
#ifdef DEBUG_TCE
        fprintf(stderr, " -> write to txaddr=0x%llx, size=0x%x\n",
                (unsigned long long)txaddr, lsize);
#endif
        /* Do it */
        cpu_physical_memory_read(txaddr, buf, lsize);
        buf += lsize;
        taddr += lsize;
        size -= lsize;
    return H_SUCCESS;