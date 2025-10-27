int spapr_tce_dma_zero(VIOsPAPRDevice *dev, uint64_t taddr, uint32_t size)

{

    /* FIXME: allocating a temp buffer is nasty, but just stepping

     * through writing zeroes is awkward.  This will do for now. */

    uint8_t zeroes[size];



#ifdef DEBUG_TCE

    fprintf(stderr, "spapr_tce_dma_zero taddr=0x%llx size=0x%x\n",

            (unsigned long long)taddr, size);

#endif



    memset(zeroes, 0, size);

    return spapr_tce_dma_write(dev, taddr, zeroes, size);

}
