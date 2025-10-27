static uint64_t bmdma_addr_read(void *opaque, target_phys_addr_t addr,

                                unsigned width)

{

    BMDMAState *bm = opaque;

    uint32_t mask = (1ULL << (width * 8)) - 1;

    uint64_t data;



    data = (bm->addr >> (addr * 8)) & mask;

#ifdef DEBUG_IDE

    printf("%s: 0x%08x\n", __func__, (unsigned)*data);

#endif

    return data;

}
