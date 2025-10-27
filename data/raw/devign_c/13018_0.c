static uint32_t slow_bar_readw(void *opaque, target_phys_addr_t addr)

{

    AssignedDevRegion *d = opaque;

    uint16_t *in = (uint16_t *)(d->u.r_virtbase + addr);

    uint32_t r;



    r = *in;

    DEBUG("slow_bar_readl addr=0x" TARGET_FMT_plx " val=0x%08x\n", addr, r);



    return r;

}
