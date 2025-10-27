static void slow_bar_writel(void *opaque, target_phys_addr_t addr, uint32_t val)

{

    AssignedDevRegion *d = opaque;

    uint32_t *out = (uint32_t *)(d->u.r_virtbase + addr);



    DEBUG("slow_bar_writel addr=0x" TARGET_FMT_plx " val=0x%08x\n", addr, val);

    *out = val;

}
