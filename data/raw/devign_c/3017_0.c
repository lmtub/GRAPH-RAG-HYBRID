static void slow_bar_writew(void *opaque, target_phys_addr_t addr, uint32_t val)

{

    AssignedDevRegion *d = opaque;

    uint16_t *out = (uint16_t *)(d->u.r_virtbase + addr);



    DEBUG("slow_bar_writew addr=0x" TARGET_FMT_plx " val=0x%04x\n", addr, val);

    *out = val;

}
