static void slow_bar_writeb(void *opaque, target_phys_addr_t addr, uint32_t val)

{

    AssignedDevRegion *d = opaque;

    uint8_t *out = d->u.r_virtbase + addr;



    DEBUG("slow_bar_writeb addr=0x" TARGET_FMT_plx " val=0x%02x\n", addr, val);

    *out = val;

}
