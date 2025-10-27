static void dp8393x_writel(void *opaque, target_phys_addr_t addr, uint32_t val)

{

    dp8393x_writew(opaque, addr, val & 0xffff);

    dp8393x_writew(opaque, addr + 2, (val >> 16) & 0xffff);

}
