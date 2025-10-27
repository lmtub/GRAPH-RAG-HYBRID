static uint32_t dp8393x_readl(void *opaque, target_phys_addr_t addr)

{

    uint32_t v;

    v = dp8393x_readw(opaque, addr);

    v |= dp8393x_readw(opaque, addr + 2) << 16;

    return v;

}
