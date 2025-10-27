static uint32_t gic_dist_readl(void *opaque, target_phys_addr_t offset)

{

    uint32_t val;

    val = gic_dist_readw(opaque, offset);

    val |= gic_dist_readw(opaque, offset + 2) << 16;

    return val;

}
