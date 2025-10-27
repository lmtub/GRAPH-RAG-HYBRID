static uint32_t gic_dist_readw(void *opaque, target_phys_addr_t offset)

{

    uint32_t val;

    val = gic_dist_readb(opaque, offset);

    val |= gic_dist_readb(opaque, offset + 1) << 8;

    return val;

}
