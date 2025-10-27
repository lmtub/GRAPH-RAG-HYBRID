static uint32_t dp8393x_readb(void *opaque, target_phys_addr_t addr)

{

    uint16_t v = dp8393x_readw(opaque, addr & ~0x1);

    return (v >> (8 * (addr & 0x1))) & 0xff;

}
