static void bitband_writel(void *opaque, target_phys_addr_t offset,

                           uint32_t value)

{

    uint32_t addr;

    uint32_t mask;

    uint32_t v;

    addr = bitband_addr(opaque, offset) & ~3;

    mask = (1 << ((offset >> 2) & 31));

    mask = tswap32(mask);

    cpu_physical_memory_read(addr, (uint8_t *)&v, 4);

    if (value & 1)

        v |= mask;

    else

        v &= ~mask;

    cpu_physical_memory_write(addr, (uint8_t *)&v, 4);

}
