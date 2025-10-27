static uint32_t bitband_readb(void *opaque, target_phys_addr_t offset)

{

    uint8_t v;

    cpu_physical_memory_read(bitband_addr(opaque, offset), &v, 1);

    return (v & (1 << ((offset >> 2) & 7))) != 0;

}
