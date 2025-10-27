static uint32_t isa_mmio_readw(void *opaque, target_phys_addr_t addr)

{

    return cpu_inw(addr & IOPORTS_MASK);

}
