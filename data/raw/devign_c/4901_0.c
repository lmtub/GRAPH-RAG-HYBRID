static uint32_t pci_apb_ioreadw (void *opaque, target_phys_addr_t addr)

{

    uint32_t val;



    val = bswap16(cpu_inw(addr & IOPORTS_MASK));

    return val;

}
