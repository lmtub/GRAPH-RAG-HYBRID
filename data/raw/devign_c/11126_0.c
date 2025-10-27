static uint32_t pci_apb_ioreadl (void *opaque, target_phys_addr_t addr)

{

    uint32_t val;



    val = bswap32(cpu_inl(addr & IOPORTS_MASK));

    return val;

}
