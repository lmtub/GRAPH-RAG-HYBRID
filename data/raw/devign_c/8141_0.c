static void pci_apb_iowriteb (void *opaque, target_phys_addr_t addr,

                                  uint32_t val)

{

    cpu_outb(addr & IOPORTS_MASK, val);

}
