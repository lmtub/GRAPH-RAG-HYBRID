static const char *addr2str(target_phys_addr_t addr)

{

    return nr2str(ehci_mmio_names, ARRAY_SIZE(ehci_mmio_names), addr);

}
