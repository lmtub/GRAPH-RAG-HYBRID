static uint64_t pci_config_get_memory_base(PCIDevice *d, uint32_t base)

{

    return ((uint64_t)pci_get_word(d->config + base) & PCI_MEMORY_RANGE_MASK)

        << 16;

}
