static void msix_table_mmio_write(void *opaque, target_phys_addr_t addr,

                                  uint64_t val, unsigned size)

{

    PCIDevice *dev = opaque;

    int vector = addr / PCI_MSIX_ENTRY_SIZE;

    bool was_masked;



    was_masked = msix_is_masked(dev, vector);

    pci_set_long(dev->msix_table + addr, val);

    msix_handle_mask_update(dev, vector, was_masked);

}
