static void msix_mmio_write(void *opaque, target_phys_addr_t addr,

                            uint64_t val, unsigned size)

{

    PCIDevice *dev = opaque;

    unsigned int offset = addr & (MSIX_PAGE_SIZE - 1) & ~0x3;

    int vector = offset / PCI_MSIX_ENTRY_SIZE;



    /* MSI-X page includes a read-only PBA and a writeable Vector Control. */

    if (vector >= dev->msix_entries_nr) {

        return;

    }



    pci_set_long(dev->msix_table_page + offset, val);

    msix_handle_mask_update(dev, vector);

}
