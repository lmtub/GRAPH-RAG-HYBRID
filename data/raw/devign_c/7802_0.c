int msix_uninit(PCIDevice *dev, MemoryRegion *bar)

{

    if (!(dev->cap_present & QEMU_PCI_CAP_MSIX))

        return 0;

    pci_del_capability(dev, PCI_CAP_ID_MSIX, MSIX_CAP_LENGTH);

    dev->msix_cap = 0;

    msix_free_irq_entries(dev);

    dev->msix_entries_nr = 0;

    memory_region_del_subregion(bar, &dev->msix_mmio);

    memory_region_destroy(&dev->msix_mmio);

    g_free(dev->msix_table_page);

    dev->msix_table_page = NULL;

    g_free(dev->msix_entry_used);

    dev->msix_entry_used = NULL;

    dev->cap_present &= ~QEMU_PCI_CAP_MSIX;

    return 0;

}
