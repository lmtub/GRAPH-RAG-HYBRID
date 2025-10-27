void msix_reset(PCIDevice *dev)

{

    if (!(dev->cap_present & QEMU_PCI_CAP_MSIX))

        return;

    msix_free_irq_entries(dev);

    dev->config[dev->msix_cap + MSIX_CONTROL_OFFSET] &=

	    ~dev->wmask[dev->msix_cap + MSIX_CONTROL_OFFSET];

    memset(dev->msix_table_page, 0, MSIX_PAGE_SIZE);

    msix_mask_all(dev, dev->msix_entries_nr);

}
