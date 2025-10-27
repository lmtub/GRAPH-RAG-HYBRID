void msix_load(PCIDevice *dev, QEMUFile *f)

{

    unsigned n = dev->msix_entries_nr;

    unsigned int vector;



    if (!(dev->cap_present & QEMU_PCI_CAP_MSIX)) {

        return;

    }



    msix_free_irq_entries(dev);

    qemu_get_buffer(f, dev->msix_table_page, n * PCI_MSIX_ENTRY_SIZE);

    qemu_get_buffer(f, dev->msix_table_page + MSIX_PAGE_PENDING, (n + 7) / 8);

    msix_update_function_masked(dev);



    for (vector = 0; vector < n; vector++) {

        msix_handle_mask_update(dev, vector, true);

    }

}
