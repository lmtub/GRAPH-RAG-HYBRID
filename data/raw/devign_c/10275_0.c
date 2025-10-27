void msix_save(PCIDevice *dev, QEMUFile *f)

{

    unsigned n = dev->msix_entries_nr;



    if (!(dev->cap_present & QEMU_PCI_CAP_MSIX)) {

        return;

    }



    qemu_put_buffer(f, dev->msix_table_page, n * PCI_MSIX_ENTRY_SIZE);

    qemu_put_buffer(f, dev->msix_table_page + MSIX_PAGE_PENDING, (n + 7) / 8);

}
