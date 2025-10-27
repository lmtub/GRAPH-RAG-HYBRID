void qvirtio_pci_set_msix_configuration_vector(QVirtioPCIDevice *d,

                                        QGuestAllocator *alloc, uint16_t entry)

{

    uint16_t vector;

    uint32_t control;

    void *addr;



    g_assert(d->pdev->msix_enabled);

    addr = d->pdev->msix_table + (entry * 16);



    g_assert_cmpint(entry, >=, 0);

    g_assert_cmpint(entry, <, qpci_msix_table_size(d->pdev));

    d->config_msix_entry = entry;



    d->config_msix_data = 0x12345678;

    d->config_msix_addr = guest_alloc(alloc, 4);



    qpci_io_writel(d->pdev, addr + PCI_MSIX_ENTRY_LOWER_ADDR,

                                                    d->config_msix_addr & ~0UL);

    qpci_io_writel(d->pdev, addr + PCI_MSIX_ENTRY_UPPER_ADDR,

                                            (d->config_msix_addr >> 32) & ~0UL);

    qpci_io_writel(d->pdev, addr + PCI_MSIX_ENTRY_DATA, d->config_msix_data);



    control = qpci_io_readl(d->pdev, addr + PCI_MSIX_ENTRY_VECTOR_CTRL);

    qpci_io_writel(d->pdev, addr + PCI_MSIX_ENTRY_VECTOR_CTRL,

                                        control & ~PCI_MSIX_ENTRY_CTRL_MASKBIT);



    qpci_io_writew(d->pdev, d->addr + VIRTIO_MSI_CONFIG_VECTOR, entry);

    vector = qpci_io_readw(d->pdev, d->addr + VIRTIO_MSI_CONFIG_VECTOR);

    g_assert_cmphex(vector, !=, VIRTIO_MSI_NO_VECTOR);

}
