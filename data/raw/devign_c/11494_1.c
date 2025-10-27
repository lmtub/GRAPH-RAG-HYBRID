void qpci_msix_enable(QPCIDevice *dev)

{

    uint8_t addr;

    uint16_t val;

    uint32_t table;

    uint8_t bir_table;

    uint8_t bir_pba;

    void *offset;



    addr = qpci_find_capability(dev, PCI_CAP_ID_MSIX);

    g_assert_cmphex(addr, !=, 0);



    val = qpci_config_readw(dev, addr + PCI_MSIX_FLAGS);

    qpci_config_writew(dev, addr + PCI_MSIX_FLAGS, val | PCI_MSIX_FLAGS_ENABLE);



    table = qpci_config_readl(dev, addr + PCI_MSIX_TABLE);

    bir_table = table & PCI_MSIX_FLAGS_BIRMASK;

    offset = qpci_iomap(dev, bir_table, NULL);

    dev->msix_table = offset + (table & ~PCI_MSIX_FLAGS_BIRMASK);



    table = qpci_config_readl(dev, addr + PCI_MSIX_PBA);

    bir_pba = table & PCI_MSIX_FLAGS_BIRMASK;

    if (bir_pba != bir_table) {

        offset = qpci_iomap(dev, bir_pba, NULL);

    }

    dev->msix_pba = offset + (table & ~PCI_MSIX_FLAGS_BIRMASK);



    g_assert(dev->msix_table != NULL);

    g_assert(dev->msix_pba != NULL);

    dev->msix_enabled = true;

}
