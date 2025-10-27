void qpci_msix_disable(QPCIDevice *dev)

{

    uint8_t addr;

    uint16_t val;



    g_assert(dev->msix_enabled);

    addr = qpci_find_capability(dev, PCI_CAP_ID_MSIX);

    g_assert_cmphex(addr, !=, 0);

    val = qpci_config_readw(dev, addr + PCI_MSIX_FLAGS);

    qpci_config_writew(dev, addr + PCI_MSIX_FLAGS,

                                                val & ~PCI_MSIX_FLAGS_ENABLE);



    qpci_iounmap(dev, dev->msix_table_bar);

    qpci_iounmap(dev, dev->msix_pba_bar);

    dev->msix_enabled = 0;

    dev->msix_table_off = 0;

    dev->msix_pba_off = 0;

}
