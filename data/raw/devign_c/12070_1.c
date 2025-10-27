bool qpci_msix_masked(QPCIDevice *dev, uint16_t entry)

{

    uint8_t addr;

    uint16_t val;

    void *vector_addr = dev->msix_table + (entry * PCI_MSIX_ENTRY_SIZE);



    g_assert(dev->msix_enabled);

    addr = qpci_find_capability(dev, PCI_CAP_ID_MSIX);

    g_assert_cmphex(addr, !=, 0);

    val = qpci_config_readw(dev, addr + PCI_MSIX_FLAGS);



    if (val & PCI_MSIX_FLAGS_MASKALL) {

        return true;

    } else {

        return (qpci_io_readl(dev, vector_addr + PCI_MSIX_ENTRY_VECTOR_CTRL)

                                            & PCI_MSIX_ENTRY_CTRL_MASKBIT) != 0;

    }

}
