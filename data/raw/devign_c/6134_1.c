bool qpci_msix_pending(QPCIDevice *dev, uint16_t entry)

{

    uint32_t pba_entry;

    uint8_t bit_n = entry % 32;

    void *addr = dev->msix_pba + (entry / 32) * PCI_MSIX_ENTRY_SIZE / 4;



    g_assert(dev->msix_enabled);

    pba_entry = qpci_io_readl(dev, addr);

    qpci_io_writel(dev, addr, pba_entry & ~(1 << bit_n));

    return (pba_entry & (1 << bit_n)) != 0;

}
