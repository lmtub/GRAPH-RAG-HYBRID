static uint32_t e1000e_macreg_read(e1000e_device *d, uint32_t reg)

{

    return qpci_io_readl(d->pci_dev, d->mac_regs + reg);

}
