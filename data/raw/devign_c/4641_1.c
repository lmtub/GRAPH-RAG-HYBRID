void *qpci_legacy_iomap(QPCIDevice *dev, uint16_t addr)

{

    return (void *)(uintptr_t)addr;

}
