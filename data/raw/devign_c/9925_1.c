static uint8_t qvirtio_pci_get_status(QVirtioDevice *d)

{

    QVirtioPCIDevice *dev = (QVirtioPCIDevice *)d;

    return qpci_io_readb(dev->pdev, dev->addr + VIRTIO_PCI_STATUS);

}
