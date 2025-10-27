static void qvirtio_pci_set_status(QVirtioDevice *d, uint8_t status)

{

    QVirtioPCIDevice *dev = (QVirtioPCIDevice *)d;

    qpci_io_writeb(dev->pdev, dev->addr + VIRTIO_PCI_STATUS, status);

}
