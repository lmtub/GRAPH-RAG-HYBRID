static uint32_t qvirtio_pci_get_features(QVirtioDevice *d)

{

    QVirtioPCIDevice *dev = (QVirtioPCIDevice *)d;

    return qpci_io_readl(dev->pdev, dev->addr + VIRTIO_PCI_HOST_FEATURES);

}
