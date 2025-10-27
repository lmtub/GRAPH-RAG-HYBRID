static void qvirtio_pci_set_features(QVirtioDevice *d, uint32_t features)

{

    QVirtioPCIDevice *dev = (QVirtioPCIDevice *)d;

    qpci_io_writel(dev->pdev, dev->addr + VIRTIO_PCI_GUEST_FEATURES, features);

}
