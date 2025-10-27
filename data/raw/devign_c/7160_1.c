static void qvirtio_pci_queue_select(QVirtioDevice *d, uint16_t index)

{

    QVirtioPCIDevice *dev = (QVirtioPCIDevice *)d;

    qpci_io_writeb(dev->pdev, dev->addr + VIRTIO_PCI_QUEUE_SEL, index);

}
