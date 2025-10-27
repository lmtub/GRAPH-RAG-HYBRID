static void qvirtio_pci_virtqueue_kick(QVirtioDevice *d, QVirtQueue *vq)

{

    QVirtioPCIDevice *dev = (QVirtioPCIDevice *)d;

    qpci_io_writew(dev->pdev, dev->addr + VIRTIO_PCI_QUEUE_NOTIFY, vq->index);

}
