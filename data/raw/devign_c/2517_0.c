static void virtio_pci_reset(DeviceState *qdev)

{

    VirtIOPCIProxy *proxy = VIRTIO_PCI(qdev);

    VirtioBusState *bus = VIRTIO_BUS(&proxy->bus);

    virtio_pci_stop_ioeventfd(proxy);

    virtio_bus_reset(bus);

    msix_unuse_all_vectors(&proxy->pci_dev);

    proxy->flags &= ~VIRTIO_PCI_FLAG_BUS_MASTER_BUG;

}
