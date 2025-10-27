static void virtio_pci_reset(DeviceState *d)

{

    VirtIOPCIProxy *proxy = container_of(d, VirtIOPCIProxy, pci_dev.qdev);

    virtio_reset(proxy->vdev);

    msix_reset(&proxy->pci_dev);

    proxy->flags = 0;

}
