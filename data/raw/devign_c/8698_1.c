static void virtio_pci_reset(DeviceState *qdev)

{

    VirtIOPCIProxy *proxy = VIRTIO_PCI(qdev);

    VirtioBusState *bus = VIRTIO_BUS(&proxy->bus);

    int i;



    virtio_pci_stop_ioeventfd(proxy);

    virtio_bus_reset(bus);

    msix_unuse_all_vectors(&proxy->pci_dev);



    for (i = 0; i < VIRTIO_QUEUE_MAX; i++) {

        proxy->vqs[i].enabled = 0;





    }

}