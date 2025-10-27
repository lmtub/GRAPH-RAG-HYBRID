static int virtio_balloon_init_pci(PCIDevice *pci_dev)

{

    VirtIOPCIProxy *proxy = DO_UPCAST(VirtIOPCIProxy, pci_dev, pci_dev);

    VirtIODevice *vdev;



    vdev = virtio_balloon_init(&pci_dev->qdev);




    virtio_init_pci(proxy, vdev);

    return 0;
