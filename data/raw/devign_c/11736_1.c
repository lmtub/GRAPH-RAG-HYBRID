static void virtio_console_init_pci(PCIDevice *pci_dev)

{

    VirtIOPCIProxy *proxy = DO_UPCAST(VirtIOPCIProxy, pci_dev, pci_dev);

    VirtIODevice *vdev;



    vdev = virtio_console_init(&pci_dev->qdev);

    virtio_init_pci(proxy, vdev,

                    PCI_VENDOR_ID_REDHAT_QUMRANET,

                    PCI_DEVICE_ID_VIRTIO_CONSOLE,

                    PCI_CLASS_DISPLAY_OTHER,

                    0x00);

}
