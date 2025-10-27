static void pci_basic(void)

{

    QVirtioPCIDevice *dev;

    QPCIBus *bus;

    QVirtQueuePCI *vqpci;

    QGuestAllocator *alloc;

    void *addr;



    bus = pci_test_start();

    dev = virtio_blk_pci_init(bus, PCI_SLOT);



    alloc = pc_alloc_init();

    vqpci = (QVirtQueuePCI *)qvirtqueue_setup(&qvirtio_pci, &dev->vdev,

                                                                    alloc, 0);



    /* MSI-X is not enabled */

    addr = dev->addr + VIRTIO_PCI_CONFIG_OFF(false);



    test_basic(&qvirtio_pci, &dev->vdev, alloc, &vqpci->vq,

                                                    (uint64_t)(uintptr_t)addr);



    /* End test */

    guest_free(alloc, vqpci->vq.desc);

    pc_alloc_uninit(alloc);

    qvirtio_pci_device_disable(dev);

    g_free(dev);

    qpci_free_pc(bus);

    test_end();

}
