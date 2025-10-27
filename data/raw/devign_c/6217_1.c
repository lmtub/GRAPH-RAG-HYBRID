static void pci_basic(gconstpointer data)

{

    QVirtioPCIDevice *dev;

    QPCIBus *bus;

    QVirtQueuePCI *tx, *rx;

    QGuestAllocator *alloc;

    void (*func) (const QVirtioBus *bus,

                  QVirtioDevice *dev,

                  QGuestAllocator *alloc,

                  QVirtQueue *rvq,

                  QVirtQueue *tvq,

                  int socket) = data;

    int sv[2], ret;



    ret = socketpair(PF_UNIX, SOCK_STREAM, 0, sv);

    g_assert_cmpint(ret, !=, -1);



    bus = pci_test_start(sv[1]);

    dev = virtio_net_pci_init(bus, PCI_SLOT);



    alloc = pc_alloc_init();

    rx = (QVirtQueuePCI *)qvirtqueue_setup(&qvirtio_pci, &dev->vdev,

                                           alloc, 0);

    tx = (QVirtQueuePCI *)qvirtqueue_setup(&qvirtio_pci, &dev->vdev,

                                           alloc, 1);



    driver_init(&qvirtio_pci, &dev->vdev);

    func(&qvirtio_pci, &dev->vdev, alloc, &rx->vq, &tx->vq, sv[0]);



    /* End test */

    close(sv[0]);

    guest_free(alloc, tx->vq.desc);

    pc_alloc_uninit(alloc);

    qvirtio_pci_device_disable(dev);

    g_free(dev);

    qpci_free_pc(bus);

    test_end();

}
