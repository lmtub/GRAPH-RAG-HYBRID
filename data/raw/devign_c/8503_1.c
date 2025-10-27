static void pci_indirect(void)

{

    QVirtioPCIDevice *dev;

    QPCIBus *bus;

    QVirtQueuePCI *vqpci;

    QGuestAllocator *alloc;

    QVirtioBlkReq req;

    QVRingIndirectDesc *indirect;

    void *addr;

    uint64_t req_addr;

    uint64_t capacity;

    uint32_t features;

    uint32_t free_head;

    uint8_t status;

    char *data;



    bus = test_start();



    dev = virtio_blk_init(bus);



    /* MSI-X is not enabled */

    addr = dev->addr + QVIRTIO_DEVICE_SPECIFIC_NO_MSIX;



    capacity = qvirtio_config_readq(&qvirtio_pci, &dev->vdev, addr);

    g_assert_cmpint(capacity, ==, TEST_IMAGE_SIZE / 512);



    features = qvirtio_get_features(&qvirtio_pci, &dev->vdev);

    g_assert_cmphex(features & QVIRTIO_F_RING_INDIRECT_DESC, !=, 0);

    features = features & ~(QVIRTIO_F_BAD_FEATURE | QVIRTIO_F_RING_EVENT_IDX |

                                                            QVIRTIO_BLK_F_SCSI);

    qvirtio_set_features(&qvirtio_pci, &dev->vdev, features);



    alloc = pc_alloc_init();

    vqpci = (QVirtQueuePCI *)qvirtqueue_setup(&qvirtio_pci, &dev->vdev,

                                                                    alloc, 0);

    qvirtio_set_driver_ok(&qvirtio_pci, &dev->vdev);



    /* Write request */

    req.type = QVIRTIO_BLK_T_OUT;

    req.ioprio = 1;

    req.sector = 0;

    req.data = g_malloc0(512);

    strcpy(req.data, "TEST");



    req_addr = virtio_blk_request(alloc, &req, 512);



    g_free(req.data);



    indirect = qvring_indirect_desc_setup(&dev->vdev, alloc, 2);

    qvring_indirect_desc_add(indirect, req_addr, 528, false);

    qvring_indirect_desc_add(indirect, req_addr + 528, 1, true);

    free_head = qvirtqueue_add_indirect(&vqpci->vq, indirect);

    qvirtqueue_kick(&qvirtio_pci, &dev->vdev, &vqpci->vq, free_head);



    g_assert(qvirtio_wait_queue_isr(&qvirtio_pci, &dev->vdev, &vqpci->vq,

                                                        QVIRTIO_BLK_TIMEOUT));

    status = readb(req_addr + 528);

    g_assert_cmpint(status, ==, 0);



    g_free(indirect);

    guest_free(alloc, req_addr);



    /* Read request */

    req.type = QVIRTIO_BLK_T_IN;

    req.ioprio = 1;

    req.sector = 0;

    req.data = g_malloc0(512);

    strcpy(req.data, "TEST");



    req_addr = virtio_blk_request(alloc, &req, 512);



    g_free(req.data);



    indirect = qvring_indirect_desc_setup(&dev->vdev, alloc, 2);

    qvring_indirect_desc_add(indirect, req_addr, 16, false);

    qvring_indirect_desc_add(indirect, req_addr + 16, 513, true);

    free_head = qvirtqueue_add_indirect(&vqpci->vq, indirect);

    qvirtqueue_kick(&qvirtio_pci, &dev->vdev, &vqpci->vq, free_head);



    g_assert(qvirtio_wait_queue_isr(&qvirtio_pci, &dev->vdev, &vqpci->vq,

                                                        QVIRTIO_BLK_TIMEOUT));

    status = readb(req_addr + 528);

    g_assert_cmpint(status, ==, 0);



    data = g_malloc0(512);

    memread(req_addr + 16, data, 512);

    g_assert_cmpstr(data, ==, "TEST");

    g_free(data);



    g_free(indirect);

    guest_free(alloc, req_addr);



    /* End test */

    guest_free(alloc, vqpci->vq.desc);

    qvirtio_pci_device_disable(dev);

    g_free(dev);

    test_end();

}
