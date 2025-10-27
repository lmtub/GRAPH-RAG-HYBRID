static void mmio_basic(void)

{

    QVirtioMMIODevice *dev;

    QVirtQueue *vq;

    QGuestAllocator *alloc;

    int n_size = TEST_IMAGE_SIZE / 2;

    uint64_t capacity;



    arm_test_start();



    dev = qvirtio_mmio_init_device(MMIO_DEV_BASE_ADDR, MMIO_PAGE_SIZE);

    g_assert(dev != NULL);

    g_assert_cmphex(dev->vdev.device_type, ==, VIRTIO_ID_BLOCK);



    qvirtio_reset(&qvirtio_mmio, &dev->vdev);

    qvirtio_set_acknowledge(&qvirtio_mmio, &dev->vdev);

    qvirtio_set_driver(&qvirtio_mmio, &dev->vdev);



    alloc = generic_alloc_init(MMIO_RAM_ADDR, MMIO_RAM_SIZE, MMIO_PAGE_SIZE);

    vq = qvirtqueue_setup(&qvirtio_mmio, &dev->vdev, alloc, 0);



    test_basic(&qvirtio_mmio, &dev->vdev, alloc, vq,

                            QVIRTIO_MMIO_DEVICE_SPECIFIC);



    qmp("{ 'execute': 'block_resize', 'arguments': { 'device': 'drive0', "

                                                    " 'size': %d } }", n_size);



    qvirtio_wait_queue_isr(&qvirtio_mmio, &dev->vdev, vq,

                           QVIRTIO_BLK_TIMEOUT_US);



    capacity = qvirtio_config_readq(&qvirtio_mmio, &dev->vdev,

                                                QVIRTIO_MMIO_DEVICE_SPECIFIC);

    g_assert_cmpint(capacity, ==, n_size / 512);



    /* End test */

    guest_free(alloc, vq->desc);

    generic_alloc_uninit(alloc);

    g_free(dev);

    test_end();

}
