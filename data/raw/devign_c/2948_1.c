static uint32_t virtio_ioport_read(VirtIOPCIProxy *proxy, uint32_t addr)

{

    VirtIODevice *vdev = proxy->vdev;

    uint32_t ret = 0xFFFFFFFF;



    switch (addr) {

    case VIRTIO_PCI_HOST_FEATURES:

        ret = vdev->get_features(vdev);

        ret |= (1 << VIRTIO_F_NOTIFY_ON_EMPTY);

        ret |= (1 << VIRTIO_RING_F_INDIRECT_DESC);

        ret |= (1 << VIRTIO_F_BAD_FEATURE);

        break;

    case VIRTIO_PCI_GUEST_FEATURES:

        ret = vdev->features;

        break;

    case VIRTIO_PCI_QUEUE_PFN:

        ret = virtio_queue_get_addr(vdev, vdev->queue_sel)

              >> VIRTIO_PCI_QUEUE_ADDR_SHIFT;

        break;

    case VIRTIO_PCI_QUEUE_NUM:

        ret = virtio_queue_get_num(vdev, vdev->queue_sel);

        break;

    case VIRTIO_PCI_QUEUE_SEL:

        ret = vdev->queue_sel;

        break;

    case VIRTIO_PCI_STATUS:

        ret = vdev->status;

        break;

    case VIRTIO_PCI_ISR:

        /* reading from the ISR also clears it. */

        ret = vdev->isr;

        vdev->isr = 0;

        qemu_set_irq(proxy->pci_dev.irq[0], 0);

        break;

    case VIRTIO_MSI_CONFIG_VECTOR:

        ret = vdev->config_vector;

        break;

    case VIRTIO_MSI_QUEUE_VECTOR:

        ret = virtio_queue_vector(vdev, vdev->queue_sel);

        break;

    default:

        break;

    }



    return ret;

}
