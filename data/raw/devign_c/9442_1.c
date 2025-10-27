static void syborg_virtio_writel(void *opaque, target_phys_addr_t offset,

                                 uint32_t value)

{

    SyborgVirtIOProxy *s = opaque;

    VirtIODevice *vdev = s->vdev;



    DPRINTF("writel 0x%x = 0x%x\n", (int)offset, value);

    if (offset >= SYBORG_VIRTIO_CONFIG) {

        return virtio_config_writel(vdev, offset - SYBORG_VIRTIO_CONFIG,

                                    value);

    }

    switch (offset >> 2) {

    case SYBORG_VIRTIO_GUEST_FEATURES:

        if (vdev->set_features)

            vdev->set_features(vdev, value);

        vdev->guest_features = value;

        break;

    case SYBORG_VIRTIO_QUEUE_BASE:

        if (value == 0)

            virtio_reset(vdev);

        else

            virtio_queue_set_addr(vdev, vdev->queue_sel, value);

        break;

    case SYBORG_VIRTIO_QUEUE_SEL:

        if (value < VIRTIO_PCI_QUEUE_MAX)

            vdev->queue_sel = value;

        break;

    case SYBORG_VIRTIO_QUEUE_NOTIFY:

        virtio_queue_notify(vdev, value);

        break;

    case SYBORG_VIRTIO_STATUS:

        virtio_set_status(vdev, value & 0xFF);

        if (vdev->status == 0)

            virtio_reset(vdev);

        break;

    case SYBORG_VIRTIO_INT_ENABLE:

        s->int_enable = value;

        virtio_update_irq(vdev);

        break;

    case SYBORG_VIRTIO_INT_STATUS:

        vdev->isr &= ~value;

        virtio_update_irq(vdev);

        break;

    default:

        BADF("Bad write offset 0x%x\n", (int)offset);

        break;

    }

}
