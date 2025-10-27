static int virtio_ccw_set_vqs(SubchDev *sch, uint64_t addr, uint32_t align,

                              uint16_t index, uint16_t num)

{

    VirtIODevice *vdev = virtio_ccw_get_vdev(sch);



    if (index > VIRTIO_PCI_QUEUE_MAX) {

        return -EINVAL;

    }



    /* Current code in virtio.c relies on 4K alignment. */

    if (addr && (align != 4096)) {

        return -EINVAL;

    }



    if (!vdev) {

        return -EINVAL;

    }



    virtio_queue_set_addr(vdev, index, addr);

    if (!addr) {

        virtio_queue_set_vector(vdev, index, 0);

    } else {

        /* Fail if we don't have a big enough queue. */

        /* TODO: Add interface to handle vring.num changing */

        if (virtio_queue_get_num(vdev, index) > num) {

            return -EINVAL;

        }

        virtio_queue_set_vector(vdev, index, index);

    }

    /* tell notify handler in case of config change */

    vdev->config_vector = VIRTIO_PCI_QUEUE_MAX;

    return 0;

}
