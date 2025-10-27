static int virtio_ccw_set_vqs(SubchDev *sch, VqInfoBlock *info,

                              VqInfoBlockLegacy *linfo)

{

    VirtIODevice *vdev = virtio_ccw_get_vdev(sch);

    uint16_t index = info ? info->index : linfo->index;

    uint16_t num = info ? info->num : linfo->num;

    uint64_t desc = info ? info->desc : linfo->queue;



    if (index >= VIRTIO_CCW_QUEUE_MAX) {

        return -EINVAL;

    }



    /* Current code in virtio.c relies on 4K alignment. */

    if (linfo && desc && (linfo->align != 4096)) {

        return -EINVAL;

    }



    if (!vdev) {

        return -EINVAL;

    }



    if (info) {

        virtio_queue_set_rings(vdev, index, desc, info->avail, info->used);

    } else {

        virtio_queue_set_addr(vdev, index, desc);

    }

    if (!desc) {

        virtio_queue_set_vector(vdev, index, VIRTIO_NO_VECTOR);

    } else {

        if (info) {

            /* virtio-1 allows changing the ring size. */

            if (virtio_queue_get_max_num(vdev, index) < num) {

                /* Fail if we exceed the maximum number. */

                return -EINVAL;

            }

            virtio_queue_set_num(vdev, index, num);

        } else if (virtio_queue_get_num(vdev, index) > num) {

            /* Fail if we don't have a big enough queue. */

            return -EINVAL;

        }

        /* We ignore possible increased num for legacy for compatibility. */

        virtio_queue_set_vector(vdev, index, index);

    }

    /* tell notify handler in case of config change */

    vdev->config_vector = VIRTIO_CCW_QUEUE_MAX;

    return 0;

}
