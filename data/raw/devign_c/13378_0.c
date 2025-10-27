static void virtio_ccw_start_ioeventfd(VirtioCcwDevice *dev)

{

    VirtIODevice *vdev;

    int n, r;



    if (!(dev->flags & VIRTIO_CCW_FLAG_USE_IOEVENTFD) ||

        dev->ioeventfd_disabled ||

        dev->ioeventfd_started) {

        return;

    }

    vdev = virtio_bus_get_device(&dev->bus);

    for (n = 0; n < VIRTIO_PCI_QUEUE_MAX; n++) {

        if (!virtio_queue_get_num(vdev, n)) {

            continue;

        }

        r = virtio_ccw_set_guest2host_notifier(dev, n, true, true);

        if (r < 0) {

            goto assign_error;

        }

    }

    dev->ioeventfd_started = true;

    return;



  assign_error:

    while (--n >= 0) {

        if (!virtio_queue_get_num(vdev, n)) {

            continue;

        }

        r = virtio_ccw_set_guest2host_notifier(dev, n, false, false);

        assert(r >= 0);

    }

    dev->ioeventfd_started = false;

    /* Disable ioeventfd for this device. */

    dev->flags &= ~VIRTIO_CCW_FLAG_USE_IOEVENTFD;

    error_report("%s: failed. Fallback to userspace (slower).", __func__);

}
