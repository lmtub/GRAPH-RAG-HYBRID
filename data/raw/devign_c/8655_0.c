static int virtio_ccw_set_guest_notifier(VirtioCcwDevice *dev, int n,

                                         bool assign, bool with_irqfd)

{

    VirtIODevice *vdev = virtio_bus_get_device(&dev->bus);

    VirtQueue *vq = virtio_get_queue(vdev, n);

    EventNotifier *notifier = virtio_queue_get_guest_notifier(vq);

    VirtioDeviceClass *k = VIRTIO_DEVICE_GET_CLASS(vdev);



    if (assign) {

        int r = event_notifier_init(notifier, 0);



        if (r < 0) {

            return r;

        }

        virtio_queue_set_guest_notifier_fd_handler(vq, true, with_irqfd);

        if (with_irqfd) {

            r = virtio_ccw_add_irqfd(dev, n);

            if (r) {

                virtio_queue_set_guest_notifier_fd_handler(vq, false,

                                                           with_irqfd);

                return r;

            }

        }

        /*

         * We do not support individual masking for channel devices, so we

         * need to manually trigger any guest masking callbacks here.

         */

        if (k->guest_notifier_mask) {

            k->guest_notifier_mask(vdev, n, false);

        }

        /* get lost events and re-inject */

        if (k->guest_notifier_pending &&

            k->guest_notifier_pending(vdev, n)) {

            event_notifier_set(notifier);

        }

    } else {

        if (k->guest_notifier_mask) {

            k->guest_notifier_mask(vdev, n, true);

        }

        if (with_irqfd) {

            virtio_ccw_remove_irqfd(dev, n);

        }

        virtio_queue_set_guest_notifier_fd_handler(vq, false, with_irqfd);

        event_notifier_cleanup(notifier);

    }

    return 0;

}
