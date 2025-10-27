static int virtio_pci_set_host_notifier_internal(VirtIOPCIProxy *proxy,

                                                 int n, bool assign, bool set_handler)

{

    VirtIODevice *vdev = virtio_bus_get_device(&proxy->bus);

    VirtQueue *vq = virtio_get_queue(vdev, n);

    EventNotifier *notifier = virtio_queue_get_host_notifier(vq);

    int r = 0;



    if (assign) {

        r = event_notifier_init(notifier, 1);

        if (r < 0) {

            error_report("%s: unable to init event notifier: %d",

                         __func__, r);

            return r;

        }

        virtio_queue_set_host_notifier_fd_handler(vq, true, set_handler);

        memory_region_add_eventfd(&proxy->bar, VIRTIO_PCI_QUEUE_NOTIFY, 2,

                                  true, n, notifier);

    } else {

        memory_region_del_eventfd(&proxy->bar, VIRTIO_PCI_QUEUE_NOTIFY, 2,

                                  true, n, notifier);

        virtio_queue_set_host_notifier_fd_handler(vq, false, false);

        event_notifier_cleanup(notifier);

    }

    return r;

}
