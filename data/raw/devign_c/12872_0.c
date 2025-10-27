static int virtio_pci_set_host_notifier(void *opaque, int n, bool assign)

{

    VirtIOPCIProxy *proxy = opaque;

    VirtQueue *vq = virtio_get_queue(proxy->vdev, n);

    EventNotifier *notifier = virtio_queue_get_host_notifier(vq);

    int r;

    if (assign) {

        r = event_notifier_init(notifier, 1);

        if (r < 0) {

            return r;

        }

        r = kvm_set_ioeventfd_pio_word(event_notifier_get_fd(notifier),

                                       proxy->addr + VIRTIO_PCI_QUEUE_NOTIFY,

                                       n, assign);

        if (r < 0) {

            event_notifier_cleanup(notifier);

        }

    } else {

        r = kvm_set_ioeventfd_pio_word(event_notifier_get_fd(notifier),

                                       proxy->addr + VIRTIO_PCI_QUEUE_NOTIFY,

                                       n, assign);

        if (r < 0) {

            return r;

        }

        event_notifier_cleanup(notifier);

    }

    return r;

}
