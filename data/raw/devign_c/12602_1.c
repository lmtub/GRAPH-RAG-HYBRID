static int virtio_pci_start_ioeventfd(VirtIOPCIProxy *proxy)

{

    int n, r;



    if (!(proxy->flags & VIRTIO_PCI_FLAG_USE_IOEVENTFD) ||

        proxy->ioeventfd_disabled ||

        proxy->ioeventfd_started) {

        return 0;

    }



    for (n = 0; n < VIRTIO_PCI_QUEUE_MAX; n++) {

        if (!virtio_queue_get_num(proxy->vdev, n)) {

            continue;

        }



        r = virtio_pci_set_host_notifier_internal(proxy, n, true);

        if (r < 0) {

            goto assign_error;

        }



        virtio_pci_set_host_notifier_fd_handler(proxy, n, true);

    }

    proxy->ioeventfd_started = true;

    return 0;



assign_error:

    while (--n >= 0) {

        if (!virtio_queue_get_num(proxy->vdev, n)) {

            continue;

        }



        virtio_pci_set_host_notifier_fd_handler(proxy, n, false);

        virtio_pci_set_host_notifier_internal(proxy, n, false);

    }

    proxy->ioeventfd_started = false;

    proxy->ioeventfd_disabled = true;

    return r;

}
