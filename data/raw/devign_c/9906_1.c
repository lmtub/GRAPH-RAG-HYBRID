static void virtio_queue_host_notifier_read(EventNotifier *n)

{

    VirtQueue *vq = container_of(n, VirtQueue, host_notifier);

    if (event_notifier_test_and_clear(n)) {

        virtio_queue_notify_vq(vq);

    }

}
