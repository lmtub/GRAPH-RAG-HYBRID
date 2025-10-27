static void virtio_queue_guest_notifier_read(EventNotifier *n)

{

    VirtQueue *vq = container_of(n, VirtQueue, guest_notifier);

    if (event_notifier_test_and_clear(n)) {

        virtio_irq(vq);

    }

}
