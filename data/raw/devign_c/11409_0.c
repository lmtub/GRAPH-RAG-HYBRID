static bool virtio_queue_host_notifier_aio_poll(void *opaque)

{

    EventNotifier *n = opaque;

    VirtQueue *vq = container_of(n, VirtQueue, host_notifier);

    bool progress;



    if (virtio_queue_empty(vq)) {

        return false;

    }



    progress = virtio_queue_notify_aio_vq(vq);



    /* In case the handler function re-enabled notifications */

    virtio_queue_set_notification(vq, 0);

    return progress;

}
