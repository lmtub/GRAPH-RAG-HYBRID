static void virtio_balloon_to_target(void *opaque, ram_addr_t target,

                                     MonitorCompletion cb, void *cb_data)

{

    VirtIOBalloon *dev = opaque;



    if (target > ram_size) {

        target = ram_size;

    }

    if (target) {

        dev->num_pages = (ram_size - target) >> VIRTIO_BALLOON_PFN_SHIFT;

        virtio_notify_config(&dev->vdev);

    } else {

        virtio_balloon_stat(opaque, cb, cb_data);

    }

}
