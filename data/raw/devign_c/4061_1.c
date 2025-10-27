static void virtio_balloon_device_realize(DeviceState *dev, Error **errp)

{

    VirtIODevice *vdev = VIRTIO_DEVICE(dev);

    VirtIOBalloon *s = VIRTIO_BALLOON(dev);

    int ret;



    virtio_init(vdev, "virtio-balloon", VIRTIO_ID_BALLOON,

                sizeof(struct virtio_balloon_config));



    ret = qemu_add_balloon_handler(virtio_balloon_to_target,

                                   virtio_balloon_stat, s);



    if (ret < 0) {

        error_setg(errp, "Adding balloon handler failed");

        virtio_cleanup(vdev);

        return;

    }



    s->ivq = virtio_add_queue(vdev, 128, virtio_balloon_handle_output);

    s->dvq = virtio_add_queue(vdev, 128, virtio_balloon_handle_output);

    s->svq = virtio_add_queue(vdev, 128, virtio_balloon_receive_stats);



    reset_stats(s);



    register_savevm(dev, "virtio-balloon", -1, 1,

                    virtio_balloon_save, virtio_balloon_load, s);



    object_property_add(OBJECT(dev), "guest-stats", "guest statistics",

                        balloon_stats_get_all, NULL, NULL, s, NULL);



    object_property_add(OBJECT(dev), "guest-stats-polling-interval", "int",

                        balloon_stats_get_poll_interval,

                        balloon_stats_set_poll_interval,

                        NULL, s, NULL);

}
