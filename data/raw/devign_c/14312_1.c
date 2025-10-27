static int virtio_rng_device_exit(DeviceState *qdev)

{

    VirtIORNG *vrng = VIRTIO_RNG(qdev);

    VirtIODevice *vdev = VIRTIO_DEVICE(qdev);



    timer_del(vrng->rate_limit_timer);

    timer_free(vrng->rate_limit_timer);

    unregister_savevm(qdev, "virtio-rng", vrng);

    virtio_cleanup(vdev);

    return 0;

}
