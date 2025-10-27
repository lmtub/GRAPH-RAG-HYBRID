static void virtio_9p_device_unrealize(DeviceState *dev, Error **errp)

{

    VirtIODevice *vdev = VIRTIO_DEVICE(dev);

    V9fsVirtioState *v = VIRTIO_9P(dev);

    V9fsState *s = &v->state;



    virtio_cleanup(vdev);

    v9fs_device_unrealize_common(s, errp);

}
