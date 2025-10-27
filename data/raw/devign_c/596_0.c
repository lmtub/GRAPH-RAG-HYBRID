static void virtio_9p_device_realize(DeviceState *dev, Error **errp)

{

    VirtIODevice *vdev = VIRTIO_DEVICE(dev);

    V9fsVirtioState *v = VIRTIO_9P(dev);

    V9fsState *s = &v->state;



    if (v9fs_device_realize_common(s, errp)) {

        goto out;

    }



    v->config_size = sizeof(struct virtio_9p_config) + strlen(s->fsconf.tag);

    virtio_init(vdev, "virtio-9p", VIRTIO_ID_9P, v->config_size);

    v->vq = virtio_add_queue(vdev, MAX_REQ, handle_9p_output);

    v9fs_register_transport(s, &virtio_9p_transport);



out:

    return;

}
