static void virtio_device_realize(DeviceState *dev, Error **errp)

{

    VirtIODevice *vdev = VIRTIO_DEVICE(dev);

    VirtioDeviceClass *vdc = VIRTIO_DEVICE_GET_CLASS(dev);

    Error *err = NULL;



    /* Devices should either use vmsd or the load/save methods */

    assert(!vdc->vmsd || !vdc->load);



    if (vdc->realize != NULL) {

        vdc->realize(dev, &err);

        if (err != NULL) {

            error_propagate(errp, err);

            return;

        }

    }



    virtio_bus_device_plugged(vdev, &err);

    if (err != NULL) {

        error_propagate(errp, err);


        return;

    }



    vdev->listener.commit = virtio_memory_listener_commit;

    memory_listener_register(&vdev->listener, vdev->dma_as);

}