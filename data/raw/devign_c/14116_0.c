static void virtio_mmio_device_plugged(DeviceState *opaque)

{

    VirtIOMMIOProxy *proxy = VIRTIO_MMIO(opaque);



    proxy->host_features |= (0x1 << VIRTIO_F_NOTIFY_ON_EMPTY);

    proxy->host_features = virtio_bus_get_vdev_features(&proxy->bus,

                                                        proxy->host_features);

}
