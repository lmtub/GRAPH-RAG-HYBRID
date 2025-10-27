static void virtio_rng_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    VirtioDeviceClass *vdc = VIRTIO_DEVICE_CLASS(klass);



    dc->props = virtio_rng_properties;

    set_bit(DEVICE_CATEGORY_MISC, dc->categories);

    vdc->realize = virtio_rng_device_realize;

    vdc->unrealize = virtio_rng_device_unrealize;

    vdc->get_features = get_features;

    vdc->load = virtio_rng_load_device;

}
