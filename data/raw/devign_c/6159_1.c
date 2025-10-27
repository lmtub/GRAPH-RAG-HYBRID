static void virtio_serial_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    VirtioDeviceClass *vdc = VIRTIO_DEVICE_CLASS(klass);

    dc->exit = virtio_serial_device_exit;

    dc->props = virtio_serial_properties;

    set_bit(DEVICE_CATEGORY_INPUT, dc->categories);

    vdc->init = virtio_serial_device_init;

    vdc->get_features = get_features;

    vdc->get_config = get_config;

    vdc->set_config = set_config;

    vdc->set_status = set_status;

    vdc->reset = vser_reset;

}
