static void virtio_net_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    VirtioDeviceClass *vdc = VIRTIO_DEVICE_CLASS(klass);

    dc->exit = virtio_net_device_exit;

    dc->props = virtio_net_properties;

    set_bit(DEVICE_CATEGORY_NETWORK, dc->categories);

    vdc->init = virtio_net_device_init;

    vdc->get_config = virtio_net_get_config;

    vdc->set_config = virtio_net_set_config;

    vdc->get_features = virtio_net_get_features;

    vdc->set_features = virtio_net_set_features;

    vdc->bad_features = virtio_net_bad_features;

    vdc->reset = virtio_net_reset;

    vdc->set_status = virtio_net_set_status;

    vdc->guest_notifier_mask = virtio_net_guest_notifier_mask;

    vdc->guest_notifier_pending = virtio_net_guest_notifier_pending;

}
