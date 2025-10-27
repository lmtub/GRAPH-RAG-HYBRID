static void virtio_ccw_bus_class_init(ObjectClass *klass, void *data)

{

    VirtioBusClass *k = VIRTIO_BUS_CLASS(klass);

    BusClass *bus_class = BUS_CLASS(klass);



    bus_class->max_dev = 1;

    k->notify = virtio_ccw_notify;

    k->vmstate_change = virtio_ccw_vmstate_change;

    k->query_guest_notifiers = virtio_ccw_query_guest_notifiers;

    k->set_guest_notifiers = virtio_ccw_set_guest_notifiers;

    k->save_queue = virtio_ccw_save_queue;

    k->load_queue = virtio_ccw_load_queue;

    k->save_config = virtio_ccw_save_config;

    k->load_config = virtio_ccw_load_config;

    k->device_plugged = virtio_ccw_device_plugged;

    k->post_plugged = virtio_ccw_post_plugged;

    k->device_unplugged = virtio_ccw_device_unplugged;

    k->ioeventfd_started = virtio_ccw_ioeventfd_started;

    k->ioeventfd_set_started = virtio_ccw_ioeventfd_set_started;

    k->ioeventfd_disabled = virtio_ccw_ioeventfd_disabled;

    k->ioeventfd_set_disabled = virtio_ccw_ioeventfd_set_disabled;

    k->ioeventfd_assign = virtio_ccw_ioeventfd_assign;

}
