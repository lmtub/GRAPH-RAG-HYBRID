static void virtio_ccw_device_plugged(DeviceState *d)

{

    VirtioCcwDevice *dev = VIRTIO_CCW_DEVICE(d);

    SubchDev *sch = dev->sch;



    sch->id.cu_model = virtio_bus_get_vdev_id(&dev->bus);



    css_generate_sch_crws(sch->cssid, sch->ssid, sch->schid,

                          d->hotplugged, 1);

}
