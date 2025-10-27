static int virtio_ccw_load_config(DeviceState *d, QEMUFile *f)

{

    VirtioCcwDevice *dev = VIRTIO_CCW_DEVICE(d);

    CcwDevice *ccw_dev = CCW_DEVICE(d);

    CCWDeviceClass *ck = CCW_DEVICE_GET_CLASS(ccw_dev);

    SubchDev *s = ccw_dev->sch;

    VirtIODevice *vdev = virtio_ccw_get_vdev(s);

    int len;



    s->driver_data = dev;

    subch_device_load(s, f);

    /* Re-fill subch_id after loading the subchannel states.*/

    if (ck->refill_ids) {

        ck->refill_ids(ccw_dev);

    }

    len = qemu_get_be32(f);

    if (len != 0) {

        dev->indicators = get_indicator(qemu_get_be64(f), len);

    } else {

        qemu_get_be64(f);

        dev->indicators = NULL;

    }

    len = qemu_get_be32(f);

    if (len != 0) {

        dev->indicators2 = get_indicator(qemu_get_be64(f), len);

    } else {

        qemu_get_be64(f);

        dev->indicators2 = NULL;

    }

    len = qemu_get_be32(f);

    if (len != 0) {

        dev->summary_indicator = get_indicator(qemu_get_be64(f), len);

    } else {

        qemu_get_be64(f);

        dev->summary_indicator = NULL;

    }

    qemu_get_be16s(f, &vdev->config_vector);

    dev->routes.adapter.ind_offset = qemu_get_be64(f);

    dev->thinint_isc = qemu_get_byte(f);

    dev->revision = qemu_get_be32(f);

    if (s->thinint_active) {

        return css_register_io_adapter(CSS_IO_ADAPTER_VIRTIO,

                                       dev->thinint_isc, true, false,

                                       &dev->routes.adapter.adapter_id);

    }



    return 0;

}
