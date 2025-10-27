static void virtio_ccw_device_realize(VirtioCcwDevice *dev, Error **errp)

{

    VirtIOCCWDeviceClass *k = VIRTIO_CCW_DEVICE_GET_CLASS(dev);

    CcwDevice *ccw_dev = CCW_DEVICE(dev);

    CCWDeviceClass *ck = CCW_DEVICE_GET_CLASS(ccw_dev);

    DeviceState *parent = DEVICE(ccw_dev);

    BusState *qbus = qdev_get_parent_bus(parent);

    VirtualCssBus *cbus = VIRTUAL_CSS_BUS(qbus);

    SubchDev *sch;

    Error *err = NULL;



    sch = css_create_sch(ccw_dev->devno, true, cbus->squash_mcss, errp);

    if (!sch) {

        return;

    }

    if (!virtio_ccw_rev_max(dev) && dev->force_revision_1) {

        error_setg(&err, "Invalid value of property max_rev "

                   "(is %d expected >= 1)", virtio_ccw_rev_max(dev));

        goto out_err;

    }



    sch->driver_data = dev;

    sch->ccw_cb = virtio_ccw_cb;

    sch->disable_cb = virtio_sch_disable_cb;

    sch->id.reserved = 0xff;

    sch->id.cu_type = VIRTIO_CCW_CU_TYPE;

    sch->do_subchannel_work = do_subchannel_work_virtual;

    ccw_dev->sch = sch;

    dev->indicators = NULL;

    dev->revision = -1;

    css_sch_build_virtual_schib(sch, 0, VIRTIO_CCW_CHPID_TYPE);



    trace_virtio_ccw_new_device(

        sch->cssid, sch->ssid, sch->schid, sch->devno,

        ccw_dev->devno.valid ? "user-configured" : "auto-configured");



    if (!kvm_eventfds_enabled()) {

        dev->flags &= ~VIRTIO_CCW_FLAG_USE_IOEVENTFD;

    }



    if (k->realize) {

        k->realize(dev, &err);

        if (err) {

            goto out_err;

        }

    }



    ck->realize(ccw_dev, &err);

    if (err) {

        goto out_err;

    }



    return;



out_err:

    error_propagate(errp, err);

    css_subch_assign(sch->cssid, sch->ssid, sch->schid, sch->devno, NULL);

    ccw_dev->sch = NULL;

    g_free(sch);

}
