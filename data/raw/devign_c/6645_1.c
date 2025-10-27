static int usb_msd_initfn_storage(USBDevice *dev)

{

    MSDState *s = DO_UPCAST(MSDState, dev, dev);

    BlockDriverState *bs = s->conf.bs;

    SCSIDevice *scsi_dev;

    Error *err = NULL;



    if (!bs) {

        error_report("drive property not set");

        return -1;

    }



    blkconf_serial(&s->conf, &dev->serial);



    /*

     * Hack alert: this pretends to be a block device, but it's really

     * a SCSI bus that can serve only a single device, which it

     * creates automatically.  But first it needs to detach from its

     * blockdev, or else scsi_bus_legacy_add_drive() dies when it

     * attaches again.

     *

     * The hack is probably a bad idea.

     */

    bdrv_detach_dev(bs, &s->dev.qdev);

    s->conf.bs = NULL;



    usb_desc_create_serial(dev);

    usb_desc_init(dev);

    scsi_bus_new(&s->bus, sizeof(s->bus), DEVICE(dev),

                 &usb_msd_scsi_info_storage, NULL);

    scsi_dev = scsi_bus_legacy_add_drive(&s->bus, bs, 0, !!s->removable,

                                         s->conf.bootindex, dev->serial,

                                         &err);

    if (!scsi_dev) {



        return -1;

    }

    s->bus.qbus.allow_hotplug = 0;

    usb_msd_handle_reset(dev);



    if (bdrv_key_required(bs)) {

        if (cur_mon) {

            monitor_read_bdrv_key_start(cur_mon, bs, usb_msd_password_cb, s);

            s->dev.auto_attach = 0;

        } else {

            autostart = 0;

        }

    }



    return 0;

}