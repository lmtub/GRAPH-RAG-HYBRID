static int usb_msd_initfn(USBDevice *dev)

{

    MSDState *s = DO_UPCAST(MSDState, dev, dev);



    if (!s->conf.dinfo || !s->conf.dinfo->bdrv) {

        error_report("usb-msd: drive property not set");





    s->dev.speed = USB_SPEED_FULL;

    scsi_bus_new(&s->bus, &s->dev.qdev, 0, 1, usb_msd_command_complete);

    s->scsi_dev = scsi_bus_legacy_add_drive(&s->bus, s->conf.dinfo, 0);




    s->bus.qbus.allow_hotplug = 0;

    usb_msd_handle_reset(dev);



    if (bdrv_key_required(s->conf.dinfo->bdrv)) {

        if (cur_mon) {

            monitor_read_bdrv_key_start(cur_mon, s->conf.dinfo->bdrv,

                                        usb_msd_password_cb, s);

            s->dev.auto_attach = 0;

        } else {

            autostart = 0;





    return 0;
