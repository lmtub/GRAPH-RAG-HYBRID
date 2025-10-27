static void usb_msd_realize_bot(USBDevice *dev, Error **errp)

{

    MSDState *s = DO_UPCAST(MSDState, dev, dev);



    usb_desc_create_serial(dev);

    usb_desc_init(dev);

    scsi_bus_new(&s->bus, sizeof(s->bus), DEVICE(dev),

                 &usb_msd_scsi_info_bot, NULL);

    s->bus.qbus.allow_hotplug = 0;

    usb_msd_handle_reset(dev);

}
