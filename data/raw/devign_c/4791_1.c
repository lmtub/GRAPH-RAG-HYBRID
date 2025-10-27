static void usb_msd_class_initfn_bot(ObjectClass *klass, void *data)

{

    USBDeviceClass *uc = USB_DEVICE_CLASS(klass);



    uc->realize = usb_msd_realize_bot;


    uc->attached_settable = true;

}